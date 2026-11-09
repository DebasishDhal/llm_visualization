import gc
import re
import time

import gradio as gr
import numpy as np
import plotly.graph_objects as go
import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_TOKENS = 48
MAX_FEATURES = 32
MAX_HEADS = 64
LAYER_RE = re.compile(r"(?:^|\.)(?:layers|h|blocks|block|layer)\.(\d+)(?:\.|$)")
PROJECTION_PATTERNS = {
    "q": re.compile(r"(?:q_proj|query|wq)$", re.I),
    "k": re.compile(r"(?:k_proj|key|wk)$", re.I),
    "v": re.compile(r"(?:v_proj|value|wv)$", re.I),
}


def _layer_index(name):
    match = LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def _first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _save_qkv(tensor, seq_len, n_heads):
    if tensor is None:
        return None
    value = tensor.detach().float()
    if value.ndim == 4:
        value = value[0]
        # Most implementations use [batch, heads, tokens, features].
        if value.shape[0] > value.shape[1] and value.shape[0] == seq_len:
            value = value.transpose(0, 1)
        value = value[:MAX_HEADS, :seq_len, :MAX_FEATURES].permute(1, 0, 2)
    else:
        if value.ndim == 3:
            value = value[0]
        if value.ndim != 2 or value.shape[0] != seq_len:
            return None
        heads = max(1, min(int(n_heads or 1), MAX_HEADS))
        if value.shape[-1] % heads == 0:
            value = value.reshape(seq_len, heads, -1)
        else:
            value = value[:, None, :]
        value = value[:, :MAX_HEADS, :MAX_FEATURES]
    return value.cpu().numpy().astype(np.float16).tolist()


def _norm_per_token(tensor, seq_len):
    if tensor is None:
        return None
    value = tensor.detach().float()
    if value.ndim >= 3:
        value = value[0]
    if value.ndim < 2 or value.shape[0] != seq_len:
        return None
    return torch.linalg.vector_norm(value.reshape(seq_len, -1), dim=-1).cpu().tolist()


def _pretty_token(token):
    return token.replace("Ġ", "␠").replace("▁", "␠").replace("\n", "↵")


def _safe_id(model_id):
    model_id = (model_id or "").strip()
    if not model_id or len(model_id) > 200 or any(part == ".." for part in model_id.split("/")):
        raise gr.Error("Enter a Hugging Face model ID, such as HuggingFaceTB/SmolLM2-360M-Instruct.")
    return model_id


@spaces.GPU(duration=180)
def analyze_model(model_id, prompt, access_token):
    """Load and inspect a causal language model on the Space's allocated ZeroGPU."""
    model_id = _safe_id(model_id)
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Enter a prompt to trace.")
    if len(prompt) > 1600:
        raise gr.Error("Keep prompts under 1,600 characters; traces are capped at 48 tokens.")

    token = (access_token or "").strip() or None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        model_kwargs = {"token": token, "dtype": "auto", "device_map": "auto"}
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, attn_implementation="eager", **model_kwargs
            )
        except (TypeError, ValueError):
            model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    except Exception as exc:
        raise gr.Error(f"Could not load {model_id}: {exc}") from exc

    model.eval()
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
    input_device = next(model.parameters()).device
    encoded = {key: value.to(input_device) for key, value in encoded.items()}
    token_ids = encoded["input_ids"][0].tolist()
    token_labels = [_pretty_token(tokenizer.convert_ids_to_tokens(i)) for i in token_ids]
    seq_len = len(token_ids)
    captures = {"qkv": {}, "ffn": {}, "residual": {}, "norm": {}}
    hooks = []
    n_heads = getattr(model.config, "num_attention_heads", 1) or 1
    n_kv_heads = getattr(model.config, "num_key_value_heads", n_heads) or n_heads

    def make_projection_hook(layer, kind, heads):
        def capture(_module, _inputs, output):
            value = _first_tensor(output)
            if value is None:
                return
            saved = _save_qkv(value, seq_len, heads)
            if saved is not None:
                captures["qkv"].setdefault(layer, {})[kind] = saved
        return capture

    def make_scalar_hook(layer, kind):
        def capture(_module, _inputs, output):
            values = _norm_per_token(_first_tensor(output), seq_len)
            if values is not None:
                captures[kind][layer] = values
        return capture

    def make_norm_hook(layer, module_name, module_type):
        def capture(_module, _inputs, output):
            value = _first_tensor(output)
            if value is None:
                return
            value = value.detach().float()
            if value.ndim >= 3:
                value = value[0]
            if value.ndim < 2 or value.shape[0] != seq_len:
                return
            rows = value.reshape(seq_len, -1)
            captures["norm"].setdefault(layer, []).append({
                "path": module_name,
                "type": module_type,
                "norms": torch.linalg.vector_norm(rows, dim=-1).cpu().tolist(),
                "features": rows[:, :MAX_FEATURES].cpu().numpy().astype(np.float16).tolist(),
            })
        return capture

    architecture_modules = {}
    norm_leaf = re.compile(r"(?:norm|layernorm|rmsnorm)$|^ln_?\d+$", re.I)
    attention_leaf = re.compile(r"(?:self_attn|attention|attn)$", re.I)
    for name, module in model.named_modules():
        layer = _layer_index(name)
        leaf = name.rsplit(".", 1)[-1]
        module_type = module.__class__.__name__
        if layer is None:
            if norm_leaf.search(leaf):
                hooks.append(module.register_forward_hook(make_norm_hook(-1, name, module_type)))
            continue
        info = architecture_modules.setdefault(layer, {"block": None, "attention": [], "ffn": [], "norms": []})
        if leaf.isdigit():
            info["block"] = name
            hooks.append(module.register_forward_hook(make_scalar_hook(layer, "residual")))
        if attention_leaf.search(leaf):
            info["attention"].append(name)
        if leaf.lower() in {"mlp", "feed_forward", "feedforward", "ffn"}:
            info["ffn"].append(name)
            hooks.append(module.register_forward_hook(make_scalar_hook(layer, "ffn")))
        if norm_leaf.search(leaf):
            info["norms"].append({"path": name, "type": module_type})
            hooks.append(module.register_forward_hook(make_norm_hook(layer, name, module_type)))

        matched = False
        for kind, pattern in PROJECTION_PATTERNS.items():
            if pattern.search(leaf):
                heads = n_heads if kind == "q" else n_kv_heads
                hooks.append(module.register_forward_hook(make_projection_hook(layer, kind, heads)))
                matched = True
                break
        if not matched and re.search(r"(?:c_attn|query_key_value|qkv_proj|Wqkv)$", leaf, re.I):
            def capture_fused(_module, _inputs, output, layer=layer):
                value = _first_tensor(output)
                if value is None or value.shape[-1] % 3:
                    return
                q, k, v = value.chunk(3, dim=-1)
                for kind, part, heads in (("q", q, n_heads), ("k", k, n_kv_heads), ("v", v, n_kv_heads)):
                    saved = _save_qkv(part, seq_len, heads)
                    if saved is not None:
                        captures["qkv"].setdefault(layer, {})[kind] = saved
            hooks.append(module.register_forward_hook(capture_fused))

    started = time.time()
    try:
        with torch.inference_mode():
            outputs = model(
                **encoded,
                output_attentions=True,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        attentions = outputs.attentions
        hidden_states = outputs.hidden_states
        if not attentions or any(attention is None for attention in attentions):
            raise gr.Error("This model did not return attention weights. Its architecture or attention backend may not expose them through Transformers.")

        attention_data = [a[0, :MAX_HEADS, :seq_len, :seq_len].float().cpu().numpy().astype(np.float16).tolist() for a in attentions]
        hidden_preview = []
        hidden_norms = []
        for hidden in hidden_states:
            row = hidden[0, :seq_len].detach().float()
            hidden_norms.append(torch.linalg.vector_norm(row, dim=-1).cpu().tolist())
            hidden_preview.append(row[:, :MAX_FEATURES].cpu().numpy().astype(np.float16).tolist())

        probs = torch.softmax(outputs.logits[0, -1].float(), dim=-1)
        top = torch.topk(probs, k=min(10, probs.numel()))
        logits = [{"token": _pretty_token(tokenizer.convert_ids_to_tokens(int(idx))), "probability": float(prob)}
                  for prob, idx in zip(top.values.cpu(), top.indices.cpu())]
        layer_count = len(attention_data)
        architecture = []
        for i in range(layer_count):
            info = architecture_modules.get(i, {})
            architecture.append({
                "layer": i,
                "block": info.get("block") or f"Transformer block {i}",
                "attention": info.get("attention", []),
                "ffn": info.get("ffn", []),
                "norms": info.get("norms", []),
            })
        trace = {
            "model": model_id,
            "tokens": token_labels,
            "attentions": attention_data,
            "hidden": hidden_preview,
            "hidden_norms": hidden_norms,
            "qkv": captures["qkv"],
            "ffn_norms": [captures["ffn"].get(i) for i in range(layer_count)],
            "residual_norms": [captures["residual"].get(i) for i in range(layer_count)],
            "norm_layers": {str(i): entries for i, entries in captures["norm"].items()},
            "architecture": architecture,
            "logits": logits,
            "elapsed": time.time() - started,
            "n_heads": int(attention_data[0].__len__()),
        }
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"Trace failed for this architecture: {exc}") from exc
    finally:
        for hook in hooks:
            hook.remove()
        try:
            del model, outputs
        except UnboundLocalError:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return trace


def _empty_plot(message):
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font={"color": "#87938f", "size": 13})
    fig.update_layout(template="plotly_white", height=300, margin={"l": 25, "r": 20, "t": 30, "b": 25}, xaxis={"visible": False}, yaxis={"visible": False})
    return fig


def _layout(fig, height=330):
    fig.update_layout(template="plotly_white", height=height, margin={"l": 45, "r": 25, "t": 45, "b": 45}, font={"family": "Inter, sans-serif", "color": "#26332e"}, paper_bgcolor="white", plot_bgcolor="white")
    return fig


def render_trace(trace, layer=0, head=0, token_index=0):
    if not trace:
        empty = _empty_plot("Run a trace to inspect model internals")
        return "Choose a model and prompt, then click **Trace on ZeroGPU**.", empty, empty, empty, empty, empty

    tokens = trace["tokens"]
    layer = max(0, min(int(layer or 0), len(trace["attentions"]) - 1))
    token_index = max(0, min(int(token_index or 0), len(tokens) - 1))
    attention = trace["attentions"][layer]
    head = max(0, min(int(head or 0), len(attention) - 1))
    matrix = attention[head]
    selected = tokens[token_index]

    attn_fig = go.Figure(go.Heatmap(z=matrix, x=tokens, y=tokens, colorscale="YlGnBu", zmin=0, zmax=1, colorbar={"title": "weight"}, hovertemplate="query %{y}<br>key %{x}<br>weight %{z:.3f}<extra></extra>"))
    attn_fig.add_hline(y=selected, line_color="#d98248", line_width=1.5, opacity=0.9)
    attn_fig.update_layout(title=f"Layer {layer} · Head {head} · attention to/from {selected}", xaxis_title="Key token", yaxis_title="Query token", yaxis={"autorange": "reversed"})
    _layout(attn_fig, max(300, min(520, 170 + 21 * len(tokens))))

    projection = trace["qkv"].get(str(layer), trace["qkv"].get(layer, {}))
    qkv_fig = go.Figure()
    colors = {"q": "#4b9b68", "k": "#5786ba", "v": "#d28b48"}
    for name in ("q", "k", "v"):
        values = projection.get(name)
        if values and token_index < len(values):
            token_values = values[token_index]
            if token_values and isinstance(token_values[0], list):
                token_values = token_values[min(head, len(token_values) - 1)]
            qkv_fig.add_trace(go.Scatter(y=token_values, mode="lines+markers", name=name.upper(), line={"color": colors[name], "width": 1.5}, marker={"size": 4}))
    if not qkv_fig.data:
        qkv_fig = _empty_plot("Q/K/V projections were not exposed by this model architecture")
    else:
        qkv_fig.update_layout(title=f"Q · K · V vectors for {selected}", xaxis_title="Feature dimension", yaxis_title="Value", legend={"orientation": "h", "y": 1.12})
        _layout(qkv_fig, 330)

    hidden = trace["hidden"][min(layer + 1, len(trace["hidden"]) - 1)][token_index]
    ffn_norm = trace["ffn_norms"][layer]
    residual_norm = trace["residual_norms"][layer]
    acts_fig = go.Figure()
    acts_fig.add_trace(go.Bar(x=list(range(len(hidden))), y=hidden, marker_color="#6cae78", name="Hidden state"))
    acts_fig.update_layout(title=f"Layer {layer} · hidden-state features for {selected}", xaxis_title="Feature dimension (first 32)", yaxis_title="Activation")
    _layout(acts_fig, 330)
    if ffn_norm and layer < len(ffn_norm) and residual_norm and layer < len(residual_norm):
        acts_fig.add_annotation(text=f"FFN norm {ffn_norm[token_index]:.2f}  ·  residual stream norm {residual_norm[token_index]:.2f}", x=0.5, y=1.12, xref="paper", yref="paper", showarrow=False, font={"size": 10, "color": "#75817c"})

    path_fig = go.Figure()
    path_fig.add_trace(go.Scatter(x=list(range(len(trace["hidden_norms"]))), y=[row[token_index] for row in trace["hidden_norms"]], mode="lines+markers", name="Residual stream", line={"color": "#388b59", "width": 2}))
    ffn_points = [row[token_index] if row and token_index < len(row) else None for row in trace["ffn_norms"]]
    if any(value is not None for value in ffn_points):
        path_fig.add_trace(go.Scatter(x=list(range(1, len(ffn_points) + 1)), y=ffn_points, mode="lines+markers", name="FFN output norm", line={"color": "#d28b48", "width": 1.5}))
    path_fig.update_layout(title=f"How {selected} changes through the stack", xaxis_title="Layer (0 = embedding)", yaxis_title="Vector L2 norm", legend={"orientation": "h", "y": 1.12})
    _layout(path_fig, 300)

    logits = trace["logits"]
    logits_fig = go.Figure(go.Bar(x=[item["probability"] for item in logits], y=[item["token"] for item in logits], orientation="h", marker_color="#76b979", hovertemplate="%{y}: %{x:.2%}<extra></extra>"))
    logits_fig.update_layout(title="Next-token distribution", xaxis_title="Probability", yaxis={"autorange": "reversed"}, xaxis={"tickformat": ".0%"})
    _layout(logits_fig, 330)
    summary = f"**{trace['model']}** · {len(trace['attentions'])} layers · {trace['n_heads']} heads · {len(tokens)} tokens · {trace['elapsed']:.1f}s on ZeroGPU  \nSelected token: `{selected}` (position {token_index})"
    return summary, attn_fig, qkv_fig, acts_fig, path_fig, logits_fig


def run_trace(model_id, prompt, access_token):
    trace = analyze_model(model_id, prompt, access_token)
    layer = len(trace["attentions"]) - 1
    head = 0
    token_index = len(trace["tokens"]) - 1
    summary, attn, qkv, acts, path, logits = render_trace(trace, layer, head, token_index)
    return (
        trace,
        gr.Slider(minimum=0, maximum=max(1, len(trace["attentions"]) - 1), value=layer, step=1, label="Layer"),
        gr.Slider(minimum=0, maximum=max(1, trace["n_heads"] - 1), value=head, step=1, label="Attention head"),
        gr.Slider(minimum=0, maximum=max(1, len(trace["tokens"]) - 1), value=token_index, step=1, label="Selected token"),
        summary, attn, qkv, acts, path, logits,
    )


def render_layer_tab(trace, layer=0, norm_name=None, token_index=0):
    empty = _empty_plot("Run a trace to inspect the model structure")
    if not trace:
        return gr.Slider(0, 1, value=0, step=1, label="Layer (last position is final norm)"), gr.Dropdown(choices=[], label="Normalization module"), [], empty, empty, "Run a trace to discover the model's layer and normalization modules."

    architecture = trace["architecture"]
    layer_count = len(architecture)
    layer = max(0, min(int(layer or 0), layer_count))
    token_index = max(0, min(int(token_index or 0), len(trace["tokens"]) - 1))
    final_norm = layer == layer_count
    norm_key = "-1" if final_norm else str(layer)
    label = "Final normalization" if final_norm else f"Layer {layer}"
    entries = trace["norm_layers"].get(norm_key, [])
    choices = [entry["path"] for entry in entries]
    if norm_name not in choices:
        norm_name = choices[0] if choices else None
    selected = next((entry for entry in entries if entry["path"] == norm_name), None)

    x = [str(row["layer"]) for row in architecture] + ["Final"]
    attention_counts = [1 if row["attention"] else 0 for row in architecture] + [0]
    ffn_counts = [len(row["ffn"]) for row in architecture] + [0]
    norm_counts = [len(row["norms"]) for row in architecture] + [len(trace["norm_layers"].get("-1", []))]
    architecture_fig = go.Figure()
    architecture_fig.add_trace(go.Bar(x=x, y=attention_counts, name="Attention weights", marker_color="#5a9f70"))
    architecture_fig.add_trace(go.Bar(x=x, y=ffn_counts, name="FFN modules", marker_color="#d39454"))
    architecture_fig.add_trace(go.Bar(x=x, y=norm_counts, name="Norm modules", marker_color="#628bb8"))
    architecture_fig.update_layout(barmode="group", title="Modules detected in each transformer block", xaxis_title="Layer index", yaxis_title="Module count", legend={"orientation": "h", "y": 1.15})
    _layout(architecture_fig, 330)

    rows = []
    for spec in architecture:
        rows.append([
            str(spec["layer"]), spec["block"], ", ".join(spec["attention"]) or "Attention weights returned",
            ", ".join(spec["ffn"]) or "Not detected", ", ".join(f"{n['path']} ({n['type']})" for n in spec["norms"]) or "Not detected",
        ])
    final_entries = trace["norm_layers"].get("-1", [])
    rows.append(["Final", "Final model output", "—", "—", ", ".join(f"{n['path']} ({n['type']})" for n in final_entries) or "Not detected"])

    if selected:
        values = selected["features"]
        norm_fig = go.Figure(go.Heatmap(z=values, x=list(range(len(values[0]))), y=trace["tokens"], colorscale="RdBu", zmid=0, colorbar={"title": "activation"}, hovertemplate="token %{y}<br>feature %{x}<br>value %{z:.3f}<extra></extra>"))
        norm_fig.update_layout(title=f"{selected['type']} output · {selected['path']}", xaxis_title="Feature dimension (first 32)", yaxis_title="Token", yaxis={"autorange": "reversed"})
        _layout(norm_fig, max(300, min(480, 170 + 20 * len(trace["tokens"]))))
        norm_value = selected["norms"][token_index]
        detail = f"**{label}** · `{selected['path']}` · {selected['type']} · selected token norm: {norm_value:.3f}"
    else:
        norm_fig = _empty_plot("No normalization output was captured for this layer")
        detail = f"**{label}** · No recognizable normalization module was exposed by this architecture. The structure table shows the module paths detected."

    slider = gr.Slider(minimum=0, maximum=max(1, layer_count), value=layer, step=1, label="Layer (last position is final norm)")
    dropdown = gr.Dropdown(choices=choices, value=norm_name, label="Normalization module")
    return slider, dropdown, rows, architecture_fig, norm_fig, detail


CSS = """
.gradio-container{max-width:1440px!important;font-family:Inter,ui-sans-serif,system-ui,sans-serif!important}
#hero{padding:24px 26px;border:1px solid #e5ece7;border-radius:18px;background:linear-gradient(120deg,#f5faf5,#fff 65%);margin-bottom:16px}
#hero h1{font-size:34px;letter-spacing:-1.2px;margin:0 0 8px;color:#18372a}
#hero p{color:#67766d;margin:0;max-width:760px}
.panel{border:1px solid #e6ece8;border-radius:14px;padding:12px;background:#fff}
footer{opacity:.75}
"""

with gr.Blocks(title="TraceSpace — Transformer Explorer") as demo:
    gr.Markdown("""
    # TraceSpace · Transformer Explorer
    Follow prompt tokens through attention heads, Q/K/V projections, FFN activations, residual streams, hidden states, and next-token logits.

    **Model execution happens on Hugging Face ZeroGPU.** Public Transformers causal language models are supported when their architecture returns attention weights; model-specific projection visualizations depend on exposed Q/K/V modules.
    """, elem_id="hero")

    with gr.Row():
        model_id = gr.Textbox(value="HuggingFaceTB/SmolLM2-360M-Instruct", label="Hugging Face model ID", placeholder="organization/model-name", scale=3)
        access_token = gr.Textbox(type="password", label="HF token (optional; gated models)", placeholder="hf_…", scale=2)
    prompt = gr.Textbox(value="The capital of France is", label="Prompt", lines=3, max_lines=6, placeholder="Enter text to trace through the model…")
    trace_button = gr.Button("Trace on ZeroGPU", variant="primary", size="lg")
    trace_state = gr.State(None)

    with gr.Tabs():
        with gr.Tab("Token Trace"):
            summary = gr.Markdown("Choose a model and prompt, then click **Trace on ZeroGPU**.")
            with gr.Row():
                layer_slider = gr.Slider(0, 1, value=0, step=1, label="Layer", scale=2)
                head_slider = gr.Slider(0, 1, value=0, step=1, label="Attention head", scale=1)
                token_slider = gr.Slider(0, 1, value=0, step=1, label="Selected token", scale=2)
            with gr.Row():
                with gr.Column(scale=1, elem_classes="panel"):
                    attention_plot = gr.Plot(label="Attention map")
                with gr.Column(scale=1, elem_classes="panel"):
                    qkv_plot = gr.Plot(label="Q/K/V vectors")
            with gr.Row():
                with gr.Column(scale=1, elem_classes="panel"):
                    activation_plot = gr.Plot(label="Hidden state and FFN/residual signals")
                with gr.Column(scale=1, elem_classes="panel"):
                    path_plot = gr.Plot(label="Token journey through layers")
            with gr.Row():
                with gr.Column(scale=1, elem_classes="panel"):
                    logits_plot = gr.Plot(label="Final token probabilities")
            gr.Markdown("""
            **Notes:** A trace runs the prompt once and reads its next-token distribution; it does not generate a long completion. Input is capped at 48 tokens to keep attention capture manageable. Very large models may exceed the available ZeroGPU memory. Gated models require an access token. Custom model code is not enabled (`trust_remote_code=False`).
            """)

        with gr.Tab("Layers & Norms"):
            gr.Markdown("Inspect the detected transformer block layout, then select a normalization module to view its output across tokens and feature dimensions.")
            with gr.Row():
                layer_detail_slider = gr.Slider(0, 1, value=0, step=1, label="Layer (last position is final norm)", scale=1)
                norm_selector = gr.Dropdown(choices=[], label="Normalization module", scale=3)
            layer_detail_summary = gr.Markdown("Run a trace to discover the model's layer and normalization modules.")
            with gr.Row():
                with gr.Column(scale=1, elem_classes="panel"):
                    architecture_plot = gr.Plot(label="Transformer block map")
                with gr.Column(scale=1, elem_classes="panel"):
                    norm_plot = gr.Plot(label="Normalization output")
            layer_table = gr.Dataframe(
                headers=["Layer", "Block module", "Attention module paths", "FFN module paths", "Normalization modules"],
                datatype=["str", "str", "str", "str", "str"],
                interactive=False,
                label="Detected model structure",
            )

    trace_outputs = [trace_state, layer_slider, head_slider, token_slider, summary, attention_plot, qkv_plot, activation_plot, path_plot, logits_plot]
    trace_button.click(run_trace, inputs=[model_id, prompt, access_token], outputs=trace_outputs)
    view_outputs = [summary, attention_plot, qkv_plot, activation_plot, path_plot, logits_plot]
    view_inputs = [trace_state, layer_slider, head_slider, token_slider]
    for control in (layer_slider, head_slider, token_slider):
        control.change(render_trace, inputs=view_inputs, outputs=view_outputs, show_progress="hidden")

    layer_tab_outputs = [layer_detail_slider, norm_selector, layer_table, architecture_plot, norm_plot, layer_detail_summary]
    layer_tab_inputs = [trace_state, layer_detail_slider, norm_selector, token_slider]
    trace_state.change(render_layer_tab, inputs=layer_tab_inputs, outputs=layer_tab_outputs, show_progress="hidden")
    layer_detail_slider.change(render_layer_tab, inputs=layer_tab_inputs, outputs=layer_tab_outputs, show_progress="hidden")
    norm_selector.change(render_layer_tab, inputs=layer_tab_inputs, outputs=layer_tab_outputs, show_progress="hidden")


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="green", neutral_hue="slate"), css=CSS)


