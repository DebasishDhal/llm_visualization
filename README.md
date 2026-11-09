---
title: TraceSpace
emoji: 🔬
colorFrom: green
colorTo: blue
sdk: gradio
app_file: app.py
python_version: 3.12.12
hardware: zero-a10g
short_description: Explore transformer internals on ZeroGPU
---

# TraceSpace — Transformer Explorer

A Gradio Space for tracing a prompt through Hugging Face causal language models. It visualizes attention maps, Q/K/V vectors when the model exposes recognizable projection layers, hidden states, FFN and residual-stream norms, and the next-token distribution.

## Hugging Face Space setup

This repository is configured for the Gradio SDK and ZeroGPU. Select **ZeroGPU** in the Space hardware settings if the hardware is not assigned automatically. The app wraps each model trace in `@spaces.GPU`; model loading and inference happen on the Space, never on the user's computer.

## Model support

Enter a public Transformers causal language model ID. Gated models require an HF read token in the UI. The app sets `trust_remote_code=False`, so models that require arbitrary custom Python code are not loaded. Attention weights must be supported by the model's Transformers implementation. Q/K/V projection charts appear for common named projection modules; fused or unusual architectures may not expose separate Q/K/V outputs. Input traces are limited to 48 tokens to bound attention memory. Large models can exceed ZeroGPU's available memory.

The app runs one forward pass over the prompt. Its final chart shows next-token probabilities; it does not generate a completion.

## Files

- `app.py` — Gradio Blocks UI, ZeroGPU inference, and interactive plots.
- `requirements.txt` — additional runtime dependencies. Gradio, PyTorch, and `spaces` are provided by the Hugging Face Space runtime.


