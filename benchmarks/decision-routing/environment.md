# Environment

All runs: one laptop, October 28 and 29, 2026, plugged in, Windows power plan "Turbo", nothing else heavy running
except where a run says so (the under-load test).

| Part | Details |
|---|---|
| CPU | AMD Ryzen 9 5900HS, 8 cores, 16 threads; decision models used 8 threads unless a file name says otherwise |
| RAM | 40 GB (39 GiB visible to Windows) |
| GPU | NVIDIA GeForce RTX 3080 Laptop GPU, 8 GB, driver 621.01 |
| OS | Windows 10 Home 10.0.19045 |
| llama.cpp | b11976 (`version: 0.8.0-dev (build 11976, commit 2e0d86beb)`), the official Windows release zips: `llama-b11976-bin-win-cpu-x64.zip` for the CPU and `llama-b11976-bin-win-cuda-12.4-x64.zip` with `cudart-llama-bin-win-cuda-12.4-x64.zip` for the GPU |
| llama-server flags | `-c 4096 -b 4096 -ub 4096 -np 1 -t 8`, `-ngl 0` on the CPU and `-ngl 99` on the GPU |
| Python | 3.12.13, `scripts/decision_bench.py` (requests only) |
| LLM | Qwen3-VL 8B Instruct (`qwen3-vl:8b-instruct`) in Ollama 0.32.14 with a 16K context, through LocalDeploy, asked by YBM's own `LLMMessageClassifier` with no conversation context and no cloud fallback. It needs 8.1 GB, so Ollama placed 68% of it on the GPU and 32% on the CPU (`ollama ps`). |

## Model files

`made` = produced from the repository's published BF16 GGUF with `llama-quantize` from the same llama.cpp build
(LocalDeploy's `"quantize"` profile field, or `scripts/make_decision_gguf.py`). Making the d3-lite file again on first
use through LocalDeploy gave a file with the same SHA-256 as the one benchmarked.

| File | Source | License | Size | SHA-256 |
|---|---|---|---:|---|
| `vllm-d3-lite-Q4_K_M.gguf` | made from [ggml-org/vllm-d3-lite-GGUF](https://huggingface.co/ggml-org/vllm-d3-lite-GGUF) | Apache-2.0 | 641 MiB | `8a00c23cdb17f4333f305e7ff276344fa0f4b93f2f8eb6109496044801c2b166` |
| `vllm-d3-lite-Q6_K.gguf` | made from the same | Apache-2.0 | 799 MiB | `e0e53954875ac62ac55dadad5c6fe14d0874299463c1c98c52c5abb50e6c7d29` |
| `vllm-d3-lite-Q8_0.gguf` | published, same repository | Apache-2.0 | 1,032 MiB | `ffed04a9607034d641b95181d79abf36bec3a03ed647949bad8277d0e2984ddf` |
| `vllm-d3-nano-Q4_K_M.gguf` | made from [ggml-org/vllm-d3-nano-GGUF](https://huggingface.co/ggml-org/vllm-d3-nano-GGUF) | Apache-2.0 | 1,488 MiB | `c6c18d647fa4cc70d19189cb3a559a47475d28d148b8806da5fa1e6f61d4fb96` |
| `vllm-d3-mini-Q4_K_M.gguf` | published, [ggml-org/vllm-d3-mini-GGUF](https://huggingface.co/ggml-org/vllm-d3-mini-GGUF) | Apache-2.0 | 3,536 MiB | `ab3d4378c268f7f3238a63607af40c0cb516e4482278b01c2f48e9d0698ce522` |
| `d1-3B-Q4_K_M.gguf` | published, [LiquidAI/d1-3B-GGUF](https://huggingface.co/LiquidAI/d1-3B-GGUF) | LFM Open License v1.0 | 1,597 MiB | `16aff27ea2eefdc32b9897f43854a5d3170c1dc8dccb9c756905af30a4e22402` |

The other files in `runs/2026-10-29-comparison` (Julia-1, Laya, d1-omni-600M, d3-edge, Kev, lev and the remaining
quantizations) were deleted after that run to free disk space; their names and sources are listed in
[docs/MODELS.md](../../docs/MODELS.md#decision-models).
