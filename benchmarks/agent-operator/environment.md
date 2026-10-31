# Environment

Run on 30 and 31 October 2026.

| | |
|---|---|
| Machine | ASUS laptop, Windows 10 Home 22H2 |
| GPU | NVIDIA GeForce RTX 3080 Laptop, 8 GB, driver 621.01. It reached 86 C under sustained load and thermally throttled during the long runs, so only timings from the first, quiet pass are reported |
| CPU / RAM | AMD Ryzen 9 5900HS (8 cores, 16 threads), 39 GB |
| Runtime | llama.cpp `llama-server` build 11977 (commit 7ac5727d6), the WinGet package `ggml.llamacpp` (Vulkan and CPU) |
| Settings for every candidate | 16K context, `-fa on`, `-ctk q8_0 -ctv q8_0`, `--reasoning off`, `-np 1`, `-fitt 1024`; layer placement left to `--fit`. Mixture-of-experts models were also run with `--n-cpu-moe` set |
| Request | the one YBM sends: the recorded system and user prompt, the `OperatorDecision` JSON schema as `response_format`, temperature 0.1, 4096 max tokens, one attempt |
| Today's setup for comparison | LocalDeploy `qwen3vl_8b_ollama`: Ollama, `qwen3-vl:8b-instruct`, 16K context, default cache, 8 CPU threads |

## Model files

All are Q4_K_M-class quantizations from Unsloth's GGUF repositories on Hugging Face.

| Model | File | SHA-256 |
|---|---|---|
| Qwen3.5-9B | `Qwen3.5-9B-Q4_K_M.gguf` (5.29 GB) | `03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8` |
| Qwen3.5-4B | `Qwen3.5-4B-Q4_K_M.gguf` (2.55 GB) | `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4` |
| Qwen3-VL-8B Instruct | `Qwen3-VL-8B-Instruct-Q4_K_M.gguf` (4.68 GB) | `108e7ff92b78eefd3db4741885104acba514255c11b617d3c7b197a5f46efe89` |
| Qwen3.6-35B-A3B | `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (20.61 GB) | `ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61` |

Source repositories: `unsloth/Qwen3.5-9B-GGUF`, `unsloth/Qwen3.5-4B-GGUF`, `unsloth/Qwen3-VL-8B-Instruct-GGUF`,
`unsloth/Qwen3.6-35B-A3B-GGUF`. The other models in `summary.json` (Gemma 4 E4B, 12B and 26B-A4B, GLM-4.6V-Flash,
gpt-oss-20b, Qwen3.6-27B) came from `unsloth/gemma-4-E4B-it-GGUF`, `unsloth/gemma-4-12b-it-GGUF`,
`unsloth/gemma-4-26B-A4B-it-GGUF`, `unsloth/GLM-4.6V-Flash-GGUF`, `unsloth/gpt-oss-20b-GGUF` and
`unsloth/Qwen3.6-27B-GGUF`.

## The cases

599 Operator steps recorded by one YBM install (504 from August, 95 from late October), labeled by what each led to:
418 good, 181 bad. 120 of them (72 good and 48 bad, drawn at random with a fixed seed) are the shared first sample
every model was run on; the three small Qwen models and today's Qwen3-VL-8B were then run on all 599. The prompts contain the
owner's messages and file names and are not published; `summary.json` has scores and timings only.
