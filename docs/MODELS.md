# Models

This page records one LocalDeploy benchmark run and explains the reference profiles in `config.example.json`. It is not a universal ranking. Model tags, runtime behavior, and quantizations change, and results vary by hardware and prompt set. Run the same suite on your own machine before choosing a default.

## Local benchmark from July 2026

Hardware: RTX 3080 Laptop GPU with 8 GB VRAM on Windows. Runtime: Ollama. Workload: 17 profiles, 25 tests, and six categories. Peak VRAM includes roughly 1.8 GB used by the desktop and other processes.

The benchmark disabled Qwen thinking output so the final `content` field could be graded consistently. That choice hurts the math score of reasoning models and should be kept in mind when reading the table.

| # | Profile | Model | Tests | Overall | Plan | Class | Code | Math | JSON | Hard JSON | Avg time | Peak VRAM |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `qwen3vl_8b_ollama` | `qwen3-vl:8b-instruct` | 25 | 0.84 | 0.93 | 1.00 | 0.93 | 0.40 | 1.00 | 0.88 | 13.2s | 7372 MB |
| 2 | `qwen25_7b_q5km_ollama` | `qwen2.5:7b-instruct-q5_K_M` | 25 | 0.79 | 0.89 | 0.60 | 0.93 | 0.60 | 1.00 | 0.86 | 7.0s | 7108 MB |
| 2 | `gemma3_12b_qat_ollama` | `gemma3:12b-it-qat` | 25 | 0.79 | 0.87 | 0.80 | 0.93 | 0.40 | 1.00 | 0.86 | 34.9s | 7460 MB |
| 2 | `qwen25_7b_ollama` | `qwen2.5:7b` | 25 | 0.79 | 0.86 | 0.60 | 0.93 | 0.60 | 1.00 | 0.86 | 5.4s | 6323 MB |
| 5 | `qwen3vl_4b_ollama` | `qwen3-vl:4b-instruct` | 25 | 0.78 | 0.92 | 0.80 | 0.93 | 0.40 | 1.00 | 0.75 | 6.3s | 7350 MB |
| 6 | `gemma3_4b_ollama_safe` | `gemma3:4b` | 25 | 0.77 | 0.73 | 1.00 | 0.93 | 0.20 | 1.00 | 0.84 | 5.6s | 6184 MB |
| 6 | `gemma3_12b_ollama_safe` | `gemma3:12b` | 25 | 0.77 | 0.73 | 0.80 | 0.93 | 0.40 | 1.00 | 0.82 | 37.1s | 7391 MB |
| 8 | `qwen25_coder_7b_ollama` | `qwen2.5-coder:7b` | 25 | 0.75 | 0.87 | 0.40 | 0.93 | 0.60 | 1.00 | 0.86 | 5.4s | 6538 MB |
| 9 | `qwen25vl_7b_ollama` | `qwen2.5vl:7b` | 25 | 0.72 | 0.89 | 0.60 | 0.93 | 0.40 | 1.00 | 0.68 | 31.1s | 7299 MB |
| 9 | `qwen3_8b_ollama` | `qwen3:8b` | 25 | 0.72 | 0.64 | 1.00 | 0.88 | 0.00 | 1.00 | 0.91 | 7.9s | 7097 MB |
| 11 | `qwen25vl_3b_ollama` | `qwen2.5vl:3b` | 25 | 0.70 | 0.88 | 0.80 | 0.90 | 0.20 | 0.79 | 0.80 | 19.1s | 1567 MB |
| 12 | `granite33_8b_ollama` | `granite3.3:8b` | 25 | 0.67 | 0.93 | 0.60 | 0.95 | 0.00 | 1.00 | 0.80 | 11.4s | 6714 MB |
| 13 | `llama31_8b_ollama` | `llama3.1:8b` | 25 | 0.66 | 0.72 | 0.60 | 0.93 | 0.00 | 1.00 | 0.89 | 8.0s | 7484 MB |
| 14 | `llama32_3b_ollama` | `llama3.2:3b` | 25 | 0.64 | 0.93 | 0.60 | 0.95 | 0.00 | 0.96 | 0.66 | 4.0s | 6770 MB |
| 15 | `mistral_7b_ollama` | `mistral:7b` | 25 | 0.60 | 0.90 | 0.40 | 0.84 | 0.00 | 1.00 | 0.73 | 6.7s | 7435 MB |
| 16 | `phi4_mini_ollama` | `phi4-mini` | 25 | 0.58 | 0.65 | 0.20 | 0.88 | 0.20 | 0.95 | 0.80 | 5.5s | 7185 MB |
| 17 | `deepseek_r1_distill_qwen_7b_ollama` | `deepseek-r1:7b` | 25 | 0.51 | 0.73 | 0.20 | 1.00 | 0.00 | 0.65 | 0.76 | 6.9s | 6943 MB |

The useful conclusions from this run are fairly narrow:

- `qwen3-vl:8b-instruct` had the highest score across this particular mixed suite.
- `qwen3:8b` and `llama3.1:8b` did well on the harder structured-output tests even though their overall scores were lower.
- `qwen2.5:7b` and `qwen2.5-coder:7b` were much faster than the 12B Gemma profiles on this laptop.
- `llama3.2:3b` was the fastest profile in the run.
- The math category is not a fair reasoning comparison because thinking output was disabled.

The hard JSON category contains four nested schema tasks: intent classification, plan construction, an approval decision with conditional fields, and multi-task extraction. It was added because the simpler JSON tests did not separate the models well.

### Reproduce the run

Pull the models you want to compare and make sure their profiles are enabled in `config.json`. Then start the server and run the benchmark:

```powershell
$env:API_PORT = "8011"
.\run.ps1
python benchmark.py --timeout 300
```

You can limit the categories:

```powershell
python benchmark.py --include-categories structured_hard
python benchmark.py --skip-categories math
```

JSON and Markdown reports are written under `reports/`. The web UI can also export self-contained HTML report cards.

## Reference profiles

Live `config.json` files start empty and fill as models are pulled. `config.example.json` shows the available fields and a few profile recipes. Copy only the entries you intend to use.

### Ollama text profiles

| Profile | Model tag |
|---|---|
| `gemma3_4b_ollama_safe` | `gemma3:4b` |
| `gemma3_12b_ollama_safe` | `gemma3:12b` |
| `qwen25_coder_7b_ollama` | `qwen2.5-coder:7b` |
| `qwen3_8b_ollama` | `qwen3:8b` |
| `llama31_8b_ollama` | `llama3.1:8b` |
| `phi4_mini_ollama` | `phi4-mini` |
| `deepseek_r1_distill_qwen_7b_ollama` | `deepseek-r1:7b` |
| `mistral_7b_ollama` | `mistral:7b` |

### Ollama vision profiles

| Profile | Model tag |
|---|---|
| `qwen3vl_8b_ollama` | `qwen3-vl:8b-instruct` |
| `qwen3vl_4b_ollama` | `qwen3-vl:4b-instruct` |
| `gemma3_12b_ollama_safe` | `gemma3:12b` |

### llama.cpp profiles

The example has GGUF recipes for Gemma 3 12B and Qwen3 8B. Their `model_id` values are placeholder Windows paths. Change them to real local files before enabling the profiles.

The recipes use flash attention and `q8_0` K/V cache types. Current llama.cpp options and supported cache types are documented in the [llama.cpp server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).

## Quantization notes

Quantization trades model size and memory for output quality. The right choice depends on model family, context length, and available memory. LocalDeploy's quant advisor checks the published Ollama tags for a model family and estimates each available option against the current fit budget.

| Quantization | Approximate effective bits per weight | Typical use |
|---|---:|---|
| `Q8_0` | 8.5 | Small models when quality matters more than memory |
| `Q6_K` | 6.5 | Extra quality when there is comfortable headroom |
| `Q5_K_M` | 5.5 | A middle ground for 7B and 8B models |
| `Q4_K_M` | 4.9 | Common default for 7B and 8B models on 8 GB cards |
| `IQ4_XS` | 4.5 | Fitting a larger model or longer context |
| `Q3_K_M` | 3.9 | A last resort when higher quants do not fit |

For llama.cpp, quantizing the KV cache can save substantial memory at long context lengths. `q8_0` uses roughly half the cache memory of `f16`; `q4_0` uses roughly one quarter. Quality can drop as cache precision is reduced. Check the current llama.cpp documentation because supported combinations change.

## Official model pages

- [Qwen3-VL on Ollama](https://ollama.com/library/qwen3-vl)
- [Qwen3 on Ollama](https://ollama.com/library/qwen3)
- [Qwen2.5-Coder on Ollama](https://ollama.com/library/qwen2.5-coder)
- [Llama 3.1 on Ollama](https://ollama.com/library/llama3.1)
- [Mistral 7B on Ollama](https://ollama.com/library/mistral)
- [Gemma 3 on Ollama](https://ollama.com/library/gemma3)
- [Phi-4-mini on Ollama](https://ollama.com/library/phi4-mini)
- [DeepSeek-R1 on Ollama](https://ollama.com/library/deepseek-r1)
