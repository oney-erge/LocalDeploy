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

## Decision models

Decision models answer typed questions at `POST /v1/systemone` instead of writing text; see
[Decision models](API_OPTIONS.md#decision-models). This run compares the ones LocalDeploy can serve on one job:
routing a message sent to a computer-agent bot to one of 15 routes (chat, status, files, browser, scheduling, and so
on). It is not a ranking of the models in general.

**Method.** 98 hand-labeled, synthetic messages (`examples/decision/routing_eval.json`) plus 12 steering and answer
cases, asked as one 15-option question. A second set with the same messages and much shorter route descriptions
(`routing_eval_compact.json`, 47% of the characters) shows what a shorter prompt buys. Hardware: Ryzen 9 5900HS (8 cores,
16 threads) and RTX 3080 Laptop (8 GB), Windows. Runtime: llama.cpp b11976 with GGUF files (CPU build at 8 threads, CUDA
12.4 build with every layer on the GPU), except where a row says PyTorch. Timings are the median and 95th percentile of
the 98 requests after two warm-up requests. Files marked `*` are not published: they were made from the published BF16
file with `scripts/make_decision_gguf.py`. "Dropped" counts real tasks answered as chat (the costly mistake) and
"Spurious" counts chat turned into a task. "Settled at 95%" is the share of messages a confidence cut-off could answer
while staying at 95% accuracy or better. ECE is how far stated confidence is from actual accuracy (lower is better).
VRAM includes about 1 GB the desktop was already using.

**What the run supports**

- **For a CPU, `d3-lite` (0.85B, Apache-2.0) at Q4_K_M is the pick.** With the short route wording it scored 93% on the
  exact route and 97% on chat versus task, mistook no real task for chat, and answered in 0.61 s at 8 threads (0.51 s at
  16, 0.82 s at 4). Through LocalDeploy's `llamacpp` adapter the same run took 0.69 s (p95 0.76 s).
- **Quantization is the biggest speed lever on a CPU, and it cost no accuracy here.** `d3-lite` went from 1.9 s (Q8_0) to
  1.05 s (Q4_K_M) with the same 92%; `d3-nano` from 5.2 s to 2.3 s at 93%; `d1-3B` from 6.1 s to 2.8 s. llama.cpp also
  beat PyTorch: `d1-3B` took 6.8 s in PyTorch fp32 against 2.8 s at Q4_K_M, with the same accuracy.
- **A shorter prompt is the second lever, and it depends on the model.** The short wording took `d3-lite` Q4_K_M from
  1.05 s to 0.61 s and its accuracy from 92% to 93%; `d3-nano` fell from 93% to 87%, so it keeps the long wording.
- **The 4B-class models are the most accurate and too slow for a CPU.** `d3-mini` (the "d3 4b", 4.5B) scored 95% with a
  confident answer for 99% of messages, in 0.18 s on the GPU, but needs 4.6 s (short wording) to 8.1 s (long wording) on
  the CPU. `Kev-4B` scored 94% but took 7.9 s on the CPU and its confidence is unreliable (ECE 0.41); `lev` scored 91% at
  15.7 s.
- **Small models that do not work zero-shot on this task:** `Julia-1` (144M) 8%, `Laya` (421M) 28%, `d1-omni-600M` 56% with
  37 real tasks answered as chat, and `d3-edge` (0.59B) 78%. Laya's card says it is a base to fine-tune. `Kev-0.8B` reached
  86% but is poorly calibrated (ECE 0.37) and slower than `d3-lite`.
- **Against a generative LLM** (qwen3-vl 8B asked the same question through YBM's classifier): the LLM scored 88% on the
  exact route (93.5% if the `status` route, which it answers through a separate field, is left out) and `d3-lite` 93%
  (92.4% by the same measure). On chat versus task `d3-lite` was better (95 against 90 of 98), and the LLM's confidence
  was 0.95 to 1.00 whether it was right or wrong. When the two agree they are right 97.6% of the time, and only 3 messages
  were wrong for both. The LLM took 15 s per message.

### CPU (8 threads), long route wording, fastest first

| Model | Params | Format | File | License | Route acc. | Chat vs task | Dropped | Spurious | ECE | Settled at 95% | Median | p95 | Steering |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Julia-1 | 144M | Q8_0 | 0.2 GB | Apache-2.0 | 8% | 86% | 0 | 14 | 0.869 | 0% | 139 ms | 164 ms | 25% |
| Laya (English) | 421M | Q8_0 | 0.4 GB | Apache-2.0 | 28% | 86% | 0 | 14 | 0.463 | 0% | 586 ms | 677 ms | 33% |
| d3-edge | 0.59B | Q8_0 | 0.8 GB | Apache-2.0 | 78% | 91% | 5 | 4 | 0.067 | 45% | 921 ms | 1,046 ms | 25% |
| d3-lite | 0.85B | Q4_K_M * | 0.6 GB | Apache-2.0 | 92% | 97% | 0 | 3 | 0.051 | 90% | 1,054 ms | 1,149 ms | 75% |
| d3-lite | 0.85B | Q5_K_M * | 0.7 GB | Apache-2.0 | 92% | 97% | 0 | 3 | 0.037 | 87% | 1,080 ms | 1,223 ms | 83% |
| d1-omni-600M | 587M | Q8_0 | 0.4 GB | LFM Open 1.0 | 56% | 61% | 37 | 1 | 0.145 | 0% | 1,099 ms | 1,283 ms | 50% |
| d3-lite | 0.85B | Q6_K * | 0.8 GB | Apache-2.0 | 93% | 98% | 0 | 2 | 0.039 | 92% | 1,121 ms | 1,195 ms | 83% |
| d3-lite | 0.85B | Q8_0 | 1.0 GB | Apache-2.0 | 92% | 97% | 0 | 3 | 0.038 | 93% | 1,912 ms | 2,126 ms | 83% |
| Kev-0.8B | 0.8B | Q8_0 | 0.8 GB | Apache-2.0 | 86% | 94% | 1 | 5 | 0.374 | 65% | 2,178 ms | 2,389 ms | 42% |
| d3-nano | 2.2B | Q5_K_M * | 1.6 GB | Apache-2.0 | 93% | 97% | 0 | 3 | 0.055 | 92% | 2,187 ms | 2,296 ms | 92% |
| d3-nano | 2.2B | Q4_K_M * | 1.5 GB | Apache-2.0 | 93% | 97% | 0 | 3 | 0.067 | 94% | 2,288 ms | 2,435 ms | 92% |
| d1-3B | 3.1B | Q4_K_M | 1.6 GB | LFM Open 1.0 | 83% | 96% | 3 | 1 | 0.018 | 55% | 2,825 ms | 3,188 ms | 83% |
| d3-nano | 2.2B | Q8_0 | 2.4 GB | Apache-2.0 | 93% | 97% | 0 | 3 | 0.036 | 93% | 5,163 ms | 5,941 ms | 92% |
| d1-3B | 3.1B | Q8_0 | 2.7 GB | LFM Open 1.0 | 84% | 96% | 3 | 1 | 0.050 | 61% | 6,066 ms | 7,283 ms | 92% |
| Kev-4B | 4B | Q4_K_M | 2.8 GB | Apache-2.0 | 94% | 97% | 2 | 1 | 0.406 | 83% | 7,880 ms | 8,804 ms | 83% |
| lev | 4B | Q4_K_M | 2.8 GB | Apache-2.0 | 91% | 96% | 1 | 3 | 0.032 | 92% | 15,732 ms | 17,262 ms | 83% |

### CPU, short route wording (and thread scaling)

| Model | Params | Format | File | License | Threads | Route acc. | Chat vs task | Dropped | Spurious | ECE | Settled at 95% | Median | p95 | Steering |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| d3-lite | 0.85B | Q4_K_M * | 0.6 GB | Apache-2.0 | 4 | 93% | 97% | 0 | 3 | 0.071 | 95% | 821 ms | 884 ms | 75% |
| d3-lite | 0.85B | Q4_K_M * | 0.6 GB | Apache-2.0 | 6 | 93% | 97% | 0 | 3 | 0.071 | 95% | 622 ms | 714 ms | 75% |
| d3-lite | 0.85B | Q4_K_M * | 0.6 GB | Apache-2.0 | 8 | 93% | 97% | 0 | 3 | 0.071 | 95% | 608 ms | 766 ms | 75% |
| d3-lite | 0.85B | Q4_K_M * | 0.6 GB | Apache-2.0 | 12 | 93% | 97% | 0 | 3 | 0.071 | 95% | 538 ms | 592 ms | 75% |
| d3-lite | 0.85B | Q4_K_M * | 0.6 GB | Apache-2.0 | 16 | 93% | 97% | 0 | 3 | 0.071 | 95% | 507 ms | 754 ms | 75% |
| d3-lite | 0.85B | Q6_K * | 0.8 GB | Apache-2.0 | 8 | 92% | 97% | 0 | 3 | 0.060 | 90% | 638 ms | 702 ms | 83% |
| d3-nano | 2.2B | Q4_K_M * | 1.5 GB | Apache-2.0 | 8 | 87% | 93% | 0 | 7 | 0.054 | 80% | 1,302 ms | 1,479 ms | 92% |

### GPU (RTX 3080 Laptop, all layers offloaded), accuracy order

| Model | Params | Format | File | License | VRAM in use* | Route acc. | Chat vs task | Dropped | Spurious | ECE | Settled at 95% | Median | p95 | Steering |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| d3-mini ("d3 4b") | 4.5B | Q4_K_M | 3.5 GB | Apache-2.0 | 5.0 GB | 95% | 97% | 0 | 3 | 0.057 | 99% | 180 ms | 188 ms | 83% |
| d3-mini ("d3 4b") | 4.5B | Q8_0 | 4.8 GB | Apache-2.0 | 6.1 GB | 95% | 97% | 0 | 3 | 0.060 | 99% | 196 ms | 212 ms | 100% |
| Kev-4B | 4B | Q4_K_M | 2.8 GB | Apache-2.0 | 7.8 GB | 94% | 97% | 2 | 1 | 0.406 | 83% | 4,576 ms | 5,064 ms | 83% |
| d3-lite | 0.85B | Q6_K * | 0.8 GB | Apache-2.0 | 2.1 GB | 93% | 98% | 0 | 2 | 0.036 | 92% | 52 ms | 64 ms | 83% |
| d3-nano | 2.2B | Q4_K_M * | 1.5 GB | Apache-2.0 | 2.8 GB | 93% | 97% | 0 | 3 | 0.059 | 93% | 81 ms | 94 ms | 92% |
| d3-nano | 2.2B | Q8_0 | 2.4 GB | Apache-2.0 | 3.8 GB | 93% | 97% | 0 | 3 | 0.036 | 93% | 85 ms | 94 ms | 92% |
| d3-nano | 2.2B | Q5_K_M * | 1.6 GB | Apache-2.0 | 3.0 GB | 93% | 97% | 0 | 3 | 0.046 | 92% | 87 ms | 94 ms | 92% |
| d3-lite | 0.85B | Q4_K_M * | 0.6 GB | Apache-2.0 | 2.0 GB | 92% | 97% | 0 | 3 | 0.057 | 90% | 50 ms | 61 ms | 75% |
| d3-lite | 0.85B | Q8_0 | 1.0 GB | Apache-2.0 | 2.5 GB | 92% | 97% | 0 | 3 | 0.043 | 93% | 50 ms | 62 ms | 83% |
| d3-lite | 0.85B | Q5_K_M * | 0.7 GB | Apache-2.0 | 2.0 GB | 92% | 97% | 0 | 3 | 0.035 | 89% | 52 ms | 64 ms | 83% |
| lev | 4B | Q4_K_M | 2.8 GB | Apache-2.0 | 4.7 GB | 91% | 96% | 1 | 3 | 0.047 | 94% | 355 ms | 375 ms | 75% |
| Kev-0.8B | 0.8B | Q8_0 | 0.8 GB | Apache-2.0 | 6.1 GB | 86% | 94% | 1 | 5 | 0.373 | 65% | 110 ms | 217 ms | 42% |
| d1-3B | 3.1B | Q8_0 | 2.7 GB | LFM Open 1.0 | 4.8 GB | 84% | 96% | 3 | 1 | 0.057 | 56% | 90 ms | 101 ms | 92% |
| d1-3B | 3.1B | Q4_K_M | 1.6 GB | LFM Open 1.0 | 3.7 GB | 83% | 96% | 3 | 1 | 0.044 | 58% | 89 ms | 96 ms | 75% |
| d3-edge | 0.59B | Q8_0 | 0.8 GB | Apache-2.0 | 2.3 GB | 77% | 90% | 6 | 4 | 0.073 | 44% | 30 ms | 40 ms | 25% |
| d1-omni-600M | 587M | Q8_0 | 0.4 GB | LFM Open 1.0 | 2.1 GB | 55% | 60% | 37 | 2 | 0.168 | 0% | 30 ms | 42 ms | 50% |
| Laya (English) | 421M | Q8_0 | 0.4 GB | Apache-2.0 | 2.1 GB | 28% | 86% | 0 | 14 | 0.465 | 0% | 21 ms | 41 ms | 33% |
| Julia-1 | 144M | Q8_0 | 0.2 GB | Apache-2.0 | 1.6 GB | 7% | 86% | 0 | 14 | 0.882 | 0% | 15 ms | 35 ms | 33% |

### Reference: earlier PyTorch runs of d1-3B

| Model | Params | Format | File | License | Route acc. | Chat vs task | Dropped | Spurious | ECE | Settled at 95% | Median | p95 | Steering |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| d1-3B | 3.1B | PyTorch bf16 | 5.8 GB | LFM Open 1.0 | 84% | 96% | 3 | 1 | 0.031 | 62% | 108 ms | 111 ms | 92% |
| d1-3B | 3.1B | PyTorch fp32 | 5.8 GB | LFM Open 1.0 | 84% | 96% | 3 | 1 | 0.031 | 62% | 6,831 ms | 7,413 ms | 92% |

### Reference: today's LLM concierge (qwen3-vl 8B through LocalDeploy)

route accuracy 88%, chat vs task 92%, dropped tasks 7, spurious 1, confidence 0.95 to 1.00 whether right or wrong (right 0.967, wrong 0.962), median 15.0 s, p95 17.8 s.

\* made locally from the published BF16 GGUF with `llama-quantize`. VRAM in use includes about 1 GB the desktop was already using.
`d3-mini` on a CPU was timed on a 15-message sample (8.1 s with the long wording, 4.6 s with the short one); its accuracy
is from the full GPU run. Kev-4B is slow on the GPU because it filled the card (7.8 GB in use).

**Cautions.** One author labeled the 98 messages, a few labels are arguable (for example "open youtube.com" as a browser or
an app action), the set is small, and everything ran on one laptop that was otherwise lightly used. The 12-message steering set
is too small for more than a hint. Licenses: `d3-*`, Kev, Julia and Laya are Apache-2.0; `d1-*` use the LFM Open License v1.0
(free for commercial use below US$10 million annual revenue). A GGUF runs no model code, so these profiles need no
`trust_remote_code`. Reproduce with `python scripts/decision_bench.py --profile NAME` against a running LocalDeploy (or
`--url` of a llama-server), and add your own messages before relying on any of this.

### Decision profiles

Nine disabled recipes use `"backend": "decision"`. Four run a GGUF file in a `llama-server` that LocalDeploy starts
(`llamacpp` adapter, no Python packages): `d3_lite_decision` (the CPU pick), `d3_nano_decision`, `d3_mini_decision` (for a
GPU) and `d1_3b_gguf_decision`. Four use PyTorch and need `requirements-decision.txt`: `d1_omni_600m_decision`,
`d1_3b_decision`, `laya_decision` and `laya_typed_decision`; they pin the Hugging Face commit that was tested, and the d1 ones
run code that ships with the model, so they need `"trust_remote_code": true`. `systemone_http_decision` forwards to another
`/v1/systemone` server on this machine. Models under the LFM Open License v1.0 also need `"license_accepted": true`. See
[Decision models](API_OPTIONS.md#decision-models).

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
