# Decision models for routing an agent's messages

Raw data, method and results of the comparison summarised in [docs/MODELS.md](../../docs/MODELS.md#decision-models)
and in the article ["Routing Agent Messages with a Local Decision Model"](https://oneyerge.com/writing/decision-model-message-routing/).

## The question

[YBM](https://github.com/oney-erge/YBM), a local agent, must decide for every incoming message whether it is chat, a
question about the agent's own work, or a task, and which of its tools the task needs: one of 15 routes. An 8B language
model (Qwen3-VL 8B Instruct, through Ollama, which places about two thirds of it on the laptop's 8 GB GPU and the
rest on the CPU) does this today. These runs ask whether a small
*decision model*, which returns a probability for every route in one pass and no text, can do it on the CPU in under
2 seconds, since the 8B model already fills the GPU.

## What was measured

| Set | Messages | Written by | Labels |
|---|---:|---|---|
| `routing_eval.json` / `_compact.json` | 98 + 12 follow-ups | the author, for this test | one route per message, by hand |
| `routing_heldout.json` / `_compact.json` | 43 | YBM's end-to-end test catalogue, written earlier for other tests | the routes of the tools each case requires, fixed by a rule and committed before any model was scored on them |

The sets differ only in the route descriptions: `_compact` uses one short line per route, which halves the prompt and
the CPU time. All four are in [examples/decision](../../examples/decision). No real user messages are included.

Every run asks one `choice` question per message through `POST /v1/systemone` and records the answer, its probability
and the time. The metrics, computed by [scripts/decision_bench.py](../../scripts/decision_bench.py):

- **Route accuracy**, with a 95% Wilson interval. For the held-out set, any of an item's accepted routes counts.
- **Dropped tasks**: real tasks answered as chat, the mistake that makes an agent silently do nothing.
- **ECE** (expected calibration error): how far stated probabilities are from observed accuracy; 0 is perfect.
- **Settled at 95%**: the largest share of messages a probability cut-off can answer while staying at least 95% right.
- **Median and p95 time** per message, measured by the client, one request at a time after two warm-up requests.

## Layout

| Path | Contents |
|---|---|
| `runs/2026-10-28-pytorch/` | first runs: d1 and Laya through PyTorch, and the LLM on the 98 messages (an earlier scoring that did not read the status task type; superseded by `llm-main.json`) |
| `runs/2026-10-29-comparison/` | 18 model and format choices on CPU and GPU, through llama.cpp, one run each |
| `runs/2026-10-29-validation/` | repeats, held-out runs, start-up, idle and under-load tests, the LLM on both sets, and d3-lite on the GPU with the short wording (`*-bench.json`, written by `decision_bench.py`) |
| `summary.csv` | one line per validation run |
| `analysis.json` | intervals, paired tests and the routing-mode simulation |
| `environment.md` | hardware, software versions and model files |
| `analyze.py` | recomputes `summary.csv` and `analysis.json` from `runs/` |

Each run file is the JSON that `decision_bench.py` writes (floats rounded to 4 decimals): per-message rows with the
text, expected and accepted routes, choice, probabilities and time.

## Results

All on 8 CPU threads unless marked GPU; decision models are Q4_K_M GGUF files in llama.cpp. Intervals are 95% Wilson
intervals. "Tasks as chat" adds both sets. Full per-run numbers: [summary.csv](summary.csv); the first 18-configuration
comparison is in [docs/MODELS.md](../../docs/MODELS.md#decision-models).

| Model | Runs on | 98 messages | 43 held-out | Median time | p95 | Tasks as chat | Memory in use |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-VL 8B, the LLM (YBM's classifier) | GPU and CPU | 92% (85-96%) | 88% (76-95%) | 15.3 s | 17.3 s | 2 | 8.1 GB |
| d3-lite 0.85B, short route wording | CPU | 93% (86-96%) | 84% (70-92%) | 0.53 s | 0.60 s | 0 | 1.0 GB |
| d3-lite 0.85B, short route wording | GPU | 93% (86-96%) | 88% (76-95%) | 0.035 s | 0.05 s | 0 | 1.2 GB |
| d3-nano 2.2B | CPU | 93% (86-96%) | 86% (73-93%) | 2.0 s | 2.2 s | 0 | 2.2 GB |
| d3-mini 4.5B | CPU | 95% (89-98%) | 93% (81-98%) | 7.8 s | 9.4 s | 0 | 4.9 GB |
| d3-mini 4.5B | GPU | 95% (89-98%) | 95% (85-99%) | 0.17 s | 0.19 s | 0 | 4.0 GB |
| d1-3B | CPU | 83% (74-89%) | 72% (57-83%) | 2.6 s | 2.8 s | 3 | 2.9 GB |

The d3 rows use the long route wording except where marked; the short wording made d3-lite 1.7 times faster at the same
accuracy but cost d3-nano 6 points (comparison run). Memory is the llama-server's RAM working set after loading; for the
LLM it is Ollama's own figure across GPU and RAM.

**Repeats.** d3-lite ran 3 times on the 98 messages, d3-nano and d3-mini (GPU) twice: every repeat chose the same route
for every message, with medians within 2%. The same file gave slightly different answers on CPU and GPU builds: d3-lite
84% on the CPU and 88% on the GPU for the held-out set, d3-mini 93% and 95%.

**Against the LLM, message by message.** On the 98 messages both were right on 86, only d3-lite on 5, only the LLM on 4,
and both wrong on 3 (exact McNemar test p = 1.0). On the held-out set: 33 both, 3 only d3-lite, 5 only the LLM, 2 neither
(p = 0.73). Neither difference is significant. The LLM reported a confidence of 0.95 to 1.00 whether right or wrong; every
d3-lite miss on the held-out set had a probability of 0.87 or less.

**YBM's routing modes, replayed** with each model's recorded answer and time (`min_confidence` 0.9):

| Mode | 98 messages: accuracy | decided by d3-lite alone (right) | mean time | 43 held-out: accuracy | alone (right) | mean time |
|---|---:|---:|---:|---:|---:|---:|
| `off` (LLM only) | 92% | 0 | 14.6 s | 88% | 0 | 17.4 s |
| `cascade` | 91% | 33 (31) | 9.8 s | 88% | 16 (16) | 11.5 s |
| `decision` | 92% | 51 (49) | 7.3 s | 88% | 18 (18) | 10.8 s |

In `decision` mode the median time on the 98 messages fell from 15.3 s to 0.6 s. The two wrong solo answers were "can you
read PDFs?" (chat, routed to documents) and "take a screenshot of github.com" (a browser screenshot, routed to the
desktop). Chat messages rarely reached 0.9 (2 of 14), so a chat reply usually still waits for the LLM.

**Start-up, idle and load (d3-lite, CPU).** llama-server was ready 1.0 to 1.3 s after start and answered its first
question 0.5 to 0.6 s later; through LocalDeploy, the first answer after a restart took 1.6 s. Enabling the shipped
`d3_lite_decision` profile on a machine without the file took 25 s for the first answer (BF16 download, quantization,
start), and the file it made was byte-identical to the one benchmarked. Kept loaded it used about 1 GB of RAM and 2% of
one core while idle, and the first answer after 5 idle minutes took 0.52 s. With the 8B LLM generating without pause, the
d3-lite median rose from 0.66 s to 1.07 s (p95 1.13 s) and the LLM slowed from 12.1 to 9.6 tokens per second.

## Limits

- 98 and 43 messages give wide intervals: a difference of 5 points between two models is within the noise of one set.
- The 98 messages and the route descriptions were written by the author, and the short descriptions were chosen on the
  same 98 messages. The held-out set exists because of this; the decision models scored 0 to 11 points lower on it
  and the LLM 4 points lower.
- Held-out labels follow the tools the e2e cases require. Those tests name "Claude Code" under `coding.agent`, while
  YBM's own route text says "Codex or Copilot"; no model was told otherwise, and two of the held-out misses are there.
- One laptop (Ryzen 9 5900HS, RTX 3080 Laptop 8 GB, Windows 10), one llama.cpp build (b11976). Times on other machines
  will differ; the ratios between models should hold better than the absolute values.
- The LLM was measured with YBM's own classifier code and no conversation context; with context, its live
  classifications took 11 to 22 s. An earlier scoring of the LLM run (in `runs/2026-10-28-pytorch/llm_baseline.json`)
  did not read its status task type and reported 88%; `llm-main.json` is the corrected rerun.

## Reproduce

1. Install llama.cpp b11976 or newer (`winget install ggml.llamacpp`, `brew install llama.cpp`, or a release).
2. Enable a decision profile in LocalDeploy's `config.json` (for example `d3_lite_decision`) and start LocalDeploy, or
   start `llama-server -m <file>.gguf -c 4096 -b 4096 -ub 4096 -np 1 -t 8 -ngl 0` yourself.
3. `python scripts/decision_bench.py --url http://127.0.0.1:8000 --profile d3_lite_decision --dataset examples/decision/routing_eval_compact.json --out run.json`
4. `python benchmarks/decision-routing/analyze.py` recomputes the summary from the published runs.
