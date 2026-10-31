# Choosing a local model for an agent (October 2026)

An agent such as YBM asks its model for one decision at a time: which tool to call, with what input, or whether the
job is done. A general quality score does not predict that. The model that ranks first in
[docs/MODELS.md](../../docs/MODELS.md) (Qwen3-VL 8B) still made up file names and job ids in live YBM runs. So this
benchmark replays the decisions a real YBM install has already made and scores them.

Hardware: RTX 3080 Laptop (8 GB), Ryzen 9 5900HS, 39 GB RAM. Everything here is Q4_K_M, 16K context.

## What it found

1. **The runtime matters more than the model.** The same Qwen3-VL 8B, answering the same 60 steps, took a median of
   **13.4 s** (slowest 1 in 20: 60 s) through LocalDeploy and Ollama, and **2.8 s** (8.9 s) through `llama-server` with
   flash attention and an 8-bit cache, fully on the GPU. The pass rate was the same. That is about five times faster for
   no change in behavior.
2. **Qwen3.5-9B decides better than Qwen3-VL 8B** on the same 599 recorded steps: 55% against 49% (paired test,
   p = 0.001), at 3.7 s a step. It also invents a file name or id more often (92% of its tool calls use only things it
   was shown, against 95% for the 8B and 97% for the 4B), and its trial in the real assistant was mixed (below). So it is
   a candidate to replace the 8B, not yet proof that it should.
3. **Bigger did not help.** Qwen3.6-35B-A3B, with its expert weights in RAM, scored 57% on the first 120 steps (the 9B:
   52% on the same ones) at 18 s a step. Qwen3.6-27B needed 113 s. Letting the 9B think before answering made it
   slower (16 s) and worse (45%).
4. **The Qwen3.5-4B is the fast option**: 51% at 2.1 s and only 3.2 GB of VRAM, which leaves room for a vision model.

## Results

"Pass" means a valid call, nothing made up, and the same choice as a step that went well (or not repeating one that
went badly). Intervals are 95%. VRAM is the model plus 16K of cache.

| Model, as run | Steps | Pass | Seconds per step (median / slowest 1 in 20) | VRAM |
|---|---:|---:|---:|---:|
| Qwen3.6-35B-A3B, experts in RAM | 120 | 57% (49-66) | 17.9 / 57 | 6.7 GB |
| **Qwen3.5-9B** | 599 | **55%** (51-59) | 3.7 / 14 | 5.6 GB |
| **Qwen3.5-4B** | 599 | **51%** (47-55) | 2.1 / 6.0 | 3.2 GB |
| Qwen3-VL 8B, `llama-server` | 599 | 49% (45-53) | 2.7 / 8.3 | 6.2 GB |
| Qwen3-VL 8B, LocalDeploy and Ollama (today) | 60 | 48% (36-61) | 13.4 / 60 | |
| Qwen3.5-9B, thinking on (768 tokens) | 60 | 45% (33-58) | 16.3 / 41 | 5.6 GB |
| Qwen3.6-27B | 8 | stopped: 113 s per step | | |

Also tried, none better than the Qwen models: Gemma 4 E4B (49%, 2.7 s), Gemma 4 12B (50%, 11 s), Gemma 4 26B-A4B
(50%, 15 s), GLM-4.6V-Flash (42%; it invented a reference in 31% of its tool calls) and gpt-oss-20b (30 to 46 s a step,
because its reasoning cannot be switched off). Scores for all of them are in [summary.json](summary.json).

On the 599 steps, between the three small models ([paired-comparison.txt](paired-comparison.txt)): the 9B beats the
Qwen3-VL 8B on overall pass (p = 0.001), on not repeating a step that went badly (p = 0.008) and on matching a good
step (p = 0.007). The 4B and the 8B are not distinguishable (p = 0.15). The 4B invents fewer references than the 9B
(p < 0.001).

## A trial in the real assistant

Qwen3.5-9B answered 15 of the 17 planned Telegram cases on a real YBM install (the run was stopped to save time, so the
4B and the Qwen3-VL 8B were not run the same way).

- Passed (6): inspecting a folder, inspecting the desktop, sending a PDF, recovering from a stale path, a three-turn
  "find it, send it, shorten it" chain, and listing a zip's contents.
- Failed (9). In five the model asked for the same tool call again and again after it had worked, and YBM's guard stopped
  the task instead of letting it loop (reading a file, summarizing a PDF, creating a schedule, extracting text, reading a
  long PDF). One case (sort files, then undo) stopped twice when the local model timed out, and YBM asked before
  sending anything to a hosted model; nothing was sent. The other three: a wrong answer on a compound request, a task
  cancelled after 11 minutes, and one stopped by the answer checker.

So single decisions are better, and finishing a job is where it still struggles. That, and the missing side-by-side run
of the 8B on the same cases, is why the 9B has an example profile here but is not the default.

## Reproduce

The tooling is in YBM: [`scripts/agent_benchmark`](https://github.com/oney-erge/YBM/tree/main/scripts/agent_benchmark).
`environment.md` has the settings, model files and checksums. The cases are one install's own history and contain its
owner's messages, so they are not published; `summary.json` has scores and timings only.

## Limits

- Labels come from what each recorded call led to, not from a person. A step can look bad for a reason that is not the
  model's fault, and there can be more than one good answer, so these scores compare models; they are not a grade.
- One attempt per step. YBM retries a bad answer up to three times.
- Single steps, replayed. How a model recovers across a whole task shows up only in the end-to-end cases.
- The first 120 steps are shared by every model. The wider 599 were run only for the three small models. Anything
  measured on 8 to 12 steps is a probe, not a result.
- The GPU reached 86 C and throttled during the long runs, so timings come only from the first quiet pass.
