# Comparing a model-fit estimate

On October 28, 2026, we asked two tools for general-purpose model suggestions
under a 6 GB GPU-memory budget and an 8,192-token context target. This is a
reproducible inspection of their recommendations, not a performance benchmark
or a claim that either tool predicts fit more accurately. No models were
downloaded or run in this comparison.

The machine had an NVIDIA RTX 3080 Laptop GPU with 8 GB VRAM and 39.4 GB RAM.
LocalDeploy was the development checkout based on 0.6.3, including the fix
that uses the requested context in the guided fit estimate; llmfit was 1.1.18.
Both were given a 6 GB effective GPU budget. LocalDeploy's request used 8 GB
free VRAM with a 2 GB safety margin; llmfit was given 6 GB directly.

| Tool | Suggested model | Estimated GPU memory | Context used |
|---|---|---:|---:|
| LocalDeploy | `qwen3.5:2b` | 3.02 GB | 8,192 tokens requested |
| LocalDeploy | `qwen2.5:0.5b` | 1.35 GB | 8,192 tokens requested |
| LocalDeploy | `gemma3:4b` | 5.24 GB | 8,192 tokens requested |
| llmfit | `TheBloke/WizardLM-Uncensored-Falcon-7B-GPTQ` | 4.35 GB | 4,096 tokens effective |
| llmfit | `cosmicproc/Qwen3.5-4B-NVFP4` | 4.00 GB | 8,192 tokens effective |
| llmfit | `cyankiwi/VulnLLM-R-7B-AWQ-4bit` | 4.75 GB | 8,192 tokens effective |

The candidates, model catalogs, runtimes, quantizations, ranking rules, and
context handling differ. In llmfit, `--max-context` is a cap; the first result
used 4,096 tokens. LocalDeploy's catalog returns Ollama tags and estimates an
unquantized KV cache with a default weight-size assumption. These rows cannot
establish which suggestion is faster or better, or even compare like-for-like
model files. A second lookup of `google/gemma-3-4b-it` in llmfit reported
5.00 GB at Q6_K, so it also should not be treated as a direct measurement of
LocalDeploy's `gemma3:4b` estimate.

With LocalDeploy running locally, repeat its read-only recommendation request:

```bash
curl -sS -X POST http://127.0.0.1:8000/registry/recommend \
  -H 'Content-Type: application/json' \
  -d '{"free_vram_mb":8192,"margin_gb":2,"expected_context":8192,"use_case":"general","priority":"balanced"}'
```

Compare the other tool's current output:

```bash
uvx --from llmfit==1.1.18 llmfit --no-dashboard recommend \
  --memory 6G --max-context 8192 -n 3 --use-case general --json
```

Both commands produce estimates. To establish actual fit and speed, run the
same model file, quantization, context, runtime, and prompt on the same machine
and record observed memory and throughput. LocalDeploy's Benchmark screen can
record a local run; it does not make the two tools' rankings equivalent.
