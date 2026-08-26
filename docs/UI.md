# Web UI

LocalDeploy serves its UI at `http://127.0.0.1:8000/ui` by default. The page is plain HTML, CSS, and JavaScript. It has no CDN assets, npm dependencies, or build step.

Set `ENABLE_WEB_UI=false` if you only want the API.

## Start it

On Windows:

```powershell
.\run.bat
```

On macOS and Linux:

```bash
./run.command  # macOS
./run.sh       # Linux
```

For API-only startup, use `.\run.ps1 -NoBrowser` on Windows or `./run.sh --no-browser` on macOS and Linux. The lower-level `.\scripts\start.ps1 -Foreground` command exposes development-specific foreground logging.

The launchers read `API_HOST` and `API_PORT` from `.env`. When the server binds to `0.0.0.0`, the local browser link still uses `127.0.0.1`. On macOS and Linux, `start.sh` starts an installed Ollama server when needed. Use `START_OLLAMA=false` when Ollama is managed elsewhere, and `./scripts/stop.sh --ollama` to stop a server started by the launcher.

llama.cpp is optional. Normal startup skips an incomplete llama.cpp configuration and leaves Ollama features available. Run `.\scripts\start_llamacpp.ps1` when you are intentionally starting a GGUF profile and want configuration errors to stop the launch.

## Setup and Deploy

Start here on a new installation. Check the detected hardware, get a model, and deploy it. Pulling a model through the UI creates a profile for it. Profiles are stored in `config.json`.

| Area | Purpose | Main endpoint |
|---|---|---|
| Hardware | GPU inventory, compatible VRAM pools, CPU, RAM, and fit budget | `GET /system/hardware` |
| Recommended | Up to three models for a use case, priority, and context size | `POST /registry/recommend` |
| Model catalog | Search local runtimes, the Ollama library, Hugging Face GGUF repositories, and ModelScope GGUF repositories | `GET /registry/providers`, `POST /registry/search-models` |
| Quant advisor | Compare common quantizations against the current memory budget | `POST /system/quant-advisor` |
| Pull or import | Download an Ollama model, import a direct GGUF URL, or register a local GGUF file | `POST /models/pull`, `POST /models/import-url`, `POST /system/check-local-gguf`, `POST /profiles/upsert` |
| Deploy | Load a profile on Auto, GPU, or CPU placement | `POST /models/serve` |
| Switch and stop | Replace or unload a running model | `POST /models/switch`, `POST /models/stop` |
| Delete and free | Remove model files or unload all models from memory | `POST /models/delete`, `POST /models/free` |
| Fit checks | Estimate memory for one model, several models, or several context sizes | `POST /system/fit-check`, `POST /system/fit-batch`, `GET /system/fit-table` |

The fit budget comes from the hardware probe. You can override it to compare against another GPU or a smaller free-memory target. Green means the model should fit in VRAM, yellow means it is tight or likely to use CPU offload, and red means the estimate does not fit available GPU or system memory. Estimates are conservative and are not guarantees.

The Recommended view labels the source of each reason as estimated, published, or measured on this machine. Download and start pulls a missing model and deploys it. The catalog has runtime and size filters, sorting, pagination, source badges, and fit badges. The quant advisor uses tags that are actually published by the Ollama library instead of constructing names from a pattern.

The Pull / Import panel accepts Ollama tags, `hf.co/...` GGUF shortcuts, and `modelscope.cn/...:<file>.gguf` shortcuts. It can also register an existing local GGUF file as a llama.cpp profile, or download a direct `.gguf` URL and register it with Ollama.

The Your models card separates disk size from estimated runtime memory. It can sort models and delete several at once. Profiles whose backing model is missing are marked and can be removed under Advanced / All run profiles. For llama.cpp profiles, LocalDeploy also checks that the GGUF file exists.

### Deployment manifests

Running model cards can export a deployment manifest or generate configuration for Open WebUI, AnythingLLM, Continue, Cline, curl, Python, JavaScript, and Docker Compose.

A manifest records the model digest, quantization, runtime settings, hardware snapshot, fit estimate, and saved measurements. The manifest section can check compatibility on another machine and recreate a deployment. Recreate may pull the model before loading it.

| Action | Endpoint |
|---|---|
| Export | `POST /system/manifest/export` |
| Validate | `POST /system/manifest/validate` |
| Recreate | `POST /system/manifest/recreate` |
| Integration snippets | `GET /system/integration-snippets` |

## Chat

The Chat tab uses `POST /v1/chat/completions` with streaming enabled. Select an installed model, load it, and send a message. Conversation state remains in the page and Clear removes it.

Images are available for profiles marked as vision-capable. Text and document attachments are added to the message as text blocks. An optional system prompt is sent with every turn. Replies support basic Markdown, fenced code blocks, and copy buttons. The timing line separates model load time, first-token latency, and generation throughput when the backend reports them.

Press Enter to send and Shift+Enter for a newline. While a reply is streaming, the Send button becomes Stop.

## Benchmark and Compare

The benchmark workspace runs saved profiles against the built-in question set or a custom JSON question set. Runs are sequential by default so several models do not compete for VRAM.

The primary history is stored in browser `localStorage` under `localdeploy.benchmarkRuns.v1`. Turn on Also store on server if you want completed runs copied to `reports/benchmark-history/`. Enabling it merges browser and server history. Deleting a mirrored run also deletes its server copy.

To run a benchmark:

1. Select one or more installed profiles.
2. Leave the question editor empty for the built-in suite, or load and validate a custom set.
3. Choose Auto, CPU, GPU, or CPU + GPU placement.
4. Choose the number of repetitions.
5. Start the suite and watch the queue.

CPU and GPU runs pin the requested Ollama `num_gpu` setting during load and inference. LocalDeploy records the actual placement reported by the runtime. If a requested GPU run lands on split CPU and GPU placement, the report says so.

Benchmark deployments are temporary. Each Ollama model is unloaded after its profile finishes. A stopped item ends without stopping the rest of the queue; Cancel stops the queue. Waiting items can be reordered or removed.

Each completed run records the LocalDeploy and Ollama versions, model digest, quantization, context, initial warm or cold state, hardware snapshot, requested placement, actual placement, latency, accuracy, and backend token rate when available. Repeated runs add median, percentile, range, and variance data.

The results area includes a leaderboard, category heatmap, speed and quality plot, per-test matrix, response previews, and filters. Select at least two runs to compare changes in pass count, accuracy, latency, throughput, runtime, digest, quantization, and hardware.

### Custom question sets

```json
{
  "version": 1,
  "questions": [
    {
      "name": "planning_triage_basic",
      "category": "planning",
      "prompt": "List 3 first steps to triage a service outage. Return a JSON array of strings.",
      "max_output_tokens": 512,
      "grader": { "type": "json_array_min_len", "min": 3 },
      "grader_explainer": "Passes if the model returns a JSON array with at least 3 steps."
    }
  ]
}
```

Uploaded question sets contain data only. A fixed registry selects the grader, so an uploaded file cannot add executable grader code.

| Grader type | Fields | Result |
|---|---|---|
| `contains_all` | `keywords`, optional `case_sensitive` | Fraction of required keywords found |
| `json_array_min_len` | `min` | Passes when the response is a JSON array with enough items |
| `number_within` | `expected`, optional `tolerance` | Passes when a parsed number is within tolerance |
| `exact_match` | `expected`, optional `case_sensitive` | Passes when the trimmed response matches |
| `classification_set` | `expected` list | Passes when the returned label set matches |
| `json_keys_present` | `required` list, optional `allow_extra` | Passes when a JSON object contains the required top-level keys |

### Import and export

A single run exports as a self-contained HTML report card with embedded JSON. Several selected runs export as a JSON bundle with `kind: "localdeploy.run_bundle"`. Both formats can be imported back into the run library. Imported and current runs use the same comparison views.

## Monitor

Monitor polls `GET /system/monitor` every five seconds while its tab is open. It shows current CPU, RAM, GPU, VRAM, loaded models, recent request timing, and short rolling charts.

Request history contains numeric metadata only. Prompts and responses are not recorded. The tab can warn about sustained VRAM pressure, unexpected placement, throughput below the model's recent median, and concurrent backend calls.

When a model stops, a session summary is written to `reports/monitor-sessions/`. It includes peak memory, median throughput, median time to first token, request counts, failures, and uptime. Calibration uses model-specific Ollama memory allocation rather than whole-machine peaks.

## Auto-pick

Find best fit checks enabled profiles, runs a short benchmark on candidates that are available, and ranks them by accuracy, speed, and memory headroom. It does not search for or download models. The endpoint is `POST /system/recommend`.

## Token and offline settings

When `API_TOKEN` is set, open `/ui?token=<secret>` once. The UI stores the token in that browser and sends it with later requests. A 401 response opens the token prompt again.

This is a shared local token sent over HTTP. Keep the service on loopback. See [../SECURITY.md](../SECURITY.md) before changing network exposure.

`OFFLINE=true` disables LocalDeploy's model-search and update-check requests. It does not configure a separately started inference runtime. The Docker setup and local launchers set `OLLAMA_NO_CLOUD=true` when they start Ollama.
Run `python scripts/egress_selftest.py` to check this mode.

## Shortcuts and transport

| Input | Action |
|---|---|
| Enter in the Pull field | Start a pull |
| Ctrl+Enter or Command+Enter in the question editor | Start a benchmark |

Pull and benchmark progress use Server-Sent Events. Errors are shown inline or as toasts. The benchmark runner calls the same local API used by the rest of the application, so the configured API address must be reachable from the LocalDeploy process.
