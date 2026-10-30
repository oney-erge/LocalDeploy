# API Options For External Clients

External clients can call either the native LocalDeploy API or the OpenAI-compatible API.

## Native `/chat`

```json
{
  "profile": "gemma3_4b_ollama_safe",
  "prompt": "Explain the tradeoff in one paragraph.",
  "system_prompt": "You are concise.",
  "temperature": 0.2,
  "top_p": 0.9,
  "repeat_penalty": 1.1,
  "max_output_tokens": 256,
  "context_limit": 4096,
  "safe_mode": true,
  "allow_clamp": false,
  "timeout_seconds": 180
}
```

The native `/chat` endpoint returns one JSON object. For streaming, use the OpenAI-compatible endpoint below with `stream: true`.

Important fields:

- `profile`: selects a configured profile from `config.json`.
- `prompt`: user text.
- `system_prompt`: optional behavior instruction.
- `max_output_tokens`: maps to Ollama `num_predict`.
- `context_limit`: maps to Ollama `num_ctx`.
- `safe_mode`: when true, caps context at the profile `safe_context_limit`.
- `allow_clamp`: when false, oversized requests return a clear error; when true, the server clamps to the configured maximum.
- `temperature`, `top_p`, `repeat_penalty`: sampling controls.
- `timeout_seconds`: per-request timeout.
- `response_format`: optional `json_object` or `json_schema` constraint. LocalDeploy forwards it to Ollama/llama.cpp native constrained generation when supported.

Request options cannot bypass server safety limits. To allow larger context, edit `config.json` first and raise the selected profile's `context_limit`, `safe_context_limit`, and `max_prompt_chars` deliberately.

## OpenAI-Compatible `/v1/chat/completions`

```json
{
  "model": "gemma3_4b_ollama_safe",
  "messages": [
    { "role": "system", "content": "You are concise." },
    { "role": "user", "content": "Explain the tradeoff in one paragraph." }
  ],
  "max_tokens": 256,
  "temperature": 0.2,
  "top_p": 0.9,
  "context_limit": 4096,
  "safe_mode": true,
  "allow_clamp": false
}
```

`model` can be either a profile name such as `gemma3_4b_ollama_safe` or a backend model id such as `gemma3:4b`. The server still applies the matching profile's safety limits.

### Streaming

Set `"stream": true` to receive a Server-Sent Events stream in OpenAI chunk format:

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

Streaming is token-by-token for Ollama profiles. For llama.cpp profiles, the full response is emitted as one content delta in the same wire format, without progressive output.

### Structured output

Pass OpenAI-style `response_format` to request JSON or schema-constrained output. LocalDeploy keeps a short schema
instruction in the prompt and also forwards the constraint to the backend: Ollama receives its native `format` value,
while llama.cpp receives `response_format` on `/v1/chat/completions`. Clients must still parse and validate the result;
application-level retry or rejection remains necessary if a backend/model returns semantically invalid data.

### Tool calls

Pass OpenAI-style `tools` and `tool_choice` fields. LocalDeploy forwards function declarations to
the selected local runtime and returns `message.tool_calls` with `finish_reason: "tool_calls"`.
It never executes a tool; the client owns validation, authorization, execution, and any follow-up
tool message. Ollama and tool-capable loopback OpenAI-compatible runtimes are supported.

## Responses and embeddings

`POST /v1/responses` accepts string or message-list `input`, optional `instructions`, function
tools, and `max_output_tokens`. Non-streaming responses return `output_text` plus message or
`function_call` output items. Streaming emits progressive typed Responses SSE events, including
`response.output_text.delta`, function-call argument events, and `response.completed`.

`POST /v1/embeddings` accepts a string or list of strings and `encoding_format` of `float` or
`base64`. Ollama uses `/api/embed` with a legacy fallback; other local providers use their
`/v1/embeddings` endpoint.

Profiles may target `ollama`, `llamacpp`, `lmstudio`, `vllm`, `docker`, or `openai`. Every backend
URL is still required to resolve to loopback; provider support does not permit cloud API URLs.

## Decision models

A decision model does not write text. You give it a piece of text (the *state*) and named
*questions*; it returns one typed answer per question with a probability for every option, in a
single forward pass. Use it inside an application to classify or route ("is this chat or a task,
and which route?") and call a language model only when the answer is not confident.

Decision profiles (`"backend": "decision"`) are off by default. Set `"enabled": true` on one in `config.json`; a model
with its own license also needs `"license_accepted": true` once you have read it (LiquidAI d1 uses the LFM Open
License v1.0). Models load on first use, which the response reports as `load_ms`, and stay loaded. The examples in
`config.example.json` are the starting point, and [Models](MODELS.md#decision-models) compares them.

**On a CPU, use the `llamacpp` adapter.** It starts a `llama-server` for a GGUF file, needs no Python packages, runs no
model code, and was the fastest option in the comparison. It needs llama.cpp b11976 or newer (it serves
`/v1/systemone`; an older `llama-server` is reported with its build number):

```bash
winget install ggml.llamacpp      # Windows (or: winget upgrade ggml.llamacpp)
brew install llama.cpp            # macOS
# or download a release from https://github.com/ggml-org/llama.cpp/releases and set LLAMACPP_SERVER_PATH
```

LocalDeploy finds `llama-server` at the profile's `server_path`, at `LLAMACPP_SERVER_PATH`, on `PATH`, or in a WinGet
install. Then set `"enabled": true` on `d3_lite_decision`. Its `"quantize": "Q4_K_M"` makes LocalDeploy download the
model's BF16 file once and turn it into a Q4_K_M file with `llama-quantize` (shipped next to `llama-server`), kept in
LocalDeploy's `models/decision` folder; on a CPU that file answers about 1.8 times faster than the published Q8_0 at the
same accuracy. The first request waits for this (about a minute); `scripts/make_decision_gguf.py` does it ahead of time.

The PyTorch adapters (`d1`, `laya`) need optional packages instead
(`python -m pip install -r requirements-decision.txt`, or `pip install "localdeploy[decision]"`). On Windows the default
`torch` wheel is CPU-only; install an NVIDIA build first for the GPU (see `requirements-decision.txt`). With
`OFFLINE=true` they load from the local Hugging Face cache only, and profiles that run code shipped with the model (`d1`)
require `"trust_remote_code": true` and a pinned `"revision"`; the shipped profiles pin the commit that was tested.

| `adapter` | Runs | Notes |
|---|---|---|
| `llamacpp` | A GGUF decision model in a `llama-server` that LocalDeploy starts, on a free loopback port, restarts if it dies and stops on exit | `model_id` is a `.gguf` path, `repo:quant`, or a repository with `quantize`; `device` `cpu`, `cuda` or `auto`; `threads`, `context`, `batch`, `server_path`, `port` |
| `d1` | LiquidAI d1 (`d1-omni-600M`, `d1-3B`) in this process through PyTorch | `device`: `auto`, `cpu` or `cuda`; `dtype` as the model card advises |
| `laya` | Laya checkpoints (`english`, `multilingual`, `typed-decisions`) in this process | Apache-2.0; `checkpoint` picks one |
| `systemone-http` | Forwards to another server on this machine that already speaks `/v1/systemone` | Loopback `base_url` only; no extra packages |

```bash
curl http://127.0.0.1:8000/v1/systemone -H "Content-Type: application/json" -d '{
  "model": "d3_lite_decision",
  "state": "remind me tomorrow at 9am to send the budget draft",
  "questions": {
    "route": {
      "type": "choice",
      "instructions": "Which route should handle this message?",
      "criteria": {
        "conversation": "chat or a general question",
        "schedule.manage": "reminders and recurring jobs",
        "filesystem.manage": "look at or change files"
      }
    },
    "urgent": {"type": "noul", "instructions": "Does the message ask for something urgent?"}
  }
}'
```

```json
{
  "answers": {
    "route": {"choice": "schedule.manage", "confidence": 0.99,
              "probabilities": {"conversation": 0.003, "schedule.manage": 0.99, "filesystem.manage": 0.007}},
    "urgent": {"noul": 0.04}
  },
  "model": "d1_omni_600m_decision",
  "usage": {"input_tokens": 118, "output_tokens": 0},
  "latency_ms": 41.3
}
```

The values above are illustrative. `state` is text or any JSON value. Several questions about one state are
answered together; where the PyTorch build has no flash-attention kernel (the Windows CUDA wheels), `d1` falls back
to asking them one at a time on the GPU, with the same answers and no shared-pass speedup. Question types:

| `type` | `criteria` | Answer |
|---|---|---|
| `choice` | `{option: description}`, 2 to 100 options | `choice`, `confidence`, `probabilities` |
| `score` | list of 2 to 10 level descriptions, lowest first | `score`, `confidence`, `probabilities` |
| `noul` | optional `{"true": "...", "false": "..."}` | `noul`: the probability of yes |

`model` (or `profile`) names a decision profile; without it the first enabled decision profile is
used, or `default_decision_profile` in `config.json`. Limits: 32 questions, 30,000 characters of
state (or the profile's `max_prompt_chars`). Errors use the same JSON error shape as the other
endpoints: `400` for a malformed request, `404` when no decision profile is enabled, `503` when
the profile cannot be used yet (packages missing, license not accepted, model files absent in
offline mode), and `502` when the model fails while answering.

Treat `confidence` as a gate, not a guarantee: act on answers above a cut-off you have measured
and send the rest to a language model. `python scripts/decision_bench.py --profile NAME` measures
that on the labeled set in `examples/decision/`, including how many messages a cut-off settles.
Decision profiles are for `/v1/systemone` only; the chat endpoints reject them with a message
saying so, and `/health` lists the decision models currently loaded.

## Context Testing

The API defaults are intentionally conservative. Use `compare_models.py` for normal comparisons. For probing larger contexts, increase limits in your local `config.json` or test directly against Ollama with `num_ctx` before changing API defaults.

## Limits Resolution Order

For every request, the server combines three sources in this order. The final value is always bounded by the profile and global caps. A request cannot raise a server-side limit.

### 1. Global caps

These environment variables are read by `get_global_limits` in `api_server.py`:

   - `GLOBAL_MAX_PROMPT_CHARS` (default 20000)
   - `GLOBAL_MAX_OUTPUT_TOKENS` (default 2048)
   - `GLOBAL_MAX_IMAGES` (default 8; hard server ceiling)
   - `GLOBAL_MAX_IMAGE_MB` (default 10)
   - `REQUEST_TIMEOUT_SECONDS` (default 180)
   - `SLOW_RESPONSE_SECONDS` (default 60)

### 2. Profile limits

The selected profile in `config.json` supplies:

   - `max_prompt_chars`, `max_output_tokens`, `max_images`, `context_limit`, `safe_context_limit`, `timeout_seconds`, `slow_response_seconds`.

### 3. Request fields

The caller can supply:

   - `context_limit`, `max_output_tokens`, `timeout_seconds`, `safe_mode`, `allow_clamp`.

### How the effective value is chosen

| Field | Effective value |
|---|---|
| `max_prompt_chars` | `min(profile.max_prompt_chars, GLOBAL_MAX_PROMPT_CHARS)`. A request cannot raise it. |
| `context_limit` | `safe_mode=true`: `min(profile.context_limit, profile.safe_context_limit, request.context_limit)`. `safe_mode=false`: `min(profile.context_limit, request.context_limit)`. |
| `max_output_tokens` | `min(profile.max_output_tokens, GLOBAL_MAX_OUTPUT_TOKENS, request.max_output_tokens)`. |
| `timeout_seconds` | `request.timeout_seconds` if set, else `profile.timeout_seconds`, else `REQUEST_TIMEOUT_SECONDS`. |
| `images` | `len(images) > min(profile.max_images, GLOBAL_MAX_IMAGES)` → reject. Profiles without `max_images` retain the legacy limit of 1. Each image: size ≤ `GLOBAL_MAX_IMAGE_MB`. |
| sampling params (`temperature`, `top_p`, `repeat_penalty`) | `request` value if provided, else profile default. No global cap. |

When the request exceeds a cap and `allow_clamp` is `false` (default), the server returns an error rather than silently shrinking. With `allow_clamp=true` the server clamps down to the allowed value and proceeds.

### Default profile resolution

The profile used when the request omits `profile` is resolved in this order:
1. `request.profile` if present
2. `DEFAULT_MODEL_PROFILE` env var
3. `default_profile` in `config.json`
4. Hard fallback: `gemma3_4b_ollama_safe`

The OpenAI-compatible endpoint additionally maps `model` to a profile by matching either a profile name or any profile's `model_id`.

### Where to change limits

- Edit `config.json` to raise limits for one profile.
- Edit `.env` or the process environment to change a `GLOBAL_*` limit.
- Set `safe_mode=false` or `allow_clamp=true` for a less restrictive request. Neither option bypasses global caps.
