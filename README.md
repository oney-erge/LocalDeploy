<p align="center">
  <img src="https://raw.githubusercontent.com/oney-erge/LocalDeploy/main/localdeploy/web/logo.svg" width="96" height="96" alt="LocalDeploy logo" />
</p>

<h1 align="center">LocalDeploy</h1>

<p align="center"><strong>Choose, run, and compare local AI models on your own hardware.</strong></p>

<p align="center">
  <a href="https://github.com/oney-erge/LocalDeploy/actions/workflows/ci.yml"><img src="https://github.com/oney-erge/LocalDeploy/actions/workflows/ci.yml/badge.svg" alt="CI status" /></a>
  <a href="https://pypi.org/project/localdeploy/"><img src="https://img.shields.io/pypi/v/localdeploy.svg" alt="PyPI version" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT license" /></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+" />
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/oney-erge/LocalDeploy/main/docs/assets/demo.gif" alt="LocalDeploy walkthrough showing hardware-aware model recommendations, deployment, local chat, benchmark comparisons, and live monitoring" width="820" />
</p>

LocalDeploy helps you answer a practical question before you download a model:
will it fit and run well on this machine? It detects hardware, estimates memory
use, manages local models, records benchmarks, and serves the deployed model
through an OpenAI-compatible API. It runs locally, has no telemetry, and does
not provide cloud inference.

## Quick start

With Python 3.10+ and [Ollama](https://ollama.com) installed:

```bash
pip install localdeploy
localdeploy        # opens http://localhost:8000/ui
```

The UI guides you through hardware detection, model search, a fit estimate,
deployment, chat, and benchmarking. If you use `uv`, `uvx localdeploy` runs it
without a permanent install.

Need help setting up Python or Ollama? Clone the repository and run its launcher:

```bash
git clone https://github.com/oney-erge/LocalDeploy.git
cd LocalDeploy
./run.sh           # Linux; macOS: ./run.command
```

On Windows, use `run.bat`. The launchers can prepare the environment and open
the UI; [Installation](docs/INSTALL.md) covers prerequisites, Docker, and the
Windows setup script. Desktop builds are experimental and unsigned; see
[Packaging](docs/PACKAGING.md).

## What it helps with

- Find models from Ollama, Hugging Face, ModelScope GGUF repositories, local
  GGUF files, or direct URLs.
- Estimate model fit before download, including likely CPU offload.
- Pull, deploy, switch, unload, and remove local models.
- Chat with text, images, and documents, then compare installed models using
  measured latency, throughput, memory, and accuracy results.
- Monitor CPU, RAM, VRAM, model placement, and request timing while a model runs.
- Connect compatible tools through `/v1/chat/completions`, `/v1/responses`,
  `/v1/embeddings`, and `/v1/models`.

If you already know which Ollama model you want and only need terminal chat,
Ollama may be enough. LocalDeploy is for selecting and evaluating models on a
particular machine.

## What to expect

Fit estimates are guidance, not guarantees. Driver versions, context length,
quantization, desktop GPU use, and the runtime all affect actual memory use.
LocalDeploy records what happened after deployment so later estimates can be
adjusted for that machine.

It works with Ollama, llama.cpp, LM Studio, vLLM, and other loopback
OpenAI-compatible runtimes. Runtime URLs must remain local. An `API_TOKEN`
is an optional shared token, not full authentication. Keep the service on
localhost; it has no TLS, user isolation, or rate limiting. Tool calls are
returned to clients and are never executed by LocalDeploy.

Model search and the UI's release check can contact external services.
Set `OFFLINE=true` to disable those requests. Separately installed runtimes
have their own network settings; see [Security](SECURITY.md).

## Documentation

| Guide | Use it for |
|---|---|
| [Installation](docs/INSTALL.md) | Launcher commands, Docker, and setup alternatives |
| [UI guide](docs/UI.md) | Choosing, deploying, chatting with, and benchmarking models |
| [API options](docs/API_OPTIONS.md) | OpenAI-compatible API and local integrations |
| [CLI](docs/CLI.md) | Commands and automation |
| [Models](docs/MODELS.md) | Runtimes, model discovery, and fit estimates |
| [Screenshots](docs/SCREENSHOTS.md) | The complete UI tour |
| [Packaging](docs/PACKAGING.md) | Experimental desktop builds and signing limitations |
| [Contributing](CONTRIBUTING.md) | Development setup and verification |

MIT licensed. Built by [Oney Erge](https://oneyerge.com).
