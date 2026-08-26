# AGENTS.md

## Project

LocalDeploy is a local-first Python web UI and OpenAI-compatible API for
discovering, selecting, running, monitoring, and benchmarking models on the
user's own hardware. It integrates with loopback runtimes such as Ollama,
llama.cpp, LM Studio, vLLM, and Docker Model Runner; LocalDeploy does not provide
cloud inference.

Use `README.md` for supported operator behavior, `SECURITY.md` before changing
network exposure, and the focused files under `docs/` for API, UI, CLI,
packaging, and deployment details.

## Commands

Development:

```powershell
python -m pip install -e ".[dev]"
.\scripts\start.ps1 -Foreground
python -m ruff check .
pytest -q
python scripts\egress_selftest.py
```

The stable cross-platform launchers are `run.bat` and `run.ps1` on Windows,
`run.command` on macOS, and `run.sh` on Linux. Keep `scripts/start.ps1` and
`scripts/start.sh` as the lower-level lifecycle implementation. The frontend is
native HTML/CSS/ES modules and has no npm build step.

Packaging and release workflows are documented in `docs/PACKAGING.md`; do not
build installers, publish packages/images, or create releases unless explicitly
requested.

## Architecture and Safety Boundaries

- `localdeploy/` contains package behavior and the bundled web frontend.
- `api_server.py` is the ASGI application; root companion modules are shipped
  console utilities.
- `scripts/` owns launch, stop, smoke, install, and egress validation workflows.
- `packaging/` owns desktop build specifications and installers.
- Keep OpenAI-compatible routes compatible with their documented request/response
  shapes and keep native lifecycle operations separate.
- Tool calls may be returned to clients but must never be executed by LocalDeploy.
- Backend runtime URLs must remain loopback-only. Do not weaken URL validation,
  bind publicly, or imply that `API_TOKEN` is full internet-grade authentication.
- Preserve `OFFLINE=true`, redaction, and no-telemetry behavior. Make every new
  outbound request explicit and cover it in the egress self-test.
- Do not commit `.env`, `config.json`, model data, benchmark state, local runtime
  paths, build outputs, or credentials.

## Change Style

- Inspect hardware/runtime detection and existing adapters before adding a new
  branch or provider.
- Make the smallest coherent change and keep platform-specific behavior isolated.
- Diagnose readiness, ports, logs, subprocess output, and runtime responses before
  retrying or changing orchestration.
- Treat packaging, installers, model deletion, process control, and network
  exposure as high-impact surfaces.

## Verification

- Documentation or guidance only: verify referenced paths and run
  `git diff --check`; application tests are not required.
- Python behavior: run the focused test, then `pytest -q`.
- Python quality: run `python -m ruff check .`.
- Network/offline behavior: run `python scripts\egress_selftest.py`.
- Frontend changes: run the relevant pytest/Playwright coverage and JavaScript
  syntax check.
- Packaging, Docker, and platform launcher changes: run only the affected smoke
  lane and report unavailable platforms explicitly.

Never claim a check ran unless its output was observed.

## Git and Handoff

- Preserve unrelated changes and keep commits focused.
- Use the configured repository-owner identity.
- Do not add assistant names, co-author trailers, session links, or tool
  attribution to Git artifacts.
- Finish with what changed, what was verified, what was skipped, and remaining
  privacy, compatibility, or release risk.


## Install and run contract

- Keep `run.bat`, `run.ps1`, `run.command`, and `run.sh` as the stable
  user entry points. They must keep the same `run`, `doctor`, `repair`,
  `docker`, `logs`, and `stop` actions where the application supports them.
- Use the `native-app-delivery` Codex skill when changing first-run setup,
  repair, Docker, or launcher behavior. That is an internal workflow name and
  must not appear in product copy or the public README.
- Keep shared install mechanics in `scripts/install-utils.ps1` and
  `scripts/install-utils.sh`. Preserve idempotent reruns, bounded transient
  retries, install locking, disk checks, user state, and `.setup/install.log`.
- Verify launcher changes with PowerShell parsing, `bash -n`, the focused
  delivery audit, and `docker compose config`. Do not run the full application
  test suite unless the change affects application behavior.
