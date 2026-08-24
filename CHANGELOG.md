# Changelog

All notable changes to this project should be documented here.

## 0.6.1 - 2026-08-24

- Published the Python package to PyPI and added a GHCR container publishing workflow with a
  post-publish health check, build provenance, and an SBOM.
- GitHub releases now receive the wheel and source archive as downloadable assets.
- Limited push CI to `main`, added cancellation for superseded runs, and validated every workflow
  with actionlint.
- Updated project and installer authorship to iodriller and removed a tool-specific repository
  guidance file.
- Ran the macOS packaging workflow successfully on a GitHub-hosted Apple Silicon runner, including
  building the app, launching it, checking `/health`, and creating the DMG.
- Renamed the GitHub account from iodriller to oney-erge; updated repository URLs, package and
  installer authorship, and the Windows registry key path to match.

## 0.6.0 - 2026-07-19

### ModelScope search and direct GGUF import

- Added ModelScope as a third model source alongside Ollama and Hugging Face in the unified
  catalog search, pulled through Ollama's native `modelscope.cn/<repo>:<file>.gguf` support
  (Ollama 0.30+) - no new pull mechanism needed. Curated each repo's exposed quant variants down
  to a standard ladder (Q3/Q4/Q5/Q6/Q8/F16) instead of listing every file a repo ships (some
  unsloth-style repos carry 20+ near-duplicate quant/IQ variants), rounded ModelScope's
  full-precision parameter counts for display (`9.197093888B` to `9.2B`), and fixed a quant-label
  regex gap that left period-separated filenames (`Model.Q4_K_M.gguf`) with no visible quant badge.
- Added GGUF import for models that aren't in any registry: a local file already on disk (becomes
  a llama.cpp run profile, no download) or a direct URL such as a GitHub Release (downloaded,
  registered with Ollama via `/api/create`, and given a run profile - so Deploy/Unload/Delete work
  exactly like a pulled model).

### Fixed

- `chat.js`'s "Open model catalog" button (shown when no models are installed) called an
  undefined function and did nothing; it now navigates correctly.
- Every tab switch re-ran hardware detection and an internet update-check on top of the model/
  profile refresh already needed for correctness, and switching tabs quickly could stack up
  several overlapping bursts of requests since navigation didn't wait for the previous switch's
  refresh to finish. Tab navigation now only refreshes Ollama reachability; hardware/update checks
  stay on their original triggers (startup, the explicit button, and actions that actually change
  VRAM usage).
- `start.ps1` could report "LocalDeploy did not start" on a successful launch if no default browser
  was configured, because opening the browser ran under the same fail-fast error handling as the
  rest of the script.
- `start.ps1`/`stop.ps1` could leave a stale server running after `-Restart`: on a venv created from
  a Conda base interpreter, this machine transparently relaunches a script under a different
  interpreter as a child process, and the launcher only ever tracked the parent PID. Both scripts
  now identify whoever is actually listening on the configured port and walk up its process-family
  ancestry, so the whole chain is stopped regardless of which hop originally got recorded.
- `install.ps1` claimed it would `git pull` an existing install to update it but never actually did;
  it now pulls (fast-forward only) when the checkout is clean, and skips with a clear warning
  instead of overwriting anything if it isn't.
- The Docker LAN-exposure warning's closing line ("Bind API_HOST=127.0.0.1") was shown even inside
  a container, where binding to loopback internally would break the port mapping; that line is now
  Docker-aware, matching the reassurance already given earlier in the same message.
- Added `start.command` so macOS users can double-click to launch, the same way `start.bat` already
  works on Windows (a plain `.sh` opens in a text editor on double-click instead of running).
- `start.sh` now offers to install Python and Ollama itself when either is missing (Homebrew on
  macOS; apt, dnf, pacman, or zypper for Python and the official installer script for Ollama on
  Linux), asking before running anything and never prompting on a non-interactive run - mirroring
  the guided install `start.ps1` already offered via winget on Windows.
- The repository is now public; the previously-private repo made every documented install path
  (the README one-liner, `install.ps1`'s own ZIP fallback, and anonymous `git clone`) 404 for
  anyone without repository access.

### Desktop installers (experimental)

- Windows: `.\packaging\build.ps1 -Msi` produces a per-user `.msi` (no admin/UAC prompt), fetching
  the WiX v3 toolset itself on first run. Verified end-to-end on a real machine: install, Start
  Menu + Desktop shortcuts, a working Settings > Apps entry, and a fully clean uninstall.
- macOS: `packaging/macos.spec` builds a menu-bar-only `.app` (no Dock icon), and
  `.github/workflows/build-macos.yml` packages it into a `.dmg` with an automated smoke test.
  The workflow has passed on a GitHub-hosted Apple Silicon runner. The resulting app still needs a
  manual desktop launch before the unsigned DMG should be treated as a normal end-user install.
- Neither installer is code-signed yet: Windows SmartScreen and macOS Gatekeeper will both warn on
  first run. See `docs/PACKAGING.md` for what each warning means and the signing/notarization cost
  and account requirements to remove them.

## 0.5.1 - 2026-07-18

### Public release audit

- Split the browser frontend into seven native ES modules and added import-graph, browser, package,
  and static-route checks for the new layout.
- Fixed several audit findings in benchmark and model workflows: streamed backend failures now end
  cleanly, benchmark queue failures no longer stay stuck as running, forced device comparisons load
  one model at a time, imported report cards keep stable ids, category summaries include every
  category, zero-valued limits reach validation, and single-profile reports retain backend data.
- Reworked the README and supporting guides in a shorter, factual voice. Added checks for local
  documentation links and the repository's punctuation rule. Corrected two nonexistent model names
  in the old reference table and removed recommendations that could not be kept current.
- Documented the first PyPI release setup. The v0.5.0 package build succeeded, but upload did not
  because the pending trusted publisher had not been registered on PyPI.
- Added Ruff to local and CI checks, and pinned GitHub Actions and the Docker base image by digest.
- Disabled Ollama's separate cloud-model feature in the bundled Docker runtime and in Ollama
  processes started by the Windows launcher. Clarified the boundary between LocalDeploy's offline
  setting and the network settings of a separately managed runtime.
- Fixed the macOS/Linux launcher so API and Ollama addresses set only in `.env` are used for its
  health checks, browser link, and server command.
- Added managed Ollama startup to the macOS/Linux launcher and a conservative `stop.sh` that only
  stops processes launched by this checkout.
- Consolidated runtime dependency metadata around `requirements.txt`; development tools now live
  only in the `.[dev]` package extra.
- Expanded the starter catalog with verified Qwen 3.5 local tags, added five data-only benchmark
  questions, and added a safe `json_keys_present` grader for custom question sets.
- Updated client examples to use the configured default profile and optional API token. Marked shell
  entry points executable for clones on macOS and Linux.

### Later on 0.5.1: guided recommendations, calibration, Monitor tab, reproducibility

- Recommended models is now a guided flow. Pick a use case, a priority (quality/speed/memory/
  context), and an expected context size; `POST /registry/recommend` returns up to three labeled
  picks (Recommended, Faster, Higher quality), each with a "why this model?" breakdown that tags
  every reason as estimated, published, or measured on this machine without blending the three.
- Fit estimates now calibrate themselves. Every deploy compares the pre-load VRAM estimate
  against what Ollama actually placed, and later fit checks for the same GPU/model/quant/context
  are corrected from that history (shown as a visible correction, never applied silently). Fit
  checks also report a confidence level and a new `/system/fit-table` sweeps several context sizes
  in one call.
- New Monitor tab: live VRAM/CPU/RAM and per-model throughput/TTFT while a model serves, a
  numeric-only request log (prompts and responses are never recorded), rule-based alerts (sustained
  VRAM pressure, placement mismatches, slow generation, true concurrent calls), and persisted
  session summaries. Calibration only consumes Ollama's model-specific VRAM allocation; system-wide
  session peaks remain diagnostic so unrelated GPU use cannot corrupt fit estimates.
- Benchmarking: true time-to-first-token, P90/P95/median/min/max statistics on repeated runs, a
  context-scaling sweep, use-case benchmark packs (coding/structured/reasoning/general), and
  regression detection in Compare (what changed between two runs' runtime/digest/quant/GPU, plus
  the resulting tok/s/TTFT/VRAM deltas).
- Deployment manifests: export a running model's exact configuration as YAML/JSON, validate it
  against a different machine (compatibility report with safe substitutions), recreate it there, and
  generate copy-paste config for Open WebUI, Continue, Cline, curl, Python, JavaScript, and Docker
  Compose.
- Update check: an opt-out (`OFFLINE=true`), no-payload GitHub release check surfaces a chip
  when a newer version exists. A local-only anonymized benchmark-sharing preview/export was added
  as groundwork for community sharing: there is no server to submit to yet, so nothing is sent.
- Fixed a UI crash that broke hardware detection, catalog search, and other buttons. Disabling a
  just-clicked button (to show a loading state) fires a `focusout` event; a tooltip handler
  dereferenced its trigger without checking for null and crashed before the underlying request ever
  ran. Also fixed: a race in `benchmark.py`'s lazy import that could return a partially-initialized
  module under concurrent first requests; the Monitor tab's "simultaneous requests" alert now uses
  real in-flight backend calls instead of a hardcoded or recent-completion proxy; Monitor's per-model state lookup
  used exact string matching against Ollama's reported name instead of the same fuzzy matcher used
  elsewhere, so a bare/tagless deploy could show blank uptime; and `compare_models.py` crashed
  instead of printing a helpful message when no profile was configured anywhere.
- An automated "pull top candidates, benchmark each, keep the winner" workflow was built and then
  disabled in the UI pending a product decision (backend intact at `/system/bakeoff/run`).

- The catalog answers "will it fit?" at a glance. Since your hardware is already
  detected, every size chip in the results is colored by fit: green fits your GPU,
  yellow is tight/CPU-only, and red won't fit. One batched estimate runs per search
  (`POST /system/fit-batch`); Hugging Face rows get a row-level fit badge parsed from
  the repo name. Descriptions clamp to two lines and capability tags sit inline, so
  the model column stops being a wall of text.
- Pulling a specific quant is now one click on a real tag. The quant advisor
  fetches the family's actual published tags from ollama.com
  (`POST /registry/library-tags`), adds a "Pull it" column with the true download size
  (e.g. `↓ 3.0GB` for `qwen2.5:7b-instruct-q5_K_M`), marks unpublished quants honestly
  as "not published", flags already-pulled ones, and offers an expandable list of every
  published tag: no more guessing tag-name conventions.

- One search, every source. The Model catalog no longer asks you to pick a source:
  one query hits the Ollama library and Hugging Face GGUF repos in parallel
  (`POST /registry/search-models`) and renders a single table where the source is just a
  column. Results update as you type (debounced), size chips are per-tag fit-checked
  pulls, exact tags (`gemma3:4b`, `hf.co/org/repo`) get a verbatim "Pull exactly" row,
  and ⚖ on any row opens the quant advisor pre-filled.
- Chat upgrades: attach text files (contents ride along as fenced blocks the model
  can read and render as code blocks in your bubble), a 5-minute keep-loaded option,
  and `keep_alive` now rides every chat request so Ollama stops silently resetting the
  session's keep-loaded choice to its 5-minute default after the first message.
  Assistant replies render lightweight markdown (headings, bullets, bold/italic, inline
  code, and links). Rendering is still escape-first, with no HTML injection possible.
- Buttons look like buttons everywhere: one uniform shape/height/alignment for all
  buttons (emoji and icon labels now center properly), squared-off attach control.

### Earlier on 0.5.1 (2026-07-17)

UX overhaul across the whole UI.

- Model catalog is now searchable. The local provider inventory gets a search box,
  runtime and size filters, a sort control (measured tok/s, name, parameters, runtime),
  and pagination instead of one giant unfiltered table. Provider chips show
  reachability with per-runtime "how to enable" hints, and a Discover section links
  Hugging Face GGUF search and the Ollama library.
- Disk vs VRAM are visually distinct everywhere. Model rows show `💾 X GB disk`
  (drive space) separately from `⚡ needs ~Y GB VRAM` (memory to run), with a legend on
  the Your models card; the serving card labels "⚡ VRAM in use" vs "💾 Size on disk".
- Your models polish: fit checks still run automatically on load (the button is now a
  small ↻ re-check), icon buttons (▶ Deploy, ⚙ Tune, 🗑 Delete), and the delete dialog
  states exactly how many GB it frees.
- Use via API: every served model card shows the full OpenAI-compatible endpoint URL
  with one-click ⧉ copy buttons for the URL and a ready-to-run curl (model pre-filled).
- Chat redesigned as a proper chat shell: header with model picker + system-prompt
  panel, welcome screen with clickable suggestion prompts, role avatars, fenced code
  blocks rendered with language tag and copy button, auto-growing composer, and
  🕒 / ⚡ metadata under each reply.
- One-paste Windows install: `irm …/scripts/install.ps1 | iex` downloads the repo
  (git or ZIP: nothing preinstalled) and hands off to the guided start script.
- Screenshot/GIF capture scripts now pick a chat model that is actually installed, so
  they don't record a timeout when a hardcoded model was deleted.

## 0.5.0 - 2026-07-17

- Detect NVIDIA, AMD, Intel, and Apple GPUs and estimate compatible multi-GPU placement without combining mixed runtime pools.
- Add a dense local-provider model catalog for Ollama, llama.cpp, LM Studio, vLLM, Docker Model Runner, and configured loopback OpenAI-compatible runtimes, including saved benchmark tok/s.
- Record repeated-run variance and benchmark provenance: LocalDeploy/Ollama versions, full model digest, quant, context, initial warm/cold state, native backend metrics, and hardware snapshot.
- Add OpenAI-compatible tool calling, `/v1/responses`, and `/v1/embeddings` with float/base64 output. LocalDeploy returns tool calls but never executes them.
- Clarify the single-user local-only security boundary: one shared token, no TLS, no multi-user isolation, and no claim of internet-facing readiness.
- Project layout: `api_server.py`, `benchmark.py`, `chat_cli.py`, and `compare_models.py`
  moved into the `localdeploy` package (`localdeploy/server.py` etc.); the root files remain
  as alias shims (`sys.modules` aliasing), so `python api_server.py`, `uvicorn api_server:app`,
  and every existing import keep working unchanged. `pytest.ini` folded into `pyproject.toml`.
- Clean-machine startup hardening: a double-clickable `start.bat` (execution-policy safe,
  works from a plain ZIP download), dependencies re-install automatically when
  `requirements.txt` changes after a `git pull`, `start.ps1` heals a stale API from an older
  checkout instead of declaring it "already running" (plus a `-Restart` switch), Ollama startup
  is polled rather than a fixed sleep, and `start.sh` now gives actionable guidance when
  Python 3.10+/venv/Ollama are missing.

## 0.4.0 - 2026-07-17

Public-launch release: a chat playground, quant advisor, disk usage tools, durable
benchmark history, and streamlining so the app only ever shows models that are
actually on your machine.

### New features

- Chat playground tab. Talk to any pulled model right in the UI: streaming
  tokens over the server's own OpenAI-compatible `/v1/chat/completions`, image
  attachments for vision-capable profiles, an optional system prompt, and a
  Send-becomes-Stop button. The reply meta line separates model-load time
  ("first token X s") from true generation speed (tok/s).
- Quantization advisor (Get a model → ⚖ Quant advisor). Fit-checks every
  common GGUF quant (Q2_K → F16) of a model size against your VRAM budget with
  the same estimator as fit checks, and tells you when there's headroom for a
  higher-quality tag than the usual Q4 default (`POST /system/quant-advisor`).
- Disk usage panel. The Your models card now shows `N models · X GB on disk`,
  sorts by size / recency / name, and bulk-deletes selected models with a
  freed-gigabytes preview.
- Opt-in server-side benchmark history. A toggle on the History tile mirrors
  completed runs to `reports/benchmark-history/` as one JSON file per run
  (`/benchmark/history` endpoints), so results survive the browser and can be
  shared as plain files. Run ids are strictly validated (no path traversal).

### No more phantom models

- Fresh installs start truly empty. `load_config()` no longer falls back to
  `config.example.json` when `config.json` is missing (that fallback made a fresh
  clone list a dozen sample profiles for models that were never pulled, most
  visibly on macOS/Linux, where `start.sh` doesn't seed a config). A missing
  config now yields zero profiles; pulling a model auto-creates its profile.
  `config.example.json` remains as field-reference documentation and a test fixture.
- `.env.example` no longer pins `DEFAULT_MODEL_PROFILE`. The pinned sample value
  made `/chat` fail with "Unknown profile" on a fresh install until that exact model
  was pulled, even after other models were. The config's `default_profile`
  (auto-set on first pull) is now used.
- Clear first-run error. With no profiles configured, `/chat` and friends now say
  "pull a model first" instead of "Unknown profile 'gemma3_4b_ollama_safe'".
- UI hides profiles whose model isn't on the machine. The benchmark model picker
  hides them by default behind a "Show N hidden (model not pulled)" toggle, the
  profile dropdowns annotate them, and a new Remove not-pulled profiles button
  (Advanced → All run profiles) deletes them in one click. For llama.cpp profiles the
  server now reports whether the GGUF file exists (`model_file_exists` on `/profiles`),
  so dead file paths get the same treatment as un-pulled Ollama models.

### Packaging

- Python package layout. The project builds as a standard Python package
  (`pyproject.toml`) with a `localdeploy` console command that serves the API + UI
  and opens the browser. The web UI ships as package data (`localdeploy/web/`,
  moved from the repo-root `web/`).
- App home for runtime state. Installed runs keep `.env`, `config.json`,
  `logs/`, and `reports/` in `~/.localdeploy` (override with `LOCALDEPLOY_HOME`);
  source checkouts keep using the repo root, so nothing changes for `git clone`
  users. Docker now mounts that state separately from Ollama's model volume, so
  profiles and benchmark history survive container recreation. `/favicon.ico`
  is now served, so `/docs` page views stop logging 404s.

### Fresh coat of paint

- New logo (`localdeploy/web/logo.svg`), regenerated favicon, and a repo social banner
  (`docs/assets/banner.png`).
- README rebuilt around an animated demo GIF (`docs/assets/demo.gif`), captured by the
  new `scripts/capture_demo_gif.py`; screenshots now ship in dark (default) and light
  themes via `scripts/capture_screenshots.py`.
- The web asset cache-bust test no longer pins the exact `?v=` token, so bumping asset
  versions can't silently break CI.

### Security and reliability

- Updated `python-multipart` to `0.0.31` after dependency auditing found three
  advisories affecting `0.0.28`; CI now audits the pinned runtime requirements.
- Bundled benchmark and chat clients now forward `API_TOKEN`, so enabling auth
  no longer breaks server-initiated benchmarks or local CLI calls.
- The code benchmark worker now validates candidate ASTs, exposes only a small
  builtin/import allowlist, and blocks common filesystem, process, network,
  registry, and native-library operations in addition to its existing timeout.
- Removed the browser-triggered `pip install psutil` endpoint. `psutil` is a
  required dependency, and hardware probe failures now degrade to a read-only
  status instead of modifying the running Python environment.

## 0.3.0 - 2026-07-08

First public release.

### Repo cleanup for public release

- One start command per platform. `scripts/start.ps1` (Windows) now opens the UI by default
  (`-NoBrowser` to skip, `-Foreground` for live logs) and `scripts/start.sh` (macOS/Linux) opens
  the UI once the server is healthy. Removed the redundant launchers: `run.ps1` / `run.sh`
  (curl-pipe bootstrappers that auto-installed Docker), `install.ps1` (start.ps1 already creates
  config/venv; model pulls belong in the fit-checked UI), and `scripts/start_ui.ps1` (now the
  default behavior).
- Renamed the root model-comparison CLI `test_models.py` → `compare_models.py` (it was never a
  pytest suite; the old name needed a pytest.ini workaround).
- Docs: `docs/TERMINAL_TESTING.md` → `docs/CLI.md`; removed the internal `GAPS.md` and
  `docs/VERIFICATION.md` trackers; rewrote the README around a per-platform quick start.
- License switched from "all rights reserved" placeholder to MIT.

### Features and fixes

- Forced CPU/GPU benchmarks now measure the device they ask for. Previously a
  `device=cpu` (or `gpu`) run only pinned the placement at warm-up; the benchmark's own
  `/chat` calls didn't pass `num_gpu`, so Ollama could silently re-place the model (a CPU
  run drifting onto the GPU). `num_gpu` is now threaded through `/chat` → `execute_test` →
  `iter_run`, so every inference call stays on the requested device end-to-end. Verified:
  `device=cpu` reports `actual_device=cpu`, `device=gpu` reports `gpu`.
- Fewer spurious "status failed" benchmark rows. Forcing CPU/GPU used to hard-fail
  whenever Ollama placed the model differently (e.g. a model too big for pure GPU lands on
  Split). It now warns and proceeds, recording the run with its actual placement.
- Benchmark history/queue management: per-run delete (×) in the run library, a confirm on
  Clear history, and the ability to dismiss finished/failed queue rows (individually or via
  Clear finished); failed rows show their error reason inline. The running-model Kill
  button is now Unload for consistency.
- Browser UI smoke tests (`tests/test_ui_playwright.py`): optional Playwright-driven checks
  that load the real `/ui` in headless Chromium and exercise tab switching and the benchmark
  history controls. They skip cleanly when Playwright or its browser isn't installed; added
  `playwright` to the development dependencies.
- Fixed: the web UI was completely broken: `web/app.js` contained smart/curly quotes
  (`“ ”`) used as string delimiters in the "Check New Models" function, a `SyntaxError` that
  prevented the *entire* script from parsing (blank/dead UI). Replaced with straight quotes.
  Added a CI step (`node --check web/app.js`) and a pytest guard so a JS parse error can never
  ship silently again (the Python-only suite never loaded the UI before).
- Benchmark workspace V2: the benchmark tab is now a local experiment workspace with a
  multi-profile Run Builder, sequential run queue, leaderboard, category heatmap, SVG
  speed/quality scatter, per-test matrix, local browser history, selected-run comparison, and
  report-card import/export. CPU + GPU now creates two queued benchmark batches instead of a
  separate special-case button.
- Benchmark queue UX: waiting benchmark rows can now be moved up/down or removed before they
  run, while the active run appears only in the main progress panel with elapsed time.
- Comparison auto-select: fresh benchmark results now join the existing selected comparison
  set instead of replacing it, so a second run appears next to the first run automatically.
- Per-test matrix demoted: the per-test matrix is now collapsed as an advanced diagnostic
  view instead of occupying the main results dashboard by default.
- Benchmark deployments are temporary: `/benchmark/run` now unloads each Ollama model after
  its profile finishes, including forced CPU/GPU benchmark deployments. The benchmark tab no
  longer leaves a model served as a side effect.
- Quieter device auto-detect (inline note instead of a per-run toast); the run progress bar is
  now an ARIA `progressbar` so screen readers announce progress.

- Honest device tag: the benchmark Device tag now defaults to Auto (detect) and is
  resolved from the model's *actual* placement (GPU/CPU/Split via `/system/status`) after the
  run, so report cards can't be silently mislabelled. Manual GPU/CPU stays an override and warns
  if it disagrees with what's detected.
- Apple Silicon (Metal) detection: `/system/hardware` now reports Apple Silicon GPUs instead
  of claiming "CPU-only" on a Mac. Unified memory is surfaced as such (no false VRAM figure; fit
  checks use system RAM). NVIDIA still takes precedence where present.
- Pull fit-gate respects the tiered warnings: a model that won't fit VRAM but runs on CPU
  is no longer blocked behind the override: the gate now hard-blocks only the "fits nowhere"
  (`severity: hard`) case and notes the CPU fallback for the soft case.
- Tune for my GPU labels skipped CPU-capable profiles as "CPU-only (skipped for GPU tuning)"
  instead of the misleading "won't fit VRAM".
- Run table: detailed benchmark rows now live below the dashboard and can be filtered by
  model, category, pass/fail, and slowest failures.
- Warm-up robustness: Deploy/replace actions now show a live "Loading…Ns" counter (with a
  "large models on CPU can take a minute" hint when targeting CPU) instead of an apparently
  frozen button. The server's load timeout scales with the device (longer for CPU offload)
  and is overridable via `OLLAMA_LOAD_TIMEOUT`; a load timeout returns a clear "it may still be
  loading: click Refresh status" message rather than a generic failure.
- Decision-grade run results: the benchmark workspace shows queued model/device variants,
  live progress, Cancel, tok/s per test, inline failure reasons, response previews,
  leaderboard ranking, per-category heatmap, and per-test matrix views.
- Report cards & compare carry speed: exported cards now include `avg_tokens_per_second`,
  a per-test tok/s column, and a By category table. The A/B compare view adds a
  tok/s (A → B) column and aggregate tok/s delta, so a "Qwen/GPU vs Qwen/CPU" diff
  shows the speed difference directly.
- Better discovery (Phase 5): "New on Hugging Face" now has a search box, results-per-query
  limit, and GGUF-only toggle: the API already accepted `queries`/`limit`/`gguf_only`, the UI
  now surfaces them. Installed rows show quantization level, parameter size, and modified date.
  HF candidates show download counts and likes alongside the modified date.
- CPU-vs-GPU benchmark comparison (Phase 6): the benchmark tab gains a *Device tag*
  (Auto / GPU / CPU) selector. The tag is stored in the exported report card (`device` field) and
  surfaces in the card header (`[GPU]`/`[CPU]` next to the model name in HTML and Markdown). The
  compare view labels runs as `model/GPU` vs `model/CPU` so A/B diffs are unambiguous.
- Tiered fit warnings (`/system/fit-check`): adds `tier`/`severity`/`headline`/`cpu_deployable`
 : green (comfortable), yellow (tight, or won't-fit-GPU-but-runs-on-CPU), red (too big anywhere) -
  using system RAM to judge CPU deployability. The coarse `verdict` is unchanged (backward-compatible).
- Model management: `POST /models/delete` (remove a model from disk) and `POST /models/free`
  (unload all models from memory/VRAM), with Delete buttons per installed model, a Free memory
  button, and a Cancel button to abort an in-flight pull.
- Hardware panel now shows CPU + RAM (`/system/hardware`): CPU model, physical/logical cores,
  and system RAM total/available via `psutil` (graceful fallback when absent).
- Choose CPU vs GPU at deploy time: the Serve panel has a *Deploy to* selector (Auto / GPU /
  CPU). Forces Ollama `num_gpu` (0 = CPU, max = GPU); Auto is unchanged from before. `/system/status`
  now labels each running model's placement (GPU / CPU / Split N%). Additive and backwards-compatible.
- Added opt-in token auth: set `API_TOKEN` to require it (`X-API-Token` /
  `Authorization: Bearer` / `?token=`); no auth and zero overhead when unset.
- "Check New Models" now filters to GGUF repos and offers a one-click Pull via Ollama's
  `hf.co/<id>` shortcut.
- Added an optional web UI at `/ui` (two tabs: Serve & Diagnose, Deploy & Benchmark);
  static, no build step, gated by `ENABLE_WEB_UI` (default on). See `docs/UI.md`.
- Added a control-plane API: `/system/hardware`, `/system/fit-check`, `/system/status`,
  `/system/recommend`, `/system/set-default`, `/registry/installed`, `/registry/check-updates`,
  `/models/pull`, `/models/serve`, `/models/stop`, `/models/switch`.
- Added benchmark-over-HTTP: `/benchmark/example`, `/benchmark/validate`, `/benchmark/run`
  (streamed), plus shareable report cards (`/benchmark/export`) and A/B compare
  (`/benchmark/compare`). `benchmark.py` gained an importable `execute_test`/`iter_run` and a
  JSON-safe grader registry shared by the CLI and the API.
- Added one-command Docker run (`Dockerfile`, `docker-compose.yml`) bundling Ollama + the API/UI,
  plus a `scripts/start.sh` launcher.
- Added a verifiable offline mode (`OFFLINE=true`) and `scripts/egress_selftest.py`.
- The original CLI, OpenAI-compatible API, and loopback-only backend guard are unchanged.

## 0.2.0 - 2026-05-16

- Added OpenAI-compatible `/v1/chat/completions` and `/v1/models` endpoints for local clients.
- Added GitHub project hygiene files, CI validation, issue templates, and contribution/security docs.
- Added terminal chat helper and local benchmark documentation.

## 0.1.0 - 2026-05-16

- Initial local Ollama-first deployment server.
- Added optional llama.cpp profile support.
- Added safe model profiles for Gemma 3 4B and Gemma 3 12B testing.
- Added benchmark runner and Windows install helper.
