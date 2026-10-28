# Tests

Most tests use the in-process FastAPI application and mocked backends. They do not need a GPU, Ollama, llama.cpp, or downloaded models.

## Install

```powershell
python -m pip install -e . --group dev
```

Browser tests also need Chromium:

```powershell
python -m playwright install chromium
```

Those tests skip when Playwright or Chromium is unavailable.

## Run

```powershell
python -m ruff check .
pytest -q
node --test tests/js/frontend-modules.test.mjs
python scripts\egress_selftest.py
```

Run one file with `pytest tests/test_guardrails.py -v`.

## Coverage map

### API and safety

- `test_api_routes.py` covers models, profiles, streaming, and unsupported operations at the HTTP layer.
- `test_guardrails.py` covers prompt, output, image, profile, and clamping limits.
- `test_security.py` covers API token enforcement and the intentionally public health and UI assets.
- `test_lan_exposure.py` covers startup behavior outside loopback.
- `test_grader_sandbox.py` covers the restricted code-grader process, blocked side effects, limits, and timeouts.
- `test_structured_outputs.py` and `test_openai_extensions.py` cover constrained output, tools, Responses, and embeddings.

### Hardware, fit, and lifecycle

- `test_hardware_cpu_ram.py` and `test_gpu_inventory.py` cover CPU, RAM, GPU, and multi-GPU discovery.
- `test_fit_tiers.py`, `test_fit_v2.py`, and `test_quant_advisor.py` cover memory estimates, warning tiers, calibration inputs, and quant choices.
- `test_device_target.py` and `test_warmup_timeout.py` cover placement and model warm-up behavior.
- `test_model_management.py`, `test_profiles_crud.py`, and `test_ollama_library_search.py` cover model and profile lifecycle, search, pull, and delete behavior.
- `test_manifest.py`, `test_monitor.py`, and `test_calibration.py` cover reproducibility, runtime monitoring, and measured corrections.

### Benchmarks

- `test_benchmark_graders.py` and `test_benchmark_registry.py` cover grader behavior and custom question-set validation.
- `test_benchmark_runner.py`, `test_benchmark_expansion.py`, and `test_benchmark_provenance.py` cover execution, repetition statistics, device data, and report provenance.
- `test_bakeoff.py`, `test_bench_history.py`, and `test_phase_b_report.py` cover candidate selection, saved history, summaries, and comparison data.

### Web UI

- `test_web_assets.py` checks the seven ES modules, import graph, cache token, static routes, and package data.
- `test_web_js_units.py` runs dependency-free Node tests for model and benchmark transforms and view builders.
- `test_ui_playwright.py` drives the real page in headless Chromium with mocked routes. It covers setup, pull, deploy, chat, attachments, benchmark queues, cancellation, import and export, comparison, and cross-module updates.
- The other `test_web_*.py` files cover control routes, registry behavior, benchmark endpoints, reports, and DOM contracts.

### Packaging and maintenance

- `test_packaging.py` checks package metadata and shipped assets.
- `test_updates.py`, `test_community.py`, and `test_utils.py` cover release checks, local benchmark sharing, paths, and URL rules.

These tests do not replace a live backend check. Use `compare_models.py` or the web benchmark with a local runtime when validating model-specific behavior.
