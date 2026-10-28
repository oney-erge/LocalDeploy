#!/usr/bin/env python3
"""Capture deterministic UI screenshots for docs/SCREENSHOTS.md and the README.

Launches the real app in-process (same pattern as tests/test_ui_playwright.py),
seeds browser localStorage with a few synthetic completed benchmark runs so
the UI renders populated rather than empty, and screenshots each tab. No
Ollama, GPU, or network access is required.

Usage:
    pip install -e . --group dev
    python -m playwright install chromium
    python scripts/capture_screenshots.py
"""
from __future__ import annotations

import json
import socket
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "docs" / "screenshots"


def _best_chat_profile(base: str) -> str | None:
    """The smallest installed Ollama model name - the chat model picker lists
    installed models directly, so the scene works whatever is on the machine."""
    import json as _json
    import urllib.request

    try:
        with urllib.request.urlopen(f"{base}/registry/installed", timeout=10) as r:
            installed = _json.load(r).get("installed", [])
    except Exception:
        return None
    candidates = [(m.get("size") or 0, m["name"]) for m in installed if m.get("name")]
    return min(candidates)[1] if candidates else None


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _seed_runs() -> str:
    """Synthetic-but-plausible completed runs: 3 models x 5 categories, so the
    leaderboard, heatmap, and scatter render populated instead of near-empty."""
    profiles = [
        # (profile, model, per-category accuracy, base latency s, tok/s)
        ("qwen3vl_8b_ollama", "qwen3-vl:8b-instruct",
         {"reasoning": 0.95, "math": 0.90, "code": 0.88, "structured_json": 0.93, "summarization": 0.91}, 3.1, 27.0),
        ("gemma3_4b_ollama_safe", "gemma3:4b",
         {"reasoning": 0.86, "math": 0.74, "code": 0.71, "structured_json": 0.88, "summarization": 0.90}, 1.8, 42.0),
        ("llama32_3b_ollama", "llama3.2:3b",
         {"reasoning": 0.78, "math": 0.61, "code": 0.66, "structured_json": 0.83, "summarization": 0.85}, 1.1, 55.0),
    ]
    runs = []
    for i, (profile, model, cats, latency, tps) in enumerate(profiles):
        tests = []
        for j, (cat, acc) in enumerate(cats.items()):
            for k in range(2):
                accuracy = round(min(1.0, max(0.0, acc + (0.04 if k else -0.03))), 2)
                tests.append(
                    {
                        "name": f"{cat}_{k + 1}",
                        "category": cat,
                        "success": accuracy >= 0.5,
                        "accuracy": accuracy,
                        "elapsed_seconds": round(latency * (1 + 0.15 * ((j + k) % 3)), 2),
                        "approx_tokens_per_second": round(tps * (1 - 0.05 * (j % 2)), 1),
                    }
                )
        passed = sum(1 for t in tests if t["success"])
        runs.append(
            {
                "id": f"run-{i}",
                "createdAt": "2026-07-05T00:00:00.000Z",
                "profile": profile,
                "modelId": model,
                "source": "restored-history",
                "tests": tests,
                "summary": {
                    "tests": len(tests),
                    "passed": passed,
                    "avg_accuracy": round(sum(t["accuracy"] for t in tests) / len(tests), 3),
                    "avg_latency_s": round(sum(t["elapsed_seconds"] for t in tests) / len(tests), 2),
                    "avg_tokens_per_second": round(sum(t["approx_tokens_per_second"] for t in tests) / len(tests), 1),
                },
            }
        )
    return json.dumps(runs)


def main() -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print('Playwright is not installed. Run: pip install -e . --group dev')
        return 1

    import uvicorn

    from api_server import app

    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 20
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    if not server.started:
        print("uvicorn did not start within 20s")
        return 1

    base = f"http://127.0.0.1:{port}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception as exc:
                print(f"chromium not available: run 'python -m playwright install chromium' ({exc})")
                return 1
            try:
                def capture(theme: str, shots: list[tuple[str, str]]) -> None:
                    """Screenshot each (tab, filename) in the given UI theme."""
                    page = browser.new_page(viewport={"width": 1440, "height": 960})
                    page.add_init_script(
                        f'window.localStorage.setItem("localdeploy.benchmarkRuns.v1", {json.dumps(_seed_runs())});'
                        f'window.localStorage.setItem("localdeploy_theme", "{theme}");'
                    )
                    page.goto(f"{base}/ui", wait_until="networkidle")
                    for tab, filename in shots:
                        if tab:
                            page.get_by_role("tab", name=tab).click()
                            page.wait_for_timeout(400)
                        page.screenshot(path=str(OUT_DIR / filename))
                        print(f"wrote {OUT_DIR / filename}")
                    page.close()

                # Dark is the app's default look; one light shot shows the toggle.
                capture("dark", [("", "setup-deploy.png"), ("Benchmark & Compare", "benchmark-compare.png")])
                capture("light", [("", "setup-deploy-light.png")])

                # Model catalog with live results (needs internet; skipped offline).
                page = browser.new_page(viewport={"width": 1440, "height": 960})
                page.add_init_script('window.localStorage.setItem("localdeploy_theme", "dark");')
                page.goto(f"{base}/ui", wait_until="networkidle")
                try:
                    page.click('.seg-btn[data-seg="hf"]')
                    page.wait_for_selector(".catalog-table tbody tr", timeout=45000)
                    page.fill("#hf-search", "qwen")
                    page.wait_for_timeout(2500)
                    page.locator("#get-model-card").screenshot(path=str(OUT_DIR / "model-catalog.png"))
                    print(f"wrote {OUT_DIR / 'model-catalog.png'}")
                except Exception as exc:
                    print(f"skipped model-catalog.png (no network? {exc})")
                finally:
                    page.close()

                # Chat playground with a real streamed reply. Needs a model that
                # can answer, so this shot is skipped (not failed) without Ollama.
                page = browser.new_page(viewport={"width": 1440, "height": 960})
                page.add_init_script('window.localStorage.setItem("localdeploy_theme", "dark");')
                page.goto(f"{base}/ui", wait_until="networkidle")
                page.get_by_role("tab", name="Chat").click()
                page.wait_for_timeout(500)
                try:
                    chat_model = _best_chat_profile(base)
                    if not chat_model:
                        raise RuntimeError("no installed model to chat with")
                    page.select_option("#chat-model", chat_model)
                    page.click("#btn-chat-session")  # explicit Load model step
                    page.wait_for_selector(".chat-session-state.ready", timeout=120000)
                    page.fill("#chat-input", "In one short sentence: why run AI models locally?")
                    page.click("#btn-chat-send")
                    page.wait_for_selector(".chat-row.assistant .chat-bubble-meta:not(:empty)", timeout=60000)
                    page.wait_for_timeout(400)
                    page.screenshot(path=str(OUT_DIR / "chat-playground.png"))
                    print(f"wrote {OUT_DIR / 'chat-playground.png'}")
                except Exception as exc:
                    print(f"skipped chat-playground.png (no model answered: {exc})")
                finally:
                    page.close()
            finally:
                browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=5)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
