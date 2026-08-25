#!/usr/bin/env python3
"""Record the animated README demo (docs/assets/demo.gif).

Same in-process app pattern as capture_screenshots.py: launches the real
server, seeds deterministic local demo data, then drives a short scripted tour
with a visible cursor while Playwright records video. The webm is converted to
a GIF with ffmpeg (the copy bundled with Playwright is found automatically, so
no separate install is needed).

The capture never requires Ollama, a downloaded model, or network access. It
intercepts only data-plane calls in the browser so the repository's real UI,
renderers, navigation, and interactions remain the subject of the recording.

Usage:
    pip install -e ".[dev]"
    python -m playwright install chromium
    python scripts/capture_demo_gif.py
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from capture_screenshots import _seed_runs  # noqa: E402  (shared helper)

OUT_PATH = ROOT / "docs" / "assets" / "demo.gif"
VIEWPORT = {"width": 1280, "height": 800}

# A soft blue dot that follows the mouse, so clicks are visible in the GIF.
CURSOR_SCRIPT = """
window.addEventListener('DOMContentLoaded', () => {
  const c = document.createElement('div');
  c.style.cssText = 'position:fixed;left:-40px;top:-40px;width:22px;height:22px;' +
    'border-radius:50%;background:rgba(79,134,247,.35);border:2.5px solid #4f86f7;' +
    'box-shadow:0 2px 10px rgba(0,0,0,.35);pointer-events:none;z-index:2147483647;' +
    'transform:translate(-50%,-50%);transition:width .12s,height .12s';
  document.body.appendChild(c);
  document.addEventListener('mousemove', (e) => {
    c.style.left = e.clientX + 'px';
    c.style.top = e.clientY + 'px';
  }, true);
  document.addEventListener('mousedown', () => { c.style.width = '15px'; c.style.height = '15px'; }, true);
  document.addEventListener('mouseup', () => { c.style.width = '22px'; c.style.height = '22px'; }, true);
});
"""


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _find_ffmpeg() -> str | None:
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", "")) / "ms-playwright"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches" / "ms-playwright"
    else:
        base = Path.home() / ".cache" / "ms-playwright"
    for hit in sorted(base.glob("ffmpeg-*/ffmpeg*")):
        if hit.is_file() and not hit.name.endswith(".txt"):
            return str(hit)
    return None


def _glide(page, selector: str, pause_ms: int = 350) -> bool:
    """Move the fake cursor smoothly onto `selector`. Returns False if absent."""
    el = page.locator(selector).first
    try:
        el.scroll_into_view_if_needed(timeout=2000)
        box = el.bounding_box()
    except Exception:
        return False
    if not box:
        return False
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=28)
    page.wait_for_timeout(pause_ms)
    return True


def _jump_to(page, selector: str, pause_ms: int = 1400) -> None:
    # Instant jump, not smooth scroll: in a GIF a scroll animation makes every
    # frame differ, which triples the file size for no narrative benefit.
    page.evaluate(
        "(sel) => { const el = document.querySelector(sel); if (el) el.scrollIntoView({behavior: 'instant', block: 'start'}); }",
        selector,
    )
    page.wait_for_timeout(pause_ms)


def _frames_to_gif(frames_dir: Path, out_path: Path, *, fps: int) -> Path:
    """Assemble PNG frames into a looping GIF with one shared adaptive palette."""
    from PIL import Image

    files = sorted(frames_dir.glob("f*.png"))
    if not files:
        raise RuntimeError("no frames extracted")
    frames = [Image.open(f).convert("RGB") for f in files]
    # Build the palette from a spread of sample frames so every scene is covered.
    step = max(1, len(frames) // 6)
    samples = frames[::step][:6]
    strip = Image.new("RGB", (frames[0].width, frames[0].height * len(samples)))
    for i, f in enumerate(samples):
        strip.paste(f, (0, i * f.height))
    palette = strip.quantize(colors=255, method=Image.MEDIANCUT)
    quantized = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    quantized[0].save(
        out_path,
        save_all=True,
        append_images=quantized[1:],
        duration=round(1000 / fps),
        loop=0,
        optimize=True,
    )
    return out_path


def _candidate(bucket: str, model_id: str, params_b: float, why: str) -> dict:
    return {
        "id": model_id,
        "pull_name": model_id,
        "family": model_id.split(":")[0],
        "params_b": params_b,
        "tier": 4,
        "vision": False,
        "use_case": "coding",
        "workload_tags": ["coding", "tools"],
        "context_native": 32768,
        "description": f"A capable local model sized for an 8 GB GPU ({params_b:g}B parameters).",
        "required_gb": round(params_b * 0.65 + 1.1, 1),
        "margin_gb": round(max(0.5, 6.0 - params_b * 0.65), 1),
        "bucket": bucket,
        "why_summary": why,
        "confidence": "medium",
        "reasons": [
            {"text": "Fits the available VRAM with working headroom", "kind": "estimated"},
            {"text": "Published 32K context window", "kind": "published"},
        ],
    }


def _install_demo_routes(page) -> None:
    """Supply deterministic, local-only data while recording the real UI."""
    running = {"value": []}
    installed = {
        "name": "gemma3:4b",
        "size": 3_300_000_000,
        "details": {"parameter_size": "4.3B", "quantization_level": "Q4_K_M"},
    }
    profile = {
        "backend": "ollama",
        "model_id": "gemma3:4b",
        "enabled": True,
        "supports_vision": True,
    }

    page.route(
        "**/profiles",
        lambda route: route.fulfill(
            json={"success": True, "default_profile": "gemma3_4b", "profiles": {"gemma3_4b": profile}}
        ),
    )
    page.route(
        "**/registry/installed",
        lambda route: route.fulfill(json={"success": True, "installed": [installed], "error": None}),
    )
    page.route(
        "**/system/hardware",
        lambda route: route.fulfill(
            json={
                "success": True,
                "gpu_available": True,
                "gpus": [{"name": "NVIDIA GPU", "vram_total_mb": 8192, "vram_free_mb": 7168}],
                "gpu_summary": {"best_pool_total_mb": 8192, "best_pool_free_mb": 7168, "gpu_count": 1},
                "system": {"ram_total_mb": 32768, "ram_available_mb": 24576},
            }
        ),
    )

    def status_route(route):
        route.fulfill(
            json={
                "success": True,
                "ollama": {"reachable": True, "running": running["value"], "error": None},
                "served_models": [item["name"] for item in running["value"]],
                "hardware": {"gpu_available": True, "gpus": [], "system": {}},
            }
        )

    def serve_route(route):
        running["value"] = [
            {
                "name": "gemma3:4b",
                "size": 3_300_000_000,
                "size_vram": 3_100_000_000,
                "placement": "GPU",
                "gpu_percent": 100,
                "expires_at": "2099-01-01T00:00:00Z",
            }
        ]
        route.fulfill(json={"success": True, "served": "gemma3:4b", "message": "Loaded for 60m."})

    page.route("**/system/status", status_route)
    page.route("**/models/serve", serve_route)
    page.route(
        "**/registry/recommend",
        lambda route: route.fulfill(
            json={
                "success": True,
                "budget_source": "vram",
                "raw_budget_gb": 8.0,
                "budget_gb": 6.0,
                "margin_relaxed": False,
                "use_case": "coding",
                "priority": "balanced",
                "expected_context": 8192,
                "recommended": _candidate(
                    "recommended", "qwen2.5-coder:7b", 7.0, "Best balance of coding quality and speed."
                ),
                "faster": _candidate("faster", "qwen2.5-coder:3b", 3.0, "Lower latency with more VRAM headroom."),
                "higher_quality": _candidate(
                    "higher_quality", "deepseek-coder-v2:8b", 8.0, "Stronger reasoning near the safe fit limit."
                ),
                "hardware": {"gpu_available": True, "gpus": []},
                "message": None,
            }
        ),
    )

    chat_events = "\n".join(
        [
            'data: {"choices":[{"delta":{"content":"Local models keep prompts, files, and tool calls on your machine"}}]}',
            "",
            'data: {"choices":[{"delta":{"content":" while giving you direct control over cost and deployment."}}]}',
            "",
            "data: [DONE]",
            "",
        ]
    )
    page.route(
        "**/v1/chat/completions",
        lambda route: route.fulfill(status=200, content_type="text/event-stream", body=chat_events),
    )
    page.route(
        "**/system/monitor",
        lambda route: route.fulfill(
            json={
                "success": True,
                "ollama_reachable": True,
                "hardware": {
                    "vram_used_mb": 4520,
                    "vram_total_mb": 8192,
                    "vram_pct": 55.2,
                    "gpu_utilization_pct": 68.0,
                    "ram_used_mb": 10120,
                    "ram_total_mb": 32768,
                    "cpu_percent": 14.0,
                },
                "history": {
                    "hardware": [
                        {"vram_pct": 42.0, "gpu_utilization_pct": 35.0},
                        {"vram_pct": 49.0, "gpu_utilization_pct": 57.0},
                        {"vram_pct": 55.2, "gpu_utilization_pct": 68.0},
                    ]
                },
                "models": [
                    {
                        "name": "gemma3:4b",
                        "placement": "GPU",
                        "gpu_percent": 100,
                        "size_mb": 3300,
                        "size_vram_mb": 3100,
                        "expires_at": "2099-01-01T00:00:00Z",
                        "requested_device": "GPU",
                        "uptime_seconds": 1500,
                        "request_count": 12,
                        "active_requests": 0,
                        "failure_count": 0,
                        "median_tokens_per_second": 35.0,
                        "recent_tokens_per_second": 38.0,
                        "median_ttft_ms": 220.0,
                    }
                ],
                "requests": [
                    {
                        "ts": 1750000000.0,
                        "profile": "gemma3_4b",
                        "model": "gemma3:4b",
                        "backend": "ollama",
                        "source": "chat",
                        "success": True,
                        "elapsed_seconds": 1.2,
                        "prompt_tokens": 14,
                        "output_tokens": 24,
                        "ttft_ms": 220.0,
                        "tokens_per_second": 38.0,
                        "context_limit": 8192,
                        "error": None,
                    }
                ],
                "alerts": [{"level": "info", "text": "Model is healthy and serving on the GPU."}],
                "note": "Seeded local demo data",
            }
        ),
    )


def main() -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print('Playwright is not installed. Run: pip install -e ".[dev]"')
        return 1

    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        print("ffmpeg not found (neither on PATH nor in the Playwright cache).")
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
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    video_path: Path | None = None
    try:
        with sync_playwright() as p, tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            browser = p.chromium.launch()
            context = browser.new_context(
                viewport=VIEWPORT,
                record_video_dir=tmp,
                record_video_size=VIEWPORT,
            )
            page = context.new_page()
            page.add_init_script(
                f'window.localStorage.setItem("localdeploy.benchmarkRuns.v1", {_seed_runs()!r});'
                'window.localStorage.setItem("localdeploy_theme", "dark");'
            )
            page.add_init_script(CURSOR_SCRIPT)
            _install_demo_routes(page)

            # --- the tour -----------------------------------------------------
            page.goto(f"{base}/ui", wait_until="networkidle")
            page.mouse.move(640, 220, steps=10)
            page.wait_for_timeout(1600)  # hardware + installed models fill in

            # 1. Curated picks for this machine's hardware.
            page.locator("#rec-use-case").select_option("coding")
            if _glide(page, "#btn-recommend-models"):
                page.locator("#btn-recommend-models").click()
                try:
                    page.wait_for_selector("#starter-pack-body .recommendation-card", timeout=8000)
                except Exception:
                    pass
                page.wait_for_timeout(1800)

            # 2. The installed-models list with live fit badges.
            _jump_to(page, "#your-models-card", pause_ms=1900)

            # 3. Chat playground: load a model, type a prompt, and stream a reply.
            if _glide(page, '.tab[data-tab="chat"]'):
                page.locator('.tab[data-tab="chat"]').click()
                page.wait_for_timeout(900)
                page.wait_for_function("document.querySelector('#chat-model')?.options.length === 1")
                page.select_option("#chat-model", "gemma3:4b")
                page.click("#btn-chat-session")
                page.wait_for_function("!document.querySelector('#chat-input')?.disabled")
                page.locator("#chat-input").click()
                page.keyboard.type("In one short sentence: why run AI models locally?", delay=16)
                page.wait_for_timeout(300)
                if _glide(page, "#btn-chat-send", pause_ms=200):
                    page.locator("#btn-chat-send").click()
                    try:
                        page.wait_for_selector(
                            ".chat-row.assistant .chat-bubble-meta:not(:empty)", timeout=45000
                        )
                        page.wait_for_timeout(1500)
                    except Exception:
                        pass

            # 4. Benchmark workspace: leaderboard, scatter, heatmap.
            if _glide(page, '.tab[data-tab="bench"]'):
                page.locator('.tab[data-tab="bench"]').click()
                page.wait_for_timeout(1300)
            _jump_to(page, "#results-dashboard-card", pause_ms=2400)
            _glide(page, "#heatmap-body", pause_ms=1500)

            # 5. Monitor the running model and recent request telemetry.
            _jump_to(page, "body", pause_ms=400)
            if _glide(page, '.tab[data-tab="monitor"]'):
                page.locator('.tab[data-tab="monitor"]').click()
                page.wait_for_selector("#monitor-models .model-title", timeout=5000)
                page.wait_for_timeout(1900)
                _jump_to(page, "#monitor-models", pause_ms=1400)

            # 6. End back on the setup tab so the GIF loops naturally.
            _jump_to(page, "body", pause_ms=400)
            if _glide(page, '.tab[data-tab="serve"]'):
                page.locator('.tab[data-tab="serve"]').click()
                page.wait_for_timeout(1400)

            video = page.video
            context.close()  # flushes the recording
            browser.close()
            raw = Path(video.path())

            # --- webm -> GIF ---------------------------------------------------
            # Playwright's bundled ffmpeg is a minimal build without a GIF muxer,
            # so decode to PNG frames (which it does support) and assemble the
            # GIF with Pillow using one shared palette (no per-frame flicker).
            frames_dir = Path(tmp) / "frames"
            frames_dir.mkdir()
            result = subprocess.run(
                [
                    # -r (not the fps filter): Playwright's minimal ffmpeg build
                    # only ships the pad/crop/scale filters.
                    ffmpeg, "-y", "-i", str(raw),
                    "-vf", "scale=820:-2",
                    "-r", "8",
                    str(frames_dir / "f%04d.png"),
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print(f"ffmpeg frame extraction failed:\n{result.stderr[-2000:]}")
                return 1
            video_path = _frames_to_gif(frames_dir, OUT_PATH, fps=8)
    finally:
        server.should_exit = True
        thread.join(timeout=5)

    if video_path:
        size_mb = video_path.stat().st_size / 1e6
        print(f"wrote {video_path} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
