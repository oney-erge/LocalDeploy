"""Tests for the Monitor tab backend (Release R3): request/session tracking,
alerts, and the /system/monitor snapshot endpoint. CI has no GPU/Ollama, so
the endpoint test monkeypatches the lazy-imported hardware/_ollama calls at
their source modules (system_monitor() imports them inside the function).
"""
from __future__ import annotations

import time

import pytest

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover
    pytest.skip("FastAPI TestClient requires httpx", allow_module_level=True)

from api_server import app
from localdeploy.control import _ollama, calibration, hardware, monitor

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolate_monitor_state(monkeypatch, tmp_path):
    monitor.reset_state()
    monkeypatch.setattr(monitor, "_sessions_dir", lambda: tmp_path / "monitor-sessions")
    monkeypatch.setattr(calibration, "_store_path", lambda: tmp_path / "calibration.json")
    yield
    monitor.reset_state()


def _hw(vram_used_mb=4096, vram_total_mb=8192, ram_available_mb=16384, ram_total_mb=32768, gpu_util=50.0, cpu_pct=10.0):
    return {
        "success": True,
        "gpu_available": True,
        "gpus": [
            {
                "name": "Fake GPU", "vendor": "NVIDIA", "backend": "CUDA",
                "vram_used_mb": vram_used_mb, "vram_total_mb": vram_total_mb,
                "utilization_pct": gpu_util,
            }
        ],
        "system": {"ram_available_mb": ram_available_mb, "ram_total_mb": ram_total_mb, "cpu_percent": cpu_pct},
        "message": None,
    }


def test_note_serve_then_stop_persists_session(tmp_path):
    monitor.note_serve("gemma3:4b", "GPU")
    monitor.record_request(
        profile="gemma3_4b", model="gemma3:4b", backend="ollama", kind="chat", success=True,
        elapsed_seconds=1.2, metrics={"tokens_per_second": 30.0, "ttft_ms": 300.0}, context_limit=4096,
    )
    summary = monitor.note_stop("gemma3:4b")
    assert summary is not None
    assert summary["model"] == "gemma3:4b"
    assert summary["request_count"] == 1
    assert summary["failure_count"] == 0
    assert summary["median_tokens_per_second"] == 30.0
    assert summary["median_ttft_ms"] == 300.0
    files = list((tmp_path / "monitor-sessions").glob("*.json"))
    assert len(files) == 1


def test_note_stop_without_serve_state_is_none():
    assert monitor.note_stop("never-served:1b") is None


def test_note_stop_does_not_calibrate_from_system_wide_vram():
    # Hardware sampling reports whole-machine GPU usage, not this model's
    # allocation. Session summaries may retain the peak for diagnostics, but
    # only Ollama's model-specific size_vram reading is valid calibration input.
    monitor.note_serve("qwen3:8b", "GPU")
    monitor._hw_history.append({"ts": time.time(), "vram_used_mb": 9000, "vram_total_mb": 16384, "vram_pct": 55.0})
    summary = monitor.note_stop("qwen3:8b")
    assert summary["peak_vram_mb"] == 9000
    assert calibration.stats()["samples"] == 0


def test_model_card_aggregates_recent_requests():
    monitor.note_serve("qwen3:8b", None)
    monitor.record_request(
        profile="p", model="qwen3:8b", backend="ollama", kind="chat", success=True,
        elapsed_seconds=1.0, metrics={"tokens_per_second": 20.0, "ttft_ms": 100.0}, context_limit=4096,
    )
    monitor.record_request(
        profile="p", model="qwen3:8b", backend="ollama", kind="chat", success=False,
        elapsed_seconds=0.5, metrics={}, context_limit=4096, error="boom",
    )
    card = monitor._model_card({"name": "qwen3:8b", "placement": "GPU"})
    assert card["request_count"] == 2
    assert card["failure_count"] == 1
    assert card["median_tokens_per_second"] == 20.0
    assert card["uptime_seconds"] is not None


def test_model_card_active_requests_counts_only_in_flight_calls():
    monitor.note_serve("qwen3:8b", None)
    first = monitor.note_request_start("qwen3:8b")
    second = monitor.note_request_start("qwen3:8b")
    monitor.record_request(
        profile="p", model="qwen3:8b", backend="ollama", kind="chat", success=True,
        elapsed_seconds=0.2, metrics={"tokens_per_second": 20.0}, context_limit=4096,
    )
    card = monitor._model_card({"name": "qwen3:8b", "placement": "GPU"})
    assert card["active_requests"] == 2
    monitor.note_request_end("qwen3:8b", first)
    card = monitor._model_card({"name": "qwen3:8b", "placement": "GPU"})
    assert card["active_requests"] == 1
    monitor.note_request_end("qwen3:8b", first)  # duplicate finish is idempotent
    assert monitor._model_card({"name": "qwen3:8b"})["active_requests"] == 1
    monitor.note_request_end("qwen3:8b", second)
    assert monitor._model_card({"name": "qwen3:8b"})["active_requests"] == 0
    assert card["request_count"] == 1


def test_model_card_active_requests_feeds_the_concurrency_alert():
    card = {
        "name": "qwen3:8b", "requested_device": None, "placement": "GPU",
        "median_tokens_per_second": None, "recent_tokens_per_second": None, "active_requests": 2,
    }
    alerts = monitor._alerts({}, [card])
    assert any("simultaneous requests" in a["text"] for a in alerts)


def test_find_serve_state_exact_match():
    monitor.note_serve("gemma3:4b", "GPU")
    state = monitor._find_serve_state("gemma3:4b")
    assert state is not None
    assert state["requested_device"] == "GPU"


def test_find_serve_state_falls_back_to_fuzzy_match():
    # note_serve was called with a bare/tagless name; Ollama reports the
    # fully-qualified ":latest" form back via /api/ps - the exact dict key
    # 'llama3' won't match 'llama3:latest' but the two clearly refer to the
    # same deployment.
    monitor.note_serve("llama3", "CPU")
    state = monitor._find_serve_state("llama3:latest")
    assert state is not None
    assert state["requested_device"] == "CPU"


def test_find_serve_state_no_match_returns_none():
    monitor.note_serve("gemma3:4b", "GPU")
    assert monitor._find_serve_state("qwen3:8b") is None


def test_model_card_uses_fuzzy_serve_state_for_uptime():
    monitor.note_serve("llama3", None)
    card = monitor._model_card({"name": "llama3:latest", "placement": "GPU"})
    assert card["uptime_seconds"] is not None


def test_model_card_matches_request_history_across_latest_alias():
    monitor.note_serve("llama3", None)
    monitor.record_request(
        profile="p", model="llama3", backend="ollama", kind="chat", success=True,
        elapsed_seconds=0.2, metrics={"tokens_per_second": 20.0}, context_limit=4096,
    )
    card = monitor._model_card({"name": "llama3:latest", "placement": "GPU"})
    assert card["request_count"] == 1
    assert card["median_tokens_per_second"] == 20.0


def test_alerts_placement_mismatch():
    card = {
        "name": "qwen3:8b", "requested_device": "GPU", "placement": "CPU",
        "median_tokens_per_second": None, "recent_tokens_per_second": None, "active_requests": 0,
    }
    alerts = monitor._alerts({}, [card])
    assert any("CPU" in a["text"] and "GPU" in a["text"] for a in alerts)


def test_alerts_slow_generation():
    card = {
        "name": "qwen3:8b", "requested_device": None, "placement": "GPU",
        "median_tokens_per_second": 40.0, "recent_tokens_per_second": 20.0, "active_requests": 0,
    }
    alerts = monitor._alerts({}, [card])
    assert any("generation speed" in a["text"].lower() for a in alerts)


def test_alerts_no_false_positive_for_healthy_model():
    card = {
        "name": "qwen3:8b", "requested_device": "GPU", "placement": "GPU",
        "median_tokens_per_second": 40.0, "recent_tokens_per_second": 39.0, "active_requests": 0,
    }
    assert monitor._alerts({}, [card]) == []


def test_alerts_sustained_high_vram_requires_full_window():
    now = time.time()
    # Only two samples spanning a few seconds - not enough coverage of the 3-minute window.
    monitor._hw_history.append({"ts": now - 2, "vram_pct": 99.0})
    monitor._hw_history.append({"ts": now, "vram_pct": 99.0})
    assert monitor._alerts({}, []) == []


def test_alerts_sustained_high_vram_fires_with_full_window():
    now = time.time()
    for i in range(40):
        monitor._hw_history.append({"ts": now - 170 + i * 4.5, "vram_pct": 96.0})
    alerts = monitor._alerts({}, [])
    assert any("VRAM usage has remained above" in a["text"] for a in alerts)


def test_system_monitor_endpoint_smoke(monkeypatch):
    monkeypatch.setattr(hardware, "detect_hardware", lambda: _hw())
    monkeypatch.setattr(_ollama, "list_running", lambda: ([
        {"name": "gemma3:4b", "size": 4_000_000_000, "size_vram": 4_000_000_000, "expires_at": "2030-01-01T00:00:00Z"}
    ], None))
    body = client.get("/system/monitor").json()
    assert body["success"] is True
    assert body["ollama_reachable"] is True
    assert body["hardware"]["vram_pct"] == 50.0
    assert body["hardware"]["gpu_utilization_pct"] == 50.0
    assert len(body["models"]) == 1
    assert body["models"][0]["name"] == "gemma3:4b"
    assert body["models"][0]["placement"] == "GPU"
    assert isinstance(body["alerts"], list)
    assert isinstance(body["requests"], list)
    assert "history" in body


def test_system_monitor_ollama_unreachable_is_graceful(monkeypatch):
    monkeypatch.setattr(hardware, "detect_hardware", lambda: _hw())
    monkeypatch.setattr(_ollama, "list_running", lambda: ([], "Ollama is not reachable."))
    body = client.get("/system/monitor").json()
    assert body["success"] is True
    assert body["ollama_reachable"] is False
    assert body["models"] == []


def test_sample_hardware_counts_only_the_largest_compatible_gpu_pool():
    """A laptop's integrated GPU must not inflate the Monitor's VRAM total."""
    gpus = [
        {"name": "RTX", "vendor": "NVIDIA", "backend": "CUDA", "vram_used_mb": 2500,
         "vram_free_mb": 5692, "vram_total_mb": 8192, "utilization_pct": 40.0},
        {"name": "Radeon iGPU", "vendor": "AMD", "backend": "Vulkan", "vram_used_mb": 0,
         "vram_free_mb": None, "vram_total_mb": 512, "utilization_pct": 0.0},
    ]
    hw = {
        "gpus": gpus,
        "gpu_summary": hardware.summarize_gpus(gpus),
        "system": {"ram_available_mb": 16384, "ram_total_mb": 32768, "cpu_percent": 10.0},
    }
    sample = monitor._sample_hardware(hw)
    assert sample["vram_total_mb"] == 8192
    assert sample["vram_used_mb"] == 2500
    assert sample["gpu_utilization_pct"] == 40.0


def test_sample_hardware_without_a_gpu_summary_still_sums_the_gpus():
    sample = monitor._sample_hardware(_hw(vram_used_mb=1024, vram_total_mb=4096))
    assert sample["vram_total_mb"] == 4096
    assert sample["vram_used_mb"] == 1024
