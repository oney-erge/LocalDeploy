"""Tests for the automated model bakeoff (Release R6): candidate selection
and the streamed /system/bakeoff/run orchestration.
"""
from __future__ import annotations

import json

import pytest

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover
    pytest.skip("FastAPI TestClient requires httpx", allow_module_level=True)

import benchmark
from api_server import app
from localdeploy.control import _ollama, bakeoff as bakeoff_mod
from localdeploy.control import models as models_mod
from localdeploy.control import starter as starter_mod

client = TestClient(app)


def _hw(vram_free_mb=24 * 1024):
    return {
        "success": True,
        "gpu_available": True,
        "gpus": [{"name": "Fake GPU", "vram_free_mb": vram_free_mb}],
        "system": {"ram_available_mb": None},
        "message": None,
    }


@pytest.fixture(autouse=True)
def isolate_hardware(monkeypatch):
    monkeypatch.setattr(starter_mod, "detect_hardware", lambda: _hw())


def _sse_events(text):
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]


# ---- candidate selection (pure) -------------------------------------------------

def test_select_candidates_respects_download_budget():
    picks = bakeoff_mod._select_candidates(None, "balanced", budget_gb=24.0, download_budget_gb=6.0, max_candidates=5)
    assert picks
    total = sum(p["download_gb"] for p in picks)
    assert total <= 6.0


def test_select_candidates_respects_even_a_tiny_download_budget():
    picks = bakeoff_mod._select_candidates(None, "balanced", budget_gb=24.0, download_budget_gb=0.01, max_candidates=5)
    assert picks == []


def test_select_candidates_uses_requested_context_for_memory():
    picks = bakeoff_mod._select_candidates(
        None, "lowest_memory", budget_gb=24.0, download_budget_gb=10.0,
        max_candidates=2, expected_context=65536,
    )
    assert picks
    for candidate in picks:
        assert candidate["required_gb"] == starter_mod._required_gb(
            candidate["params_b"], context=65536
        )


def test_select_candidates_workload_bias():
    picks = bakeoff_mod._select_candidates("coding", "balanced", budget_gb=24.0, download_budget_gb=50.0, max_candidates=3)
    assert any("coding" in p.get("workload_tags", []) for p in picks)


def test_select_candidates_caps_at_max_candidates():
    picks = bakeoff_mod._select_candidates(None, "balanced", budget_gb=24.0, download_budget_gb=100.0, max_candidates=2)
    assert len(picks) <= 2


# ---- full streamed run -----------------------------------------------------------

def _fake_execute_test(base_url, profile_name, profile, test, timeout, **kwargs):
    return benchmark.TestResult(
        name=test.name, category=test.category, success=True, elapsed_seconds=0.1,
        response_length=4, response_preview="ok", accuracy=1.0, approx_tokens_per_second=30.0,
    )


def _fake_ensure_profile_for_model(model_id):
    # Real profile creation is refused against the read-only example config
    # this test suite uses (see test_profiles_crud.py); the orchestration
    # here only needs *a* stable name, not a persisted profile.
    return f"profile_{model_id.replace(':', '_').replace('.', '_').replace('-', '_')}", True, None


def test_bakeoff_no_budget_errors(monkeypatch):
    monkeypatch.setattr(starter_mod, "detect_hardware", lambda: {"success": True, "gpu_available": False, "gpus": [], "system": {"ram_available_mb": None}, "message": None})
    resp = client.post("/system/bakeoff/run", json={})
    events = _sse_events(resp.text)
    assert events[0]["event"] == "error"


def test_bakeoff_happy_path_picks_a_winner(monkeypatch):
    monkeypatch.setattr(_ollama, "list_installed", lambda: ([], None))
    pulled = []

    def fake_pull_stream(model_id):
        pulled.append(model_id)
        yield {"status": "success", "done": True}

    monkeypatch.setattr(_ollama, "pull_stream", fake_pull_stream)
    monkeypatch.setattr(_ollama, "unload_model", lambda model: {})

    def fake_serve(model_id, keep_alive, num_gpu=None):
        return {"success": True, "running": [{"name": model_id, "size": 1, "size_vram": 1, "placement": "GPU"}]}

    monkeypatch.setattr(models_mod, "_serve_ollama", fake_serve)
    monkeypatch.setattr(bakeoff_mod, "_serve_ollama", fake_serve)
    monkeypatch.setattr(bakeoff_mod, "ensure_profile_for_model", _fake_ensure_profile_for_model)
    monkeypatch.setattr(benchmark, "execute_test", _fake_execute_test)

    resp = client.post("/system/bakeoff/run", json={"max_candidates": 2, "sample_size": 1, "download_budget_gb": 200})
    assert resp.status_code == 200
    events = _sse_events(resp.text)
    kinds = [e["event"] for e in events]
    assert "bakeoff_start" in kinds
    assert kinds.count("candidate_start") >= 1
    assert "bakeoff_end" in kinds
    end = next(e for e in events if e["event"] == "bakeoff_end")
    assert end["winner"] == end["ranked"][0]["profile"]
    assert end["winner"] not in end["losers"]
    assert end["winner_deployed"] is True


def test_bakeoff_pull_failure_marks_candidate_failed_but_continues(monkeypatch):
    monkeypatch.setattr(_ollama, "list_installed", lambda: ([], None))

    call_count = {"n": 0}

    def flaky_pull_stream(model_id):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("network error")
        yield {"status": "success", "done": True}

    monkeypatch.setattr(_ollama, "pull_stream", flaky_pull_stream)
    monkeypatch.setattr(_ollama, "unload_model", lambda model: {})

    def fake_serve(model_id, keep_alive, num_gpu=None):
        return {"success": True, "running": [{"name": model_id, "size": 1, "size_vram": 1, "placement": "GPU"}]}

    monkeypatch.setattr(models_mod, "_serve_ollama", fake_serve)
    monkeypatch.setattr(bakeoff_mod, "_serve_ollama", fake_serve)
    monkeypatch.setattr(bakeoff_mod, "ensure_profile_for_model", _fake_ensure_profile_for_model)
    monkeypatch.setattr(benchmark, "execute_test", _fake_execute_test)

    resp = client.post("/system/bakeoff/run", json={"max_candidates": 2, "sample_size": 1, "download_budget_gb": 200})
    events = _sse_events(resp.text)
    kinds = [e["event"] for e in events]
    assert "candidate_failed" in kinds
    # A second candidate still ran and the bakeoff still concluded.
    assert "bakeoff_end" in kinds


def test_bakeoff_all_candidates_fail_reports_error(monkeypatch):
    monkeypatch.setattr(_ollama, "list_installed", lambda: ([], None))

    def always_fails(model_id):
        raise RuntimeError("no network")
        yield  # pragma: no cover - unreachable, keeps this a generator

    monkeypatch.setattr(_ollama, "pull_stream", always_fails)
    resp = client.post("/system/bakeoff/run", json={"max_candidates": 1, "sample_size": 1, "download_budget_gb": 200})
    events = _sse_events(resp.text)
    assert events[-1]["event"] == "error"
