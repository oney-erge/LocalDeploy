"""Tests for the guided /registry/recommend endpoint (Release R1).

Mirrors test_starter_pack.py's approach: monkeypatch detect_hardware to
simulate representative hardware, since CI has no GPU.
"""
from __future__ import annotations

import pytest

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover
    pytest.skip("FastAPI TestClient requires httpx", allow_module_level=True)

from api_server import app
from localdeploy.control import starter

client = TestClient(app)


def _hw(vram_free_mb=None, ram_available_mb=None, gpu_available=None):
    gpus = [{"name": "Fake GPU", "vram_free_mb": vram_free_mb}] if vram_free_mb is not None else []
    return {
        "success": True,
        "gpu_available": gpu_available if gpu_available is not None else bool(gpus),
        "gpus": gpus,
        "system": {"ram_available_mb": ram_available_mb},
        "message": None,
    }


@pytest.fixture(autouse=True)
def no_measured_history(monkeypatch):
    # Isolate from whatever real benchmark history sits on the dev machine -
    # the "measured on this machine" signal gets its own dedicated test below.
    monkeypatch.setattr(starter, "_measured_stats", lambda model_id: None)


def test_recommend_returns_three_labeled_buckets(monkeypatch):
    monkeypatch.setattr(starter, "detect_hardware", lambda: _hw(vram_free_mb=24 * 1024))
    body = client.post("/registry/recommend", json={}).json()
    assert body["success"] is True
    assert body["recommended"] is not None
    assert body["recommended"]["bucket"] == "recommended"
    assert body["faster"] is not None
    assert body["faster"]["bucket"] == "faster"
    assert body["higher_quality"] is not None
    assert body["higher_quality"]["bucket"] == "higher_quality"
    # Buckets should be distinct picks when the catalog has enough fitting variety.
    ids = {body["recommended"]["id"], body["faster"]["id"], body["higher_quality"]["id"]}
    assert len(ids) >= 2


def test_recommend_faster_is_not_larger_than_recommended(monkeypatch):
    monkeypatch.setattr(starter, "detect_hardware", lambda: _hw(vram_free_mb=24 * 1024))
    body = client.post("/registry/recommend", json={"priority": "best_quality"}).json()
    assert body["faster"]["params_b"] <= body["recommended"]["params_b"]


def test_recommend_workload_bias_prefers_matching_tag(monkeypatch):
    monkeypatch.setattr(starter, "detect_hardware", lambda: _hw(vram_free_mb=24 * 1024))
    body = client.post("/registry/recommend", json={"use_case": "coding"}).json()
    assert "coding" in body["recommended"]["workload_tags"]


def test_recommend_unknown_use_case_is_ignored_not_rejected(monkeypatch):
    monkeypatch.setattr(starter, "detect_hardware", lambda: _hw(vram_free_mb=24 * 1024))
    body = client.post("/registry/recommend", json={"use_case": "not-a-real-tag"}).json()
    assert body["success"] is True
    assert body["use_case"] is None


def test_recommend_lowest_memory_priority_minimizes_required_gb(monkeypatch):
    monkeypatch.setattr(starter, "detect_hardware", lambda: _hw(vram_free_mb=24 * 1024))
    balanced = client.post("/registry/recommend", json={"priority": "balanced"}).json()
    lowest = client.post("/registry/recommend", json={"priority": "lowest_memory"}).json()
    assert lowest["recommended"]["required_gb"] <= balanced["recommended"]["required_gb"]


def test_recommend_flags_context_beyond_published_window(monkeypatch):
    monkeypatch.setattr(starter, "detect_hardware", lambda: _hw(vram_free_mb=24 * 1024))
    body = client.post("/registry/recommend", json={"expected_context": 200_000}).json()
    reasons_text = " ".join(r["text"] for r in body["recommended"]["reasons"])
    assert "exceeds" in reasons_text


def test_recommend_uses_requested_context_for_memory_estimate(monkeypatch):
    monkeypatch.setattr(starter, "detect_hardware", lambda: _hw(vram_free_mb=24 * 1024))
    short = client.post("/registry/recommend", json={"priority": "lowest_memory", "expected_context": 4096}).json()
    long = client.post("/registry/recommend", json={"priority": "lowest_memory", "expected_context": 65536}).json()
    assert short["recommended"]["id"] == long["recommended"]["id"]
    assert long["recommended"]["required_gb"] > short["recommended"]["required_gb"]
    assert long["recommended"]["required_gb"] == starter._required_gb(
        long["recommended"]["params_b"], context=65536
    )


def test_recommend_reasons_are_labeled_by_provenance(monkeypatch):
    monkeypatch.setattr(starter, "detect_hardware", lambda: _hw(vram_free_mb=24 * 1024))
    body = client.post("/registry/recommend", json={}).json()
    kinds = {r["kind"] for r in body["recommended"]["reasons"]}
    assert kinds <= {"estimated", "published", "measured"}
    assert "estimated" in kinds  # the fit-budget reason is always present


def test_recommend_no_hardware_info_is_graceful(monkeypatch):
    monkeypatch.setattr(starter, "detect_hardware", lambda: _hw())
    body = client.post("/registry/recommend", json={}).json()
    assert body["success"] is True
    assert body["recommended"] is None
    assert body["message"]


def test_recommend_tiny_budget_relaxes_margin(monkeypatch):
    smallest = min(starter.STARTER_CATALOG, key=lambda e: e["params_b"])
    required = starter._required_gb(smallest["params_b"])
    monkeypatch.setattr(starter, "detect_hardware", lambda: _hw(vram_free_mb=int(required * 1024) + 1))
    body = client.post("/registry/recommend", json={"margin_gb": 2.0, "expected_context": 4096}).json()
    assert body["success"] is True
    assert body["recommended"] is not None
    assert body["margin_relaxed"] is True
    assert body["message"]


def test_recommend_measured_stats_boost_confidence(monkeypatch):
    monkeypatch.setattr(starter, "detect_hardware", lambda: _hw(vram_free_mb=24 * 1024))
    monkeypatch.setattr(
        starter, "_measured_stats", lambda model_id: {"tokens_per_second": 42.0, "sample_count": 3}
    )
    body = client.post("/registry/recommend", json={}).json()
    assert body["recommended"]["confidence"] == "high"
    assert body["recommended"]["measured_tokens_per_second"] == 42.0
    reasons_text = " ".join(r["text"] for r in body["recommended"]["reasons"])
    assert "Measured on this machine" in reasons_text
