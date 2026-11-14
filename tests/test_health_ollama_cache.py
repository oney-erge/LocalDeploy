"""/health does not wait for an Ollama that is not running on every call."""
from __future__ import annotations

import pytest
import requests

from localdeploy.backends import ollama


@pytest.fixture(autouse=True)
def _fresh_cache():
    ollama._probe_cache.clear()
    yield
    ollama._probe_cache.clear()


def _counting_get(monkeypatch, behaviour):
    calls = []

    def fake_get(url, timeout=None, **kwargs):
        calls.append((url, timeout))
        return behaviour()

    monkeypatch.setattr(ollama.requests, "get", fake_get)
    return calls


def test_a_refused_connection_costs_half_a_second_at_most(monkeypatch):
    def refused():
        raise requests.ConnectionError("refused")

    calls = _counting_get(monkeypatch, refused)
    models, error = ollama.ollama_models("http://127.0.0.1:11434")
    assert models == [] and "not reachable" in error
    assert calls[0][1] == (0.5, 3)


def test_the_answer_is_reused_for_a_few_seconds_then_asked_again(monkeypatch):
    class Reply:
        def raise_for_status(self):
            pass

        def json(self):
            return {"models": [{"name": "qwen3:8b"}]}

    calls = _counting_get(monkeypatch, Reply)
    clock = {"now": 1000.0}
    monkeypatch.setattr(ollama.time, "monotonic", lambda: clock["now"])
    first = ollama.ollama_models_cached("http://127.0.0.1:11434")
    clock["now"] += ollama.OLLAMA_PROBE_TTL_SECONDS - 1
    second = ollama.ollama_models_cached("http://127.0.0.1:11434")
    assert first == second == (["qwen3:8b"], None) and len(calls) == 1
    clock["now"] += 2
    ollama.ollama_models_cached("http://127.0.0.1:11434")
    assert len(calls) == 2


def test_each_address_has_its_own_answer(monkeypatch):
    def refused():
        raise requests.ConnectionError("refused")

    calls = _counting_get(monkeypatch, refused)
    ollama.ollama_models_cached("http://127.0.0.1:11434")
    ollama.ollama_models_cached("http://127.0.0.1:11435")
    ollama.ollama_models_cached("http://127.0.0.1:11434")
    assert len(calls) == 2


def test_the_health_route_uses_the_remembered_answer(monkeypatch):
    from fastapi.testclient import TestClient

    import api_server

    def refused():
        raise requests.ConnectionError("refused")

    calls = _counting_get(monkeypatch, refused)
    monkeypatch.setattr(api_server, "require_gpu_only", lambda: False, raising=False)
    client = TestClient(api_server.app)
    for _ in range(3):
        body = client.get("/health").json()
        assert body["ollama"]["reachable"] is False
    assert len(calls) <= 1
