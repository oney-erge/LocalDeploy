"""/health does not wait for a llama.cpp that is not running on every call."""
from __future__ import annotations

import pytest
import requests

from localdeploy.backends import llamacpp


@pytest.fixture(autouse=True)
def _fresh_cache():
    llamacpp._probe_cache.clear()
    yield
    llamacpp._probe_cache.clear()


def test_a_refused_connection_costs_half_a_second_at_most(monkeypatch):
    seen = []

    def refused(url, timeout=None, **kwargs):
        seen.append(timeout)
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(llamacpp.requests, "get", refused)
    result = llamacpp.llama_health("http://127.0.0.1:8080")
    assert result["reachable"] is False and "not reachable" in result["error"]
    assert seen == [(0.5, 2)]


def test_the_answer_is_reused_for_a_few_seconds_then_asked_again(monkeypatch):
    calls = []

    class Ok:
        ok = True

    monkeypatch.setattr(llamacpp.requests, "get", lambda url, timeout=None, **kw: calls.append(url) or Ok())
    clock = {"now": 500.0}
    monkeypatch.setattr(llamacpp.time, "monotonic", lambda: clock["now"])
    first = llamacpp.llama_health_cached("http://127.0.0.1:8080")
    clock["now"] += llamacpp.LLAMA_PROBE_TTL_SECONDS - 1
    assert llamacpp.llama_health_cached("http://127.0.0.1:8080") == first and len(calls) == 1
    clock["now"] += 2
    llamacpp.llama_health_cached("http://127.0.0.1:8080")
    assert len(calls) == 2
