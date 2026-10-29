"""Decision models (POST /v1/systemone). No model is downloaded or loaded: a stub adapter stands in."""
from __future__ import annotations

import contextlib
import copy
import json
import types
from pathlib import Path

import pytest

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover
    pytest.skip("FastAPI TestClient requires httpx", allow_module_level=True)

import api_server
from api_server import app
from localdeploy.backends import decision

client = TestClient(app)

ROUTE = {
    "type": "choice",
    "instructions": "Which route should handle this message?",
    "criteria": {"conversation": "plain chat", "filesystem.manage": "look at or change files"},
}


class _StubAdapter(decision._Adapter):
    loads = 0

    def load(self) -> None:
        type(self).loads += 1
        self.loaded = True

    def decide(self, state, questions):
        return {
            "answers": {name: {"choice": "filesystem.manage", "confidence": 0.93} for name in questions},
            "usage": {"input_tokens": 12, "output_tokens": 0},
        }


def _profile(**overrides):
    base = {
        "name": "stub", "backend": "decision", "adapter": "stub", "model_id": "stub/model", "enabled": True,
        "max_prompt_chars": 5000,
    }
    base.update(overrides)
    return base


@pytest.fixture(autouse=True)
def _clean_runtime(monkeypatch):
    decision.reset()
    _StubAdapter.loads = 0
    monkeypatch.setitem(decision._ADAPTER_CLASSES, "stub", _StubAdapter)
    yield
    decision.reset()


def _use_config(monkeypatch, profiles, **extra):
    config = {"version": 1, "default_profile": "gemma3_4b_ollama_safe", "profiles": profiles, **extra}
    monkeypatch.setattr(api_server, "load_config", lambda: config)
    return config


def _post(body):
    return client.post("/v1/systemone", json=body)


# --- request validation ------------------------------------------------------


def test_a_well_formed_request_passes_validation() -> None:
    decision.validate_request("hello", {"route": ROUTE, "urgent": {"type": "noul", "instructions": "Is it urgent?"}})
    decision.validate_request({"text": "hi"}, {"level": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "high"]}})


@pytest.mark.parametrize(
    "questions, fragment",
    [
        ({}, "non-empty"),
        ({"bad name": ROUTE}, "invalid question name"),
        ({"q": {"type": "essay", "instructions": "x"}}, "type must be one of"),
        ({"q": {"type": "choice", "instructions": "x", "criteria": {"only": "one"}}}, "2-100 options"),
        ({"q": {"type": "choice", "instructions": "", "criteria": {"a": "x", "b": "y"}}}, "instructions"),
        ({"q": {"type": "score", "instructions": "x", "criteria": ["just one"]}}, "2-10 levels"),
        ({"q": {"type": "noul", "instructions": "x", "criteria": {"maybe": "x"}}}, "only define 'true' and 'false'"),
        ({f"q{i}": ROUTE for i in range(33)}, "too many questions"),
    ],
)
def test_bad_questions_are_refused_before_any_model_loads(questions, fragment) -> None:
    with pytest.raises(decision.DecisionRequestError, match=fragment):
        decision.validate_request("hello", questions)
    assert _StubAdapter.loads == 0


def test_an_oversized_state_is_refused() -> None:
    with pytest.raises(decision.DecisionRequestError, match="too large"):
        decision.validate_request("x" * 101, {"route": ROUTE}, max_state_chars=100)


# --- endpoint ---------------------------------------------------------------


def test_systemone_returns_answers_usage_and_timing(monkeypatch) -> None:
    _use_config(monkeypatch, {"stub": _profile()})

    first = _post({"state": "list my downloads", "questions": {"route": ROUTE}, "model": "stub"})
    second = _post({"state": "hello", "questions": {"route": ROUTE}, "model": "stub"})

    assert first.status_code == 200
    body = first.json()
    assert body["answers"]["route"]["choice"] == "filesystem.manage"
    assert body["model"] == "stub" and body["usage"]["output_tokens"] == 0
    assert "latency_ms" in body and "load_ms" in body  # the first call paid for loading
    assert "load_ms" not in second.json()
    assert _StubAdapter.loads == 1


def test_without_a_model_the_first_enabled_decision_profile_is_used(monkeypatch) -> None:
    _use_config(
        monkeypatch,
        {"chat": {"backend": "ollama", "model_id": "x", "enabled": True}, "stub": _profile()},
    )
    assert _post({"state": "hi", "questions": {"route": ROUTE}}).json()["model"] == "stub"


def test_default_decision_profile_setting_wins(monkeypatch) -> None:
    _use_config(
        monkeypatch,
        {"a": _profile(name="a"), "b": _profile(name="b")},
        default_decision_profile="b",
    )
    assert _post({"state": "hi", "questions": {"route": ROUTE}}).json()["model"] == "b"


def test_no_decision_profile_is_a_clear_404(monkeypatch) -> None:
    _use_config(monkeypatch, {"chat": {"backend": "ollama", "model_id": "x", "enabled": True}})
    response = _post({"state": "hi", "questions": {"route": ROUTE}})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "no_decision_profile"


def test_a_chat_profile_cannot_answer_systemone(monkeypatch) -> None:
    _use_config(monkeypatch, {"chat": {"backend": "ollama", "model_id": "x", "enabled": True}})
    response = _post({"state": "hi", "questions": {"route": ROUTE}, "model": "chat"})
    assert response.status_code == 400
    assert "not a decision model" in response.json()["error"]["message"]


def test_a_disabled_decision_profile_is_not_served(monkeypatch) -> None:
    _use_config(monkeypatch, {"stub": _profile(enabled=False)})
    response = _post({"state": "hi", "questions": {"route": ROUTE}, "model": "stub"})
    assert response.status_code == 404
    assert "disabled" in response.json()["error"]["message"]


def test_a_bad_question_is_a_400_with_the_reason(monkeypatch) -> None:
    _use_config(monkeypatch, {"stub": _profile()})
    response = _post({"state": "hi", "questions": {"q": {"type": "essay", "instructions": "x"}}})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"
    assert "type must be one of" in response.json()["error"]["message"]


def test_a_chat_request_to_a_decision_profile_explains_what_to_do(monkeypatch) -> None:
    _use_config(monkeypatch, {"stub": _profile()})
    response = client.post("/v1/chat/completions", json={"model": "stub", "messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 400
    assert "/v1/systemone" in response.json()["error"]["message"]


def test_models_and_health_report_decision_profiles(monkeypatch) -> None:
    _use_config(monkeypatch, {"stub": _profile()})
    listed = client.get("/v1/models").json()["data"]
    assert [m["localdeploy"]["kind"] for m in listed] == ["decision"]

    assert client.get("/health").json()["decision"]["loaded"] == []
    _post({"state": "hi", "questions": {"route": ROUTE}})
    assert client.get("/health").json()["decision"]["loaded"][0]["profile"] == "stub"


def test_the_all_profiles_benchmark_skips_decision_models(monkeypatch) -> None:
    _use_config(monkeypatch, {"stub": _profile(), "chat": {"backend": "ollama", "model_id": "x", "enabled": True}})
    seen = []
    monkeypatch.setattr(
        api_server, "run_local_request",
        lambda kind, call_data: seen.append(call_data.get("profile")) or {"success": True, "elapsed_seconds": 0.1, "response": "ok"},
    )
    api_server.run_benchmark({"all_profiles": True, "prompt": "hi"})
    assert set(seen) == {"chat"}


def test_gpu_only_mode_refuses_a_cpu_decision_profile(monkeypatch) -> None:
    _use_config(monkeypatch, {"stub": _profile(device="cpu")})
    monkeypatch.setattr(api_server, "require_gpu_only", lambda: True)
    response = _post({"state": "hi", "questions": {"route": ROUTE}, "model": "stub"})
    assert response.status_code == 404
    assert "GPU-only" in response.json()["error"]["message"]


# --- gating: licence, remote code, packages, offline ------------------------


def test_a_model_licence_must_be_accepted_first(monkeypatch) -> None:
    gated = _profile(license_acceptance_required="Example Licence 1.0", license_url="https://example.test/L", license_accepted=False)
    _use_config(monkeypatch, {"stub": gated})

    refused = _post({"state": "hi", "questions": {"route": ROUTE}})
    assert refused.status_code == 503
    message = refused.json()["error"]["message"]
    assert "Example Licence 1.0" in message and "license_accepted" in message and "https://example.test/L" in message
    assert _StubAdapter.loads == 0

    gated["license_accepted"] = True
    assert _post({"state": "hi", "questions": {"route": ROUTE}}).status_code == 200


def test_remote_code_needs_explicit_trust_and_a_pinned_revision() -> None:
    profile = {"backend": "decision", "adapter": "d1", "model_id": "Org/model", "enabled": True}

    with pytest.raises(decision.DecisionUnavailable, match="trust_remote_code"):
        decision.D1Adapter("p", dict(profile)).load()
    with pytest.raises(decision.DecisionUnavailable, match="pinned"):
        decision.D1Adapter("p", {**profile, "trust_remote_code": True}).load()


def test_missing_packages_give_the_install_command(monkeypatch) -> None:
    def _missing(module):
        raise decision.DecisionUnavailable(f"{decision.INSTALL_HINT} (missing: {module})")

    monkeypatch.setattr(decision, "_require", _missing)
    with pytest.raises(decision.DecisionUnavailable, match="requirements-decision.txt"):
        decision.D1Adapter("p", {"model_id": "Org/model", "trust_remote_code": True, "revision": "abc"}).load()


class _FakeDtype:
    pass


def _fake_runtime(monkeypatch, from_pretrained):
    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        dtype=_FakeDtype, float32=_FakeDtype(), set_num_threads=lambda n: None,
    )
    transformers = types.SimpleNamespace(AutoModel=types.SimpleNamespace(from_pretrained=from_pretrained))
    monkeypatch.setattr(decision, "_require", lambda module: {"torch": torch, "transformers": transformers}[module])


_TRUSTED_D1 = {"model_id": "Org/model", "trust_remote_code": True, "revision": "a" * 40}


def test_a_package_the_model_code_needs_is_a_503_with_the_install_command(monkeypatch) -> None:
    def _import_error(*args, **kwargs):
        raise ImportError("This modeling file requires the following packages: torchvision")

    _fake_runtime(monkeypatch, _import_error)
    with pytest.raises(decision.DecisionUnavailable, match="requirements-decision.txt") as caught:
        decision.D1Adapter("p", dict(_TRUSTED_D1)).load()
    assert "torchvision" in str(caught.value)


def test_an_unexpected_load_failure_is_a_502_not_a_traceback(monkeypatch) -> None:
    class _CannotLoad(_StubAdapter):
        def load(self):
            raise RuntimeError("weights are corrupt")

    monkeypatch.setitem(decision._ADAPTER_CLASSES, "stub", _CannotLoad)
    _use_config(monkeypatch, {"stub": _profile()})
    response = _post({"state": "hi", "questions": {"route": ROUTE}})
    assert response.status_code == 502
    assert "could not load 'stub'" in response.json()["error"]["message"]
    assert "weights are corrupt" in response.json()["error"]["message"]


class _FlashlessModel:
    """Answers one question at a time, as d1 does on a PyTorch build without flash attention."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def system_one(self, state, questions):
        self.calls.append(list(questions))
        if len(questions) > 1:
            raise RuntimeError("USE_FLASH_ATTENTION was not enabled for build.")
        (name,) = questions
        return {"answers": {name: {"choice": "a", "confidence": 0.9}}, "usage": {"input_tokens": 10, "output_tokens": 0}}


def _loaded_d1(model) -> "decision.D1Adapter":
    adapter = decision.D1Adapter("p", dict(_TRUSTED_D1))
    adapter.torch = types.SimpleNamespace(inference_mode=lambda: contextlib.nullcontext())
    adapter.device, adapter.model, adapter.one_question_at_a_time, adapter.loaded = "cuda", model, False, True
    return adapter


def test_several_questions_fall_back_to_one_at_a_time_without_flash_attention() -> None:
    model = _FlashlessModel()
    adapter = _loaded_d1(model)
    questions = {"route": ROUTE, "team": ROUTE}

    first = adapter.decide("hi", questions)
    again = adapter.decide("hi", questions)

    assert set(first["answers"]) == {"route", "team"} and first["usage"]["input_tokens"] == 20
    assert again == first
    # the failed shared pass is tried once; afterwards each question goes alone straight away
    assert model.calls == [["route", "team"], ["route"], ["team"], ["route"], ["team"]]
    assert adapter.describe()["one_question_at_a_time"] is True


def test_other_runtime_errors_are_not_swallowed_by_the_fallback() -> None:
    class _Broken(_FlashlessModel):
        def system_one(self, state, questions):
            raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="out of memory"):
        _loaded_d1(_Broken()).decide("hi", {"a": ROUTE, "b": ROUTE})


def test_an_unknown_adapter_is_named() -> None:
    with pytest.raises(decision.DecisionUnavailable, match="adapter must be one of"):
        decision.check_profile("p", {"backend": "decision", "adapter": "nope"})


def test_offline_mode_tells_you_to_fetch_the_files_once(monkeypatch) -> None:
    monkeypatch.setattr(decision, "offline_mode", lambda: True)
    assert "OFFLINE=false" in decision._files_hint("p", OSError("not cached"))


# --- the HTTP forwarder -----------------------------------------------------


class _Reply:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code, self._payload, self.text = status, payload, text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def _http_profile(**overrides):
    return _profile(**{"adapter": "systemone-http", "base_url": "http://127.0.0.1:8100", **overrides})


def test_the_forwarder_only_talks_to_this_machine() -> None:
    adapter = decision.HttpAdapter("p", _http_profile(base_url="http://example.com:8100"))
    with pytest.raises(decision.DecisionUnavailable, match="loopback"):
        adapter.load()


def test_the_forwarder_passes_the_request_through(monkeypatch) -> None:
    sent = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        sent.update(url=url, body=json, headers=headers)
        return _Reply(payload={"answers": {"route": {"choice": "conversation"}}, "usage": {"input_tokens": 3}})

    monkeypatch.setattr(decision.requests, "post", fake_post)
    _use_config(monkeypatch, {"remote": _http_profile(model_id="laya", api_key="secret")})

    body = _post({"state": "hi", "questions": {"route": ROUTE}, "model": "remote"}).json()

    assert body["answers"]["route"]["choice"] == "conversation"
    assert sent["url"] == "http://127.0.0.1:8100/v1/systemone"
    assert sent["body"]["model"] == "laya" and sent["headers"]["Authorization"] == "Bearer secret"


@pytest.mark.parametrize(
    "reply, fragment",
    [
        (_Reply(status=500, text="boom"), "HTTP 500"),
        (_Reply(payload=None), "did not return JSON"),
        (_Reply(payload={"nope": 1}), "no 'answers'"),
    ],
)
def test_forwarder_failures_are_a_502(monkeypatch, reply, fragment) -> None:
    monkeypatch.setattr(decision.requests, "post", lambda *a, **k: reply)
    _use_config(monkeypatch, {"remote": _http_profile()})
    response = _post({"state": "hi", "questions": {"route": ROUTE}, "model": "remote"})
    assert response.status_code == 502
    assert fragment in response.json()["error"]["message"]


def test_a_model_crash_is_a_502_not_a_traceback(monkeypatch) -> None:
    class _Broken(_StubAdapter):
        def decide(self, state, questions):
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setitem(decision._ADAPTER_CLASSES, "stub", _Broken)
    _use_config(monkeypatch, {"stub": _profile()})
    response = _post({"state": "hi", "questions": {"route": ROUTE}})
    assert response.status_code == 502
    assert "CUDA out of memory" in response.json()["error"]["message"]


# --- shipped configuration --------------------------------------------------


def _example_profiles():
    path = Path(__file__).resolve().parent.parent / "config.example.json"
    return json.loads(path.read_text(encoding="utf-8"))["profiles"]


def test_shipped_decision_profiles_are_off_pinned_and_licence_gated() -> None:
    decision_profiles = {n: p for n, p in _example_profiles().items() if p.get("backend") == "decision"}
    assert {"d1_omni_600m_decision", "d1_3b_decision", "laya_decision", "laya_typed_decision"} <= set(decision_profiles)
    for name, profile in decision_profiles.items():
        assert profile["enabled"] is False, name
        assert profile["adapter"] in decision.ADAPTER_NAMES, name
        if profile["adapter"] in {"d1", "laya"}:
            assert len(profile["revision"]) == 40, f"{name} must pin a full commit hash"
        if profile["adapter"] == "d1":
            assert profile["trust_remote_code"] is True and profile["license_accepted"] is False, name
            assert profile["license_acceptance_required"], name


def test_example_profiles_pass_the_adapter_check() -> None:
    for name, profile in _example_profiles().items():
        if profile.get("backend") != "decision":
            continue
        accepted = copy.deepcopy(profile)
        accepted["license_accepted"] = True
        decision.check_profile(name, accepted)  # must not raise
