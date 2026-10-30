"""Decision models (POST /v1/systemone). No model is downloaded or loaded: a stub adapter stands in."""
from __future__ import annotations

import contextlib
import copy
import json
import sys
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


# --- the llama.cpp server LocalDeploy starts ---------------------------------


class _FakeProcess:
    def __init__(self, exit_code=None):
        self.exit_code, self.terminated, self.returncode = exit_code, False, exit_code

    def poll(self):
        return self.exit_code

    def terminate(self):
        self.terminated, self.exit_code, self.returncode = True, 0, 0

    def wait(self, timeout=None):
        return self.exit_code

    def kill(self):
        self.terminate()


def _llama_profile(tmp_path, **overrides):
    model = tmp_path / "d3-lite.gguf"
    model.write_bytes(b"gguf")
    server = tmp_path / "llama-server.exe"
    server.write_bytes(b"exe")
    return _profile(**{"adapter": "llamacpp", "model_id": str(model), "server_path": str(server), "port": 18099, **overrides})


@pytest.fixture
def _llama(monkeypatch, tmp_path):
    """Replace the process and the health check; hand back what was started."""
    started = {"commands": [], "process": _FakeProcess()}
    monkeypatch.setenv("LOCALDEPLOY_HOME", str(tmp_path))
    monkeypatch.setattr(decision, "app_home", lambda: tmp_path)

    def popen(command, **kwargs):
        started["commands"].append(command)
        started["process"] = _FakeProcess()
        return started["process"]

    monkeypatch.setattr(decision.subprocess, "Popen", popen)
    monkeypatch.setattr(decision.requests, "get", lambda url, timeout=None: _Reply(status=200, payload={"status": "ok"}))
    monkeypatch.setattr(decision.time, "sleep", lambda s: None)
    return started


def test_llama_server_is_started_once_on_first_use_and_asked_for_the_decision(monkeypatch, tmp_path, _llama) -> None:
    sent = {}
    monkeypatch.setattr(
        decision.requests, "post",
        lambda url, json=None, headers=None, timeout=None: sent.update(url=url, body=json)
        or _Reply(payload={"answers": {"route": {"choice": "filesystem.manage"}}, "usage": {"input_tokens": 9}}),
    )
    profile = _llama_profile(tmp_path, device="cpu", threads=6)
    _use_config(monkeypatch, {"gguf": profile})

    first = _post({"state": "hi", "questions": {"route": ROUTE}, "model": "gguf"})
    _post({"state": "again", "questions": {"route": ROUTE}, "model": "gguf"})

    assert first.status_code == 200 and first.json()["answers"]["route"]["choice"] == "filesystem.manage"
    assert len(_llama["commands"]) == 1  # not once per request
    command = _llama["commands"][0]
    assert command[command.index("-m") + 1] == profile["model_id"]
    assert command[command.index("--host") + 1] == "127.0.0.1" and command[command.index("--port") + 1] == "18099"
    assert command[command.index("-t") + 1] == "6" and command[command.index("-ngl") + 1] == "0"
    assert command[command.index("--device") + 1] == "none"  # a Vulkan or CUDA build would otherwise use the GPU
    assert "-b" in command and "-ub" in command  # the prompt is read in one batch
    assert sent["url"] == "http://127.0.0.1:18099/v1/systemone"
    assert "model" not in sent["body"]  # llama-server hosts exactly one model


def test_llama_server_gpu_and_huggingface_forms(monkeypatch, tmp_path, _llama) -> None:
    adapter = decision.LlamaCppAdapter("p", _llama_profile(tmp_path, device="cuda", model_id="ggml-org/vllm-d3-lite-GGUF:Q8_0"))
    command = adapter._command(adapter._server(), 18099)
    assert command[command.index("-hf") + 1] == "ggml-org/vllm-d3-lite-GGUF:Q8_0" and "-m" not in command
    assert command[command.index("-ngl") + 1] == "99" and "--offline" not in command
    assert "--device" not in command  # the GPU profile keeps llama.cpp's own device choice

    monkeypatch.setattr(decision, "offline_mode", lambda: True)
    assert "--offline" in adapter._command(adapter._server(), 18099)


def test_a_missing_llama_server_says_where_to_get_it(monkeypatch, tmp_path, _llama) -> None:
    monkeypatch.delenv("LLAMACPP_SERVER_PATH", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setattr(decision.shutil, "which", lambda name: None)
    adapter = decision.LlamaCppAdapter("p", _llama_profile(tmp_path, server_path=None))
    with pytest.raises(decision.DecisionUnavailable, match="winget install ggml.llamacpp.*LLAMACPP_SERVER_PATH"):
        adapter.load()
    assert _llama["commands"] == []


def test_llama_server_is_found_in_a_winget_install_like_the_launcher_finds_it(monkeypatch, tmp_path) -> None:
    """WinGet puts it on PATH only for programs started after the install."""
    package = tmp_path / "Microsoft" / "WinGet" / "Packages" / "ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe"
    package.mkdir(parents=True)
    (package / "llama-server.exe").write_bytes(b"exe")
    (package / "llama-quantize.exe").write_bytes(b"exe")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(decision.shutil, "which", lambda name: None)

    server = decision.find_llama_server("")
    assert server == str(package / "llama-server.exe")
    assert decision.find_llama_quantize(server) == str(package / "llama-quantize.exe")
    # A configured path that is wrong is reported, not quietly swapped for another copy.
    assert decision.find_llama_server(str(tmp_path / "elsewhere" / "llama-server.exe")) is None


def test_an_old_llama_server_without_systemone_is_named_with_its_build(monkeypatch, tmp_path, _llama) -> None:
    monkeypatch.setattr(decision.requests, "post", lambda *a, **k: _Reply(status=404, payload={"error": "not found"}))
    monkeypatch.setattr(decision, "llama_server_build", lambda server: 9297)
    _use_config(monkeypatch, {"gguf": _llama_profile(tmp_path)})

    response = _post({"state": "hi", "questions": {"route": ROUTE}, "model": "gguf"})

    assert response.status_code == 503
    message = response.json()["error"]["message"]
    assert "build 9297" in message and "/v1/systemone" in message and "winget upgrade ggml.llamacpp" in message


@pytest.mark.parametrize(
    "output, build",
    [
        ("ggml_cuda_init: found 1 CUDA devices\nversion: 0.8.0-dev (build 11976, commit 2e0d86beb)\nbuilt with Clang\n", 11976),
        ("version: 9297 (b0df4c0cf)\nbuilt with Clang 19.1.5 for Windows x86_64\n", 9297),
        ("nothing useful", None),
    ],
)
def test_the_build_number_is_read_from_llama_server_version(monkeypatch, output, build) -> None:
    """Older releases print the build as the version; newer ones print a semantic version and the build."""
    monkeypatch.setattr(decision.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout="", stderr=output))
    assert decision.llama_server_build("llama-server") == build


class _PsProcess:
    """Stands in for psutil.Process: one recorded llama-server, or another program with a reused id."""

    killed = []

    def __init__(self, pid, name="llama-server.exe", created=1000.0):
        self.pid, self._name, self._created = pid, name, created

    def name(self):
        return self._name

    def create_time(self):
        return self._created

    def terminate(self):
        _PsProcess.killed.append(self.pid)

    def wait(self, timeout=None):
        return 0


@pytest.mark.parametrize(
    "running, stopped",
    [
        (dict(name="llama-server.exe", created=1000.0), True),  # the one the killed LocalDeploy started
        (dict(name="llama-server.exe", created=2000.0), False),  # same id, started later: not ours
        (dict(name="chrome.exe", created=1000.0), False),  # the id now belongs to another program
    ],
)
def test_a_llama_server_left_by_a_killed_localdeploy_is_stopped_before_a_new_one(monkeypatch, tmp_path, _llama, running, stopped) -> None:
    """A forced stop skips the exit handler; its llama-server would otherwise keep the model in memory."""
    _PsProcess.killed = []
    monkeypatch.setitem(sys.modules, "psutil", types.SimpleNamespace(Process=lambda pid: _PsProcess(pid, **running)))
    adapter = decision.LlamaCppAdapter("gguf", _llama_profile(tmp_path))
    pid_file = tmp_path / "logs" / "decision-gguf.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text("4242 1000.0", encoding="utf-8")

    adapter.load()

    assert _PsProcess.killed == ([4242] if stopped else [])
    adapter.stop()
    assert not pid_file.exists()


def test_the_started_llama_server_is_recorded_and_forgotten_on_stop(monkeypatch, tmp_path, _llama) -> None:
    monkeypatch.setitem(sys.modules, "psutil", types.SimpleNamespace(Process=lambda pid: _PsProcess(pid, created=1234.5)))

    def popen(command, **kwargs):
        process = _FakeProcess()
        process.pid = 777
        _llama["process"] = process
        return process

    monkeypatch.setattr(decision.subprocess, "Popen", popen)
    adapter = decision.LlamaCppAdapter("gguf", _llama_profile(tmp_path))
    adapter.load()
    pid_file = tmp_path / "logs" / "decision-gguf.pid"
    assert pid_file.read_text(encoding="utf-8") == "777 1234.5"
    adapter.stop()
    assert not pid_file.exists()


# --- quantize on first use ----------------------------------------------------

REPO_FILES = ["README.md", "mmproj-vllm-d3-lite-BF16.gguf", "vllm-d3-lite-BF16.gguf", "vllm-d3-lite-Q8_0.gguf"]


@pytest.fixture
def _hub(monkeypatch, tmp_path):
    """A fake huggingface_hub and llama-quantize; hands back what was asked of them."""
    seen = {"listed": 0, "downloaded": [], "commands": []}
    source = tmp_path / "hub" / "vllm-d3-lite-BF16.gguf"
    source.parent.mkdir()
    source.write_bytes(b"bf16")

    class _Api:
        def list_repo_files(self, repo):
            seen["listed"] += 1
            return REPO_FILES

    def download(repo, name, local_files_only=False):
        seen["downloaded"].append(name)
        return str(source)

    def run(command, **kwargs):
        seen["commands"].append(command)
        Path(command[2]).write_bytes(b"q4")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=_Api, hf_hub_download=download))
    monkeypatch.setattr(decision.subprocess, "run", run)
    monkeypatch.setattr(decision, "offline_mode", lambda: False)
    seen["source"] = source
    return seen


def test_quantize_makes_the_file_once_from_the_bf16_model_not_the_vision_projector(tmp_path, _hub) -> None:
    out = tmp_path / "models"

    first = decision.quantized_gguf("ggml-org/vllm-d3-lite-GGUF:Q8_0", "Q4_K_M", out, quantizer="llama-quantize")
    again = decision.quantized_gguf("ggml-org/vllm-d3-lite-GGUF", "Q4_K_M", out, quantizer="llama-quantize")

    assert first == again == out / "vllm-d3-lite-Q4_K_M.gguf" and first.read_bytes() == b"q4"
    assert _hub["downloaded"] == ["vllm-d3-lite-BF16.gguf"] and _hub["listed"] == 1  # the second call used the file
    command = _hub["commands"][0]
    assert command[0] == "llama-quantize" and command[1] == str(_hub["source"]) and command[3] == "Q4_K_M"
    assert not list(out.glob("*.partial"))


def test_quantize_reports_a_failure_and_leaves_no_half_written_file(monkeypatch, tmp_path, _hub) -> None:
    monkeypatch.setattr(
        decision.subprocess, "run",
        lambda command, **kwargs: types.SimpleNamespace(returncode=1, stdout="", stderr="main: invalid ftype 'Q9'"),
    )
    with pytest.raises(decision.DecisionUnavailable, match="exit 1.*invalid ftype"):
        decision.quantized_gguf("ggml-org/vllm-d3-lite-GGUF", "Q4_K_M", tmp_path / "models", quantizer="q")
    assert not (tmp_path / "models" / "vllm-d3-lite-Q4_K_M.gguf").exists()


def test_quantize_refuses_odd_types_and_offline_downloads(monkeypatch, tmp_path, _hub) -> None:
    with pytest.raises(decision.DecisionUnavailable, match="llama.cpp type"):
        decision.quantized_gguf("ggml-org/vllm-d3-lite-GGUF", "Q4 --help", tmp_path, quantizer="q")
    monkeypatch.setattr(decision, "offline_mode", lambda: True)
    with pytest.raises(decision.DecisionUnavailable, match="OFFLINE=true"):
        decision.quantized_gguf("ggml-org/vllm-d3-lite-GGUF", "Q4_K_M", tmp_path, quantizer="q")


def test_a_profile_with_quantize_starts_llama_server_on_the_made_file(monkeypatch, tmp_path, _llama, _hub) -> None:
    server = tmp_path / "bin" / "llama-server.exe"
    server.parent.mkdir()
    server.write_bytes(b"exe")
    (server.parent / "llama-quantize.exe").write_bytes(b"exe")
    adapter = decision.LlamaCppAdapter(
        "p", _profile(adapter="llamacpp", model_id="ggml-org/vllm-d3-lite-GGUF", quantize="Q4_K_M", server_path=str(server))
    )

    command = adapter._command(adapter._server(), 18099)

    made = tmp_path / "models" / "decision" / "vllm-d3-lite-Q4_K_M.gguf"
    assert command[command.index("-m") + 1] == str(made) and "-hf" not in command
    assert _hub["commands"][0][0] == str(server.parent / "llama-quantize.exe")


def test_a_missing_model_file_is_named(monkeypatch, tmp_path, _llama) -> None:
    adapter = decision.LlamaCppAdapter("p", _llama_profile(tmp_path, model_id=str(tmp_path / "gone.gguf")))
    with pytest.raises(decision.DecisionUnavailable, match="does not exist"):
        adapter.load()


def test_a_server_that_dies_while_starting_reports_its_log(monkeypatch, tmp_path, _llama) -> None:
    def popen(command, stdout=None, **kwargs):
        stdout.write(b"error: failed to load model\n")
        stdout.flush()
        return _FakeProcess(exit_code=1)

    monkeypatch.setattr(decision.subprocess, "Popen", popen)
    adapter = decision.LlamaCppAdapter("p", _llama_profile(tmp_path))
    with pytest.raises(decision.DecisionUnavailable, match="exited with code 1.*failed to load model"):
        adapter.load()


def test_a_server_that_never_becomes_ready_is_stopped(monkeypatch, tmp_path, _llama) -> None:
    monkeypatch.setattr(decision.requests, "get", lambda url, timeout=None: _Reply(status=503, payload={}))
    clock = iter(range(0, 1000, 100))
    monkeypatch.setattr(decision.time, "monotonic", lambda: next(clock))
    adapter = decision.LlamaCppAdapter("p", _llama_profile(tmp_path, startup_timeout_seconds=150))
    with pytest.raises(decision.DecisionUnavailable, match="did not become ready"):
        adapter.load()
    assert _llama["process"].terminated


def test_a_server_that_died_is_started_again_and_reset_stops_it(monkeypatch, tmp_path, _llama) -> None:
    monkeypatch.setattr(decision.requests, "post", lambda *a, **k: _Reply(payload={"answers": {"q": {}}}))
    _use_config(monkeypatch, {"gguf": _llama_profile(tmp_path)})
    _post({"state": "hi", "questions": {"route": ROUTE}, "model": "gguf"})
    _llama["process"].exit_code = 137  # killed from outside (out of memory, a user closing it)

    assert _post({"state": "hi", "questions": {"route": ROUTE}, "model": "gguf"}).status_code == 200
    assert len(_llama["commands"]) == 2

    decision.reset()
    assert _llama["process"].terminated


# --- shipped configuration --------------------------------------------------


def _example_profiles():
    path = Path(__file__).resolve().parent.parent / "config.example.json"
    return json.loads(path.read_text(encoding="utf-8"))["profiles"]


def test_shipped_decision_profiles_are_off_pinned_and_licence_gated() -> None:
    decision_profiles = {n: p for n, p in _example_profiles().items() if p.get("backend") == "decision"}
    assert {
        "d1_omni_600m_decision", "d1_3b_decision", "laya_decision", "laya_typed_decision",
        "d3_lite_decision", "d3_nano_decision", "d3_mini_decision", "d1_3b_gguf_decision",
    } <= set(decision_profiles)
    for name, profile in decision_profiles.items():
        assert profile["enabled"] is False, name
        assert profile["adapter"] in decision.ADAPTER_NAMES, name
        if profile["adapter"] in {"d1", "laya"}:
            assert len(profile["revision"]) == 40, f"{name} must pin a full commit hash"
        if profile["adapter"] == "d1":
            assert profile["trust_remote_code"] is True and profile["license_accepted"] is False, name
            assert profile["license_acceptance_required"], name
        if profile["adapter"] == "llamacpp":
            assert profile["model_id"], name
            assert "trust_remote_code" not in profile, f"{name}: a GGUF runs no model code"
        if profile.get("license_acceptance_required"):
            assert profile["license_accepted"] is False, f"{name} must not ship pre-accepted"


def test_example_profiles_pass_the_adapter_check() -> None:
    for name, profile in _example_profiles().items():
        if profile.get("backend") != "decision":
            continue
        accepted = copy.deepcopy(profile)
        accepted["license_accepted"] = True
        decision.check_profile(name, accepted)  # must not raise
