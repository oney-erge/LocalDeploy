"""LocalDeploy starts, switches and stops chat models served by llama-server (``"managed": true`` profiles)."""

from __future__ import annotations

import pytest

from localdeploy.backends import llamacpp, llamacpp_managed
from localdeploy.utils import BackendCallError


class _Process:
    pid = 4242

    def __init__(self) -> None:
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.returncode = 0

    def wait(self, timeout=None):
        return 0

    def kill(self) -> None:
        self.returncode = -9


class _Reply:
    status_code = 200


@pytest.fixture()
def started(monkeypatch, tmp_path):
    """Replace the process start and the health check; return the record of what was started."""
    record = {"commands": [], "processes": []}

    def popen(command, **kwargs):
        record["commands"].append(command)
        process = _Process()
        record["processes"].append(process)
        return process

    monkeypatch.setattr(llamacpp_managed, "app_home", lambda: tmp_path)
    monkeypatch.setattr(llamacpp_managed, "find_llama_server", lambda configured="": "llama-server")
    monkeypatch.setattr(llamacpp_managed.subprocess, "Popen", popen)
    monkeypatch.setattr(llamacpp_managed.requests, "get", lambda url, timeout=None: _Reply())
    monkeypatch.setattr(llamacpp_managed.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(llamacpp_managed, "_start_reaper", lambda: None)
    llamacpp_managed.stop_all()
    yield record
    llamacpp_managed.stop_all()


def _profile(tmp_path, name="qwen35_9b", **overrides):
    model = tmp_path / f"{name}.gguf"
    model.write_bytes(b"gguf")
    return {
        "name": name, "backend": "llamacpp", "managed": True, "model_id": str(model), "context_limit": 16384,
        "flash_attention": True, "kv_cache_type_k": "q8_0", "kv_cache_type_v": "q8_0",
        "server_args": ["--reasoning", "off", "-fitt", "1024"], **overrides,
    }


def test_only_managed_llamacpp_profiles_are_started_by_localdeploy() -> None:
    assert llamacpp_managed.is_managed({"backend": "llamacpp", "managed": True})
    assert not llamacpp_managed.is_managed({"backend": "llamacpp"})
    assert not llamacpp_managed.is_managed({"backend": "ollama", "managed": True})


def test_the_command_passes_only_what_the_profile_sets(tmp_path) -> None:
    cmd = llamacpp_managed.command(_profile(tmp_path), "llama-server", 18080)

    assert cmd[:2] == ["llama-server", "-m"] and cmd[cmd.index("--port") + 1] == "18080"
    assert cmd[cmd.index("-c") + 1] == "16384"
    assert cmd[cmd.index("-fa") + 1] == "on"
    assert cmd[cmd.index("-ctk") + 1] == "q8_0" and cmd[cmd.index("-ctv") + 1] == "q8_0"
    assert "-ngl" not in cmd  # llama.cpp's --fit places the layers when gpu_layers is unset
    assert cmd[cmd.index("-np") + 1] == "1"
    assert cmd[-4:] == ["--reasoning", "off", "-fitt", "1024"]


def test_gpu_layers_threads_and_memory_flags_are_mapped(tmp_path) -> None:
    cmd = llamacpp_managed.command(
        _profile(tmp_path, gpu_layers="all", threads=8, batch_size=512, ubatch_size=256, mmap=False, mlock=True, server_args=["-np", "2"]),
        "llama-server", 1,
    )

    assert cmd[cmd.index("-ngl") + 1] == "99" and cmd[cmd.index("-t") + 1] == "8"
    assert cmd[cmd.index("-b") + 1] == "512" and cmd[cmd.index("-ub") + 1] == "256"
    assert "--no-mmap" in cmd and "--mlock" in cmd
    assert cmd.count("-np") == 1 and cmd[cmd.index("-np") + 1] == "2"  # the profile's own -np is kept, not doubled


def test_a_missing_model_file_is_named(tmp_path) -> None:
    profile = _profile(tmp_path)
    profile["model_id"] = str(tmp_path / "gone.gguf")
    with pytest.raises(BackendCallError, match="does not exist"):
        llamacpp_managed.command(profile, "llama-server", 1)


def test_the_first_request_starts_the_server_and_the_next_reuses_it(started, tmp_path) -> None:
    profile = _profile(tmp_path)

    first = llamacpp_managed.ensure("qwen35_9b", profile)
    second = llamacpp_managed.ensure("qwen35_9b", profile)

    assert first == second and first.startswith("http://127.0.0.1:")
    assert len(started["commands"]) == 1
    assert llamacpp_managed.status()["loaded"] == "qwen35_9b"


def test_asking_for_another_model_stops_the_first(started, tmp_path) -> None:
    llamacpp_managed.ensure("qwen35_9b", _profile(tmp_path))
    llamacpp_managed.ensure("qwen3vl_8b", _profile(tmp_path, name="qwen3vl_8b"))

    assert started["processes"][0].returncode == 0  # the 9B was stopped to make room
    assert started["processes"][1].returncode is None
    assert llamacpp_managed.status()["loaded"] == "qwen3vl_8b"


def test_changed_settings_and_a_dead_server_both_start_fresh(started, tmp_path) -> None:
    profile = _profile(tmp_path)
    llamacpp_managed.ensure("qwen35_9b", profile)
    llamacpp_managed.ensure("qwen35_9b", {**profile, "context_limit": 32768})
    assert len(started["commands"]) == 2

    started["processes"][1].returncode = 1  # it died
    llamacpp_managed.ensure("qwen35_9b", {**profile, "context_limit": 32768})
    assert len(started["commands"]) == 3


def test_a_server_that_exits_while_starting_reports_why(started, tmp_path, monkeypatch) -> None:
    def popen(command, **kwargs):
        process = _Process()
        process.returncode = 1
        return process

    monkeypatch.setattr(llamacpp_managed.subprocess, "Popen", popen)
    with pytest.raises(BackendCallError, match="exited with code 1"):
        llamacpp_managed.ensure("qwen35_9b", _profile(tmp_path))


def test_a_chat_request_for_a_managed_profile_goes_to_the_server_localdeploy_started(started, tmp_path, monkeypatch) -> None:
    sent = {}

    class Reply:
        ok = True
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "hi"}}]}

    monkeypatch.setattr(llamacpp.requests, "post", lambda url, json=None, timeout=None: sent.update(url=url) or Reply())
    profile = _profile(tmp_path)
    prepared = {
        "profile": profile, "model": profile["model_id"], "system_prompt": "s", "prompt": "p", "temperature": 0.1,
        "top_p": 0.9, "max_output_tokens_used": 16, "timeout_seconds": 5,
    }

    assert llamacpp.call_llamacpp(prepared) == "hi"
    assert sent["url"].startswith(llamacpp_managed.status()["server"]) and sent["url"].endswith("/v1/chat/completions")
    assert len(started["commands"]) == 1
