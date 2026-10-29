"""scripts/make_decision_gguf.py: download a BF16 GGUF and quantize it. Nothing is downloaded or run for real."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "make_decision_gguf.py"
spec = importlib.util.spec_from_file_location("make_decision_gguf", SCRIPT)
tool = importlib.util.module_from_spec(spec)
sys.modules["make_decision_gguf"] = tool
spec.loader.exec_module(tool)

FILES = [
    "README.md",
    "mmproj-vllm-d3-lite-BF16.gguf",
    "vllm-d3-lite-BF16.gguf",
    "vllm-d3-lite-Q8_0.gguf",
    "mmproj-vllm-d3-lite-Q8_0.gguf",
]


def test_the_bf16_model_file_is_picked_not_the_vision_projector() -> None:
    assert tool.pick_source(FILES, None) == "vllm-d3-lite-BF16.gguf"


def test_a_named_source_file_must_exist_in_the_repository() -> None:
    assert tool.pick_source(FILES, "vllm-d3-lite-Q8_0.gguf") == "vllm-d3-lite-Q8_0.gguf"
    with pytest.raises(SystemExit, match="not a model file"):
        tool.pick_source(FILES, "mmproj-vllm-d3-lite-BF16.gguf")


def test_a_repository_without_a_half_precision_file_says_so() -> None:
    with pytest.raises(SystemExit, match="No BF16/F16"):
        tool.pick_source(["only-Q8_0.gguf"], None)


def test_the_quantizer_is_found_next_to_llama_server(monkeypatch, tmp_path) -> None:
    server = tmp_path / ("llama-server.exe" if tool.os.name == "nt" else "llama-server")
    quantizer = tmp_path / ("llama-quantize.exe" if tool.os.name == "nt" else "llama-quantize")
    server.write_bytes(b"x")
    quantizer.write_bytes(b"x")
    monkeypatch.setenv("LLAMACPP_SERVER_PATH", str(server))
    assert tool.find_quantizer(None) == str(quantizer)


def test_a_missing_quantizer_says_where_to_get_it(monkeypatch) -> None:
    monkeypatch.delenv("LLAMACPP_SERVER_PATH", raising=False)
    monkeypatch.setattr(tool.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit, match="llama.cpp/releases"):
        tool.find_quantizer(None)


def test_it_downloads_quantizes_and_prints_the_profile_line(monkeypatch, tmp_path, capsys) -> None:
    quantizer = tmp_path / "llama-quantize"
    quantizer.write_bytes(b"x")
    source = tmp_path / "cache" / "vllm-d3-lite-BF16.gguf"
    source.parent.mkdir()
    source.write_bytes(b"bf16")
    ran = {}

    class _Api:
        def list_repo_files(self, repo):
            ran["repo"] = repo
            return FILES

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=_Api, hf_hub_download=lambda repo, name, local_files_only=False: str(source)))

    def fake_run(command, check=False):
        ran["command"] = command
        Path(command[2]).write_bytes(b"q4")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(tool.subprocess, "run", fake_run)
    out = tmp_path / "models"

    assert tool.main(["ggml-org/vllm-d3-lite-GGUF", "--quant", "Q4_K_M", "--out", str(out), "--llama-quantize", str(quantizer), "--threads", "6"]) == 0

    assert ran["repo"] == "ggml-org/vllm-d3-lite-GGUF"
    assert ran["command"] == [str(quantizer), str(source), str(out / "vllm-d3-lite-Q4_K_M.gguf"), "Q4_K_M", "6"]
    printed = capsys.readouterr().out
    assert '"adapter": "llamacpp"' in printed and "vllm-d3-lite-Q4_K_M.gguf" in printed


def test_a_failed_quantize_is_reported_not_hidden(monkeypatch, tmp_path) -> None:
    quantizer = tmp_path / "llama-quantize"
    quantizer.write_bytes(b"x")
    source = tmp_path / "vllm-d3-lite-BF16.gguf"
    source.write_bytes(b"bf16")

    class _Api:
        def list_repo_files(self, repo):
            return FILES

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=_Api, hf_hub_download=lambda *a, **k: str(source)))
    monkeypatch.setattr(tool.subprocess, "run", lambda command, check=False: types.SimpleNamespace(returncode=3))

    assert tool.main(["ggml-org/vllm-d3-lite-GGUF", "--out", str(tmp_path / "out"), "--llama-quantize", str(quantizer)]) == 3
