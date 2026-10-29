"""scripts/make_decision_gguf.py: the command line around quantized_gguf (tested in test_decision.py).
Nothing is downloaded or run for real."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from localdeploy.backends import decision

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "make_decision_gguf.py"
spec = importlib.util.spec_from_file_location("make_decision_gguf", SCRIPT)
tool = importlib.util.module_from_spec(spec)
sys.modules["make_decision_gguf"] = tool
spec.loader.exec_module(tool)


def test_it_makes_the_file_and_prints_the_profile_line(monkeypatch, tmp_path, capsys) -> None:
    quantizer = tmp_path / "llama-quantize"
    quantizer.write_bytes(b"x")
    asked = {}

    def fake_quantized(repo, quant, out_dir, *, quantizer, source_file=None, threads=0, log=None):
        asked.update(repo=repo, quant=quant, out_dir=out_dir, quantizer=quantizer, threads=threads)
        target = out_dir / "vllm-d3-lite-Q4_K_M.gguf"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"q4")
        return target

    monkeypatch.setattr(tool, "quantized_gguf", fake_quantized)
    out = tmp_path / "models"

    assert tool.main(["ggml-org/vllm-d3-lite-GGUF", "--out", str(out), "--llama-quantize", str(quantizer), "--threads", "6"]) == 0

    assert asked == {"repo": "ggml-org/vllm-d3-lite-GGUF", "quant": "Q4_K_M", "out_dir": out, "quantizer": str(quantizer), "threads": 6}
    printed = capsys.readouterr().out
    assert '"adapter": "llamacpp"' in printed and "vllm-d3-lite-Q4_K_M.gguf" in printed


def test_a_failure_is_reported_not_hidden(monkeypatch, tmp_path, capsys) -> None:
    def failing(*args, **kwargs):
        raise decision.DecisionUnavailable("llama-quantize failed (exit 3) making vllm-d3-lite-Q4_K_M.gguf: bad file")

    monkeypatch.setattr(tool, "quantized_gguf", failing)
    assert tool.main(["ggml-org/vllm-d3-lite-GGUF", "--out", str(tmp_path), "--llama-quantize", "q"]) == 1
    assert "exit 3" in capsys.readouterr().err


def test_a_missing_quantizer_says_where_to_get_it(monkeypatch, capsys) -> None:
    monkeypatch.setattr(tool, "find_llama_quantize", lambda server=None: None)
    assert tool.main(["ggml-org/vllm-d3-lite-GGUF"]) == 1
    assert "winget install ggml.llamacpp" in capsys.readouterr().err
