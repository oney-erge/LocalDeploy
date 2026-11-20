"""Run dependency-free Node unit tests for pure frontend module exports."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from tests import node_flags
from tests.node_flags import node_module_flags

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_frontend_node_units() -> None:
    tests = sorted((PROJECT_ROOT / "tests" / "js").glob("*.test.mjs"))
    result = subprocess.run(
        ["node", *node_module_flags(), "--test", *map(str, tests)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_module_flag_is_passed_only_when_node_accepts_it(monkeypatch) -> None:
    def fake_run(returncode):
        return lambda *_args, **_kwargs: subprocess.CompletedProcess([], returncode, "", "")

    node_flags.node_module_flags.cache_clear()
    monkeypatch.setattr(node_flags.subprocess, "run", fake_run(0))
    assert node_flags.node_module_flags() == (node_flags.MODULE_FLAG,)

    node_flags.node_module_flags.cache_clear()
    monkeypatch.setattr(node_flags.subprocess, "run", fake_run(9))  # Node 25: "bad option"
    assert node_flags.node_module_flags() == ()
    node_flags.node_module_flags.cache_clear()
