"""Packaging guards: app-home resolution, CLI entry point, version consistency."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time

import pytest

import localdeploy
from localdeploy import cli
from localdeploy.utils import app_home, web_dir

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_app_home_env_override_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALDEPLOY_HOME", str(tmp_path / "custom"))
    assert app_home() == tmp_path / "custom"


def test_app_home_in_source_checkout_is_repo_root(monkeypatch):
    monkeypatch.delenv("LOCALDEPLOY_HOME", raising=False)
    # This test runs from the checkout, where config.example.json exists.
    assert app_home() == PROJECT_ROOT


def test_web_dir_ships_all_ui_assets():
    for asset in ("index.html", "styles.css", "favicon.png", "logo.svg"):
        assert (web_dir() / asset).is_file(), asset
    expected_modules = {
        "app.js", "shared.js", "system.js", "models.js", "chat.js",
        "benchmark.js", "benchmark-views.js",
    }
    assert {path.name for path in (web_dir() / "js").glob("*.js")} == expected_modules


def test_cli_version_flag_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert localdeploy.__version__ in capsys.readouterr().out


def test_changelog_covers_current_version():
    changelog = (PROJECT_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"## {localdeploy.__version__}" in changelog, (
        "CHANGELOG.md has no section for localdeploy.__version__ - "
        "bump both together."
    )


def test_runtime_dependencies_have_one_source():
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'dynamic = ["version", "dependencies"]' in pyproject
    assert 'dependencies = { file = ["requirements.txt"] }' in pyproject
    assert not (PROJECT_ROOT / "requirements-dev.txt").exists()


def test_docker_persists_localdeploy_runtime_state():
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    compose = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "LOCALDEPLOY_HOME=/data/localdeploy" in dockerfile
    assert "localdeploy-data:/data/localdeploy" in compose
    assert "localdeploy-data:" in compose


def test_bundled_ollama_cloud_models_are_disabled():
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    compose = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    env_example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    windows_launcher = (PROJECT_ROOT / "scripts" / "start.ps1").read_text(encoding="utf-8")
    unix_launcher = (PROJECT_ROOT / "scripts" / "start.sh").read_text(encoding="utf-8")

    for content in (dockerfile, compose, env_example, windows_launcher, unix_launcher):
        assert "OLLAMA_NO_CLOUD" in content
    assert "OLLAMA_NO_CLOUD=true" in dockerfile
    assert "OLLAMA_NO_CLOUD=true" in compose
    assert "OLLAMA_NO_CLOUD=true" in env_example


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _process_is_active(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat_path = Path(f"/proc/{pid}/stat")
    if stat_path.is_file():
        fields = stat_path.read_text(encoding="utf-8").split()
        return len(fields) < 3 or fields[2] != "Z"
    return True


def _prepare_unix_launcher(tmp_path: Path):
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    for name in ("start.sh", "stop.sh", "install-utils.sh"):
        shutil.copy2(PROJECT_ROOT / "scripts" / name, scripts_dir / name)

    requirements = tmp_path / "requirements.txt"
    requirements.write_text("", encoding="utf-8")
    marker_dir = tmp_path / ".venv"
    (marker_dir / "bin").mkdir(parents=True)
    (marker_dir / "bin" / "activate").write_text("", encoding="utf-8")
    (marker_dir / "requirements.sha256").write_text(
        hashlib.sha256(b"").hexdigest() + "\n", encoding="utf-8"
    )

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    env = os.environ.copy()
    for name in ("API_HOST", "API_PORT", "OLLAMA_BASE_URL", "START_OLLAMA", "OLLAMA_NO_CLOUD"):
        env.pop(name, None)
    env.update(
        {
            "NO_BROWSER": "1",
            "PYTHON": shutil.which("python3") or shutil.which("python") or "python3",
            "PATH": f"{fake_bin}{os.pathsep}/usr/bin{os.pathsep}/bin",
        }
    )
    return scripts_dir, fake_bin, env


@pytest.mark.skipif(os.name == "nt" or shutil.which("bash") is None, reason="Unix launcher test")
def test_unix_launcher_reads_addresses_from_dotenv(tmp_path):
    scripts_dir, fake_bin, env = _prepare_unix_launcher(tmp_path)
    (tmp_path / ".env").write_text(
        "API_HOST=0.0.0.0\n"
        "API_PORT=8123\n"
        "OLLAMA_BASE_URL=http://127.0.0.1:19999/\n",
        encoding="utf-8",
    )

    curl_log = tmp_path / "curl-args.txt"
    uvicorn_log = tmp_path / "uvicorn-args.txt"
    _write_executable(
        fake_bin / "curl",
        f'#!/usr/bin/env bash\nprintf "%s" "$*" > "{curl_log}"\n',
    )
    _write_executable(
        fake_bin / "uvicorn",
        f'#!/usr/bin/env bash\nprintf "%s" "$*" > "{uvicorn_log}"\n',
    )
    result = subprocess.run(
        ["bash", str(scripts_dir / "start.sh")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )

    assert "LocalDeploy UI:  http://127.0.0.1:8123/ui" in result.stdout
    assert curl_log.read_text(encoding="utf-8").endswith("http://127.0.0.1:19999/api/tags")
    assert uvicorn_log.read_text(encoding="utf-8") == "api_server:app --host 0.0.0.0 --port 8123"


@pytest.mark.skipif(os.name == "nt" or shutil.which("bash") is None, reason="Unix launcher test")
def test_unix_launcher_starts_and_stops_managed_ollama(tmp_path):
    scripts_dir, fake_bin, env = _prepare_unix_launcher(tmp_path)
    ready_marker = tmp_path / "ollama-ready"
    ollama_env_log = tmp_path / "ollama-env.txt"
    (tmp_path / ".env").write_text(
        "API_HOST=127.0.0.1\n"
        "API_PORT=8124\n"
        "OLLAMA_BASE_URL=http://127.0.0.1:19999/\n"
        "START_OLLAMA=true\n"
        "OLLAMA_NO_CLOUD=true\n",
        encoding="utf-8",
    )
    _write_executable(
        fake_bin / "curl",
        f'#!/usr/bin/env bash\n[ -f "{ready_marker}" ]\n',
    )
    _write_executable(fake_bin / "uvicorn", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(
        fake_bin / "ollama",
        "#!/usr/bin/env bash\n"
        "[ \"${1:-}\" = serve ] || exit 2\n"
        f'printf "%s\\t%s" "$OLLAMA_NO_CLOUD" "$OLLAMA_HOST" > "{ollama_env_log}"\n'
        f'touch "{ready_marker}"\n'
        "trap 'exit 0' TERM INT\n"
        "while :; do sleep 1; done\n",
    )

    ollama_pid = None
    try:
        started = subprocess.run(
            ["bash", str(scripts_dir / "start.sh")],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        assert "starting Ollama" in started.stdout
        assert "Ollama is ready" in started.stdout
        assert ollama_env_log.read_text(encoding="utf-8") == "true\t127.0.0.1:19999"
        ollama_pid = int((tmp_path / "logs" / "ollama.pid").read_text(encoding="utf-8"))
        os.kill(ollama_pid, 0)

        stopped = subprocess.run(
            ["bash", str(scripts_dir / "stop.sh"), "--ollama"],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        assert "stopped Ollama" in stopped.stdout
        assert not (tmp_path / "logs" / "ollama.pid").exists()
        assert not (tmp_path / "logs" / "api_server.pid").exists()
        for _ in range(20):
            if not _process_is_active(ollama_pid):
                break
            time.sleep(0.05)
        else:
            pytest.fail("managed Ollama process was not stopped")
    finally:
        if ollama_pid is not None:
            try:
                os.kill(ollama_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.skipif(os.name == "nt" or shutil.which("bash") is None, reason="Unix launcher test")
def test_unix_launcher_rejects_non_loopback_ollama_url(tmp_path):
    scripts_dir, _fake_bin, env = _prepare_unix_launcher(tmp_path)
    (tmp_path / ".env").write_text(
        "OLLAMA_BASE_URL=http://192.0.2.10:11434\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        ["bash", str(scripts_dir / "start.sh")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode != 0
    assert "must use localhost or a loopback IP address" in result.stderr
    assert not (tmp_path / "logs" / "ollama.pid").exists()
