"""Chat models served by a ``llama-server`` that LocalDeploy starts, switches and stops itself.

A ``llamacpp`` profile with ``"managed": true`` needs no separate launch step and no fixed port. The first request
for it starts ``llama-server`` on a free loopback port with the profile's settings (context, flash attention, KV
cache types, GPU layers, ``server_args``), waits until it is ready and forwards to it. Only one managed chat model is
kept loaded, because an 8 GB card holds one: asking for another profile stops the current one first. The server is
restarted if it dies, stopped when LocalDeploy exits (and, if ``idle_stop_seconds`` is set, after that long without a
request), and a ``llama-server`` left over by a LocalDeploy that was killed outright is ended on the next start.

This is the chat-model counterpart of the decision models' ``llamacpp`` adapter in ``decision.py``, and uses its
``llama-server`` lookup (``LLAMACPP_SERVER_PATH``, PATH, a WinGet install).
"""

from __future__ import annotations

import atexit
import json
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from ..utils import BackendCallError, app_home
from .decision import LLAMACPP_GET, _free_port, find_llama_server

_LOCK = threading.RLock()
_ACTIVE: Optional["_Server"] = None
_REAPER_STARTED = False


def is_managed(profile: Dict[str, Any]) -> bool:
    return str(profile.get("backend", "")).lower() == "llamacpp" and bool(profile.get("managed"))


def command(profile: Dict[str, Any], server: str, port: int) -> List[str]:
    """The ``llama-server`` command line for a profile. Only the fields the profile sets are passed, so llama.cpp's
    own ``--fit`` places the layers when ``gpu_layers`` is left unset."""
    model = str(profile.get("model_id") or "").strip()
    if not model:
        raise BackendCallError(f"Profile '{profile.get('name')}' needs a model_id (a .gguf path).")
    if not Path(model).is_file():
        raise BackendCallError(f"Profile '{profile.get('name')}': the model file {model} does not exist.")
    args = [str(item) for item in (profile.get("server_args") or []) if str(item) != ""]
    cmd = [server, "-m", model, "--host", "127.0.0.1", "--port", str(port), "-c", str(int(profile.get("context_limit") or 8192))]
    if profile.get("flash_attention") is True:
        cmd += ["-fa", "on"]
    for flag, key in (("-ctk", "kv_cache_type_k"), ("-ctv", "kv_cache_type_v")):
        if profile.get(key):
            cmd += [flag, str(profile[key])]
    layers = profile.get("gpu_layers")
    if str(layers).lower() == "all":
        cmd += ["-ngl", "99"]
    elif layers not in (None, ""):
        cmd += ["-ngl", str(int(layers))]
    for flag, key in (("-b", "batch_size"), ("-ub", "ubatch_size"), ("-t", "threads")):
        if profile.get(key):
            cmd += [flag, str(int(profile[key]))]
    if profile.get("mmap") is False:
        cmd.append("--no-mmap")
    if profile.get("mlock") is True:
        cmd.append("--mlock")
    if not {"-np", "--parallel"} & set(args):
        cmd += ["-np", "1"]  # one request at a time: more slots would split the context between them
    return cmd + args


def _fingerprint(profile: Dict[str, Any]) -> str:
    keys = ("model_id", "context_limit", "flash_attention", "kv_cache_type_k", "kv_cache_type_v", "gpu_layers",
            "batch_size", "ubatch_size", "threads", "mmap", "mlock", "server_args")
    return json.dumps({key: profile.get(key) for key in keys}, sort_keys=True, default=str)


def _pid_path() -> Path:
    return app_home() / "logs" / "chat-llamacpp.pid"


def _stop_left_over() -> None:
    """End the llama-server that a LocalDeploy killed outright left behind, matched by id and start time so no other
    llama-server (a user's own, a decision model's) is touched even if the id has been reused."""
    path = _pid_path()
    try:
        pid_text, created_text = path.read_text(encoding="utf-8").split()
        pid, created = int(pid_text), float(created_text)
    except (OSError, ValueError):
        return
    try:
        import psutil

        process = psutil.Process(pid)
        if abs(process.create_time() - created) < 1 and "llama-server" in process.name().lower():
            process.terminate()
            process.wait(timeout=10)
    except Exception:  # noqa: BLE001 - already gone, or not the process that was recorded
        pass
    path.unlink(missing_ok=True)


class _Server:
    def __init__(self, name: str, profile: Dict[str, Any]) -> None:
        self.name = name
        self.fingerprint = _fingerprint(profile)
        self.profile = profile
        self.process: Optional[subprocess.Popen] = None
        self.port = 0
        self.last_used = time.monotonic()
        self.log_path = app_home() / "logs" / f"chat-llamacpp-{re.sub(r'[^A-Za-z0-9_.-]', '_', name)}.log"

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self) -> None:
        configured = str(self.profile.get("server_path") or "").strip()
        server = find_llama_server(configured)
        if not server:
            raise BackendCallError(f"Profile '{self.name}' needs llama.cpp's llama-server. {LLAMACPP_GET}")
        self.port = int(self.profile.get("port") or _free_port())
        cmd = command({**self.profile, "name": self.name}, server, self.port)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        _stop_left_over()
        with open(self.log_path, "ab") as log:
            self.process = subprocess.Popen(  # noqa: S603 - the command is built here from validated profile fields
                cmd, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        try:
            import psutil

            _pid_path().write_text(f"{self.process.pid} {psutil.Process(self.process.pid).create_time()}", encoding="utf-8")
        except Exception:  # noqa: BLE001 - only a convenience for the next start
            pass
        deadline = time.monotonic() + float(self.profile.get("startup_timeout_seconds") or 300)
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise BackendCallError(
                    f"llama-server for '{self.name}' exited with code {self.process.returncode} while starting. "
                    f"Last log lines: {self.log_tail()}"
                )
            try:
                if requests.get(f"{self.base_url}/health", timeout=2).status_code == 200:
                    return
            except requests.RequestException:
                pass  # still loading the model
            time.sleep(0.5)
        self.stop()
        raise BackendCallError(f"llama-server for '{self.name}' was not ready in time. Last log lines: {self.log_tail()}")

    def log_tail(self, lines: int = 6) -> str:
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return "(no log)"
        return " | ".join(text.strip().splitlines()[-lines:]) or "(empty log)"

    def stop(self) -> None:
        process, self.process = self.process, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        if process is not None:
            _pid_path().unlink(missing_ok=True)


def ensure(name: str, profile: Dict[str, Any]) -> str:
    """Make sure the profile's llama-server is running, stopping any other managed one, and return its base URL."""
    global _ACTIVE
    with _LOCK:
        current = _ACTIVE
        if current is not None and current.name == name and current.fingerprint == _fingerprint(profile) and current.alive():
            current.last_used = time.monotonic()
            return current.base_url
        if current is not None:
            current.stop()  # a different profile, changed settings, or a server that died: start fresh
            _ACTIVE = None
        fresh = _Server(name, profile)
        fresh.start()
        _ACTIVE = fresh
        _start_reaper()
        return fresh.base_url


def stop_all() -> None:
    global _ACTIVE
    with _LOCK:
        if _ACTIVE is not None:
            _ACTIVE.stop()
            _ACTIVE = None


def status() -> Dict[str, Any]:
    with _LOCK:
        if _ACTIVE is None or not _ACTIVE.alive():
            return {"loaded": None}
        return {"loaded": _ACTIVE.name, "server": _ACTIVE.base_url}


def _start_reaper() -> None:
    """Stop an idle model so the GPU is free again, for profiles that set ``idle_stop_seconds``."""
    global _REAPER_STARTED
    if _REAPER_STARTED:
        return
    _REAPER_STARTED = True

    def watch() -> None:
        while True:
            time.sleep(15)
            with _LOCK:
                active = _ACTIVE
                limit = float((active.profile.get("idle_stop_seconds") or 0)) if active else 0
                if active and limit > 0 and time.monotonic() - active.last_used > limit:
                    stop_all()

    threading.Thread(target=watch, name="llamacpp-idle-reaper", daemon=True).start()


atexit.register(stop_all)
