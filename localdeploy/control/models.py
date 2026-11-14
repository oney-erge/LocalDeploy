"""Steps 5-6 - model lifecycle.

Step 5:  POST /models/pull    -> stream `ollama pull`, gated by the fit-check
Step 6:  GET  /system/status  -> served model(s), Ollama health, VRAM
         POST /models/serve    -> warm a model (Ollama keep-alive)
         POST /models/stop      -> unload a model
         POST /models/switch    -> pivot from one model to another

Ollama is driven fully. llama.cpp lifecycle is process-managed by the existing
scripts, so serve/stop return clear guidance instead of spawning processes.

Also covers importing GGUF files that aren't in any registry:
         POST /system/check-local-gguf -> validate a local .gguf path
         POST /models/import-url        -> download a direct GGUF URL, register
                                            it with Ollama, create a profile
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import unquote, urlsplit

import requests
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..backends import llamacpp_managed
from ..backends.llamacpp import llama_health
from ..utils import BackendCallError, app_home, env_float, offline_mode, require_gpu_only
from . import _ollama, calibration, monitor
from ._config import ensure_profile_for_model
from .fit import FitRequest, _base_family, _parse_quant, fit_check
from .hardware import detect_hardware

router = APIRouter()


def _ollama_binary_installed() -> bool:
    """Best-effort check that the ollama CLI/binary exists on this machine.

    A missing binary is a different problem than "installed but not running" -
    the pull-blocked message needs to tell those apart so the UI can offer
    "install" vs. "just start it" instead of one generic error.
    """
    if shutil.which("ollama"):
        return True
    candidate = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
    return candidate.exists()


@router.get("/system/ollama-status")
def ollama_status() -> Dict[str, Any]:
    installed = _ollama_binary_installed()
    reachable = False
    if installed:
        _, err = _ollama.list_installed()
        reachable = err is None
    return {"success": True, "installed": installed, "reachable": reachable}


def _resolve_ollama_exe() -> Optional[str]:
    exe = shutil.which("ollama")
    if exe:
        return exe
    candidate = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
    return str(candidate) if candidate.exists() else None


def _try_start_ollama() -> bool:
    """Launch `ollama serve` detached and poll briefly for it to come up."""
    exe = _resolve_ollama_exe()
    if not exe:
        return False
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        subprocess.Popen(
            [exe, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=creationflags
        )
    except (OSError, subprocess.SubprocessError):
        return False
    for _ in range(10):
        time.sleep(0.5)
        _, err = _ollama.list_installed()
        if err is None:
            return True
    return False


@router.post("/system/install-ollama")
def install_ollama() -> Dict[str, Any]:
    """Best-effort automatic install via winget, plus a start attempt either way.

    Returns a consistent success/already_installed/error shape for the
    frontend. Handles both
    "not installed at all" and "installed but not running" - the two cases the
    pull flow's generic "Ollama is not reachable" error used to lump together.
    """
    if _ollama_binary_installed():
        _, err = _ollama.list_installed()
        if err is None:
            return {"success": True, "already_installed": True, "message": "Ollama is already installed and running."}
        if _try_start_ollama():
            return {
                "success": True,
                "already_installed": True,
                "message": "Ollama was installed but not running - started it. Retry the pull shortly.",
            }
        return {
            "success": False,
            "already_installed": True,
            "error": "Ollama is installed but not reachable, and it could not be started automatically. "
            "Start it manually (run 'ollama serve' or launch the Ollama app), then retry.",
        }

    if platform.system() != "Windows":
        return {
            "success": False,
            "error": "Automatic install is only wired up for Windows (winget) here. "
            "Install Ollama from https://ollama.com/download.",
        }
    if not shutil.which("winget"):
        return {
            "success": False,
            "error": "winget is not available on this system. Install Ollama manually from "
            "https://ollama.com/download.",
        }

    cmd = ["winget", "install", "-e", "--id", "Ollama.Ollama", "--accept-source-agreements", "--accept-package-agreements"]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"success": False, "error": f"Could not run winget: {exc}"}

    if completed.returncode != 0:
        stderr = (completed.stderr or completed.stdout or "").strip()
        return {"success": False, "error": stderr[-1000:] or "winget install failed."}

    if not _ollama_binary_installed():
        return {
            "success": False,
            "error": "winget reported success but the ollama binary was not found yet. "
            "It may need a new terminal/PATH refresh - restart the LocalDeploy API and try again.",
        }
    _try_start_ollama()
    return {
        "success": True,
        "already_installed": False,
        "message": "Installed Ollama. It may take a moment to start serving - retry the pull shortly.",
    }


def _sse(obj: Dict[str, Any]) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


# A large layer count forces Ollama to offload everything it can to the GPU.
_FORCE_GPU_LAYERS = 999


def _resolve_num_gpu(device: Optional[str], num_gpu: Optional[int]) -> Optional[int]:
    """Map a friendly device choice to an Ollama ``num_gpu`` value.

    - explicit ``num_gpu`` wins (advanced override)
    - "cpu"  -> 0   (force CPU)
    - "gpu"  -> 999 (force max GPU offload)
    - "auto"/None -> None (Ollama decides; identical to prior behaviour)
    """
    if num_gpu is not None:
        return num_gpu
    d = (device or "").strip().lower()
    if d == "cpu":
        return 0
    if d == "gpu":
        return _FORCE_GPU_LAYERS
    return None


def _target_label(num_gpu: Optional[int]) -> str:
    if num_gpu is None:
        return "auto"
    if num_gpu == 0:
        return "CPU"
    if num_gpu == _FORCE_GPU_LAYERS:
        return "GPU"
    return f"{num_gpu} GPU layer(s)"


def _placement(size: Any, size_vram: Any) -> Dict[str, Any]:
    """Derive a GPU/CPU placement label from total vs VRAM-resident bytes."""
    if isinstance(size, int) and isinstance(size_vram, int) and size > 0:
        pct = round(100 * size_vram / size)
        label = "GPU" if pct >= 99 else ("CPU" if pct <= 1 else "Split")
        return {"gpu_percent": pct, "placement": label}
    return {"gpu_percent": None, "placement": None}


def _matches_model_name(running_name: Any, model_id: str) -> bool:
    name = str(running_name or "").strip().lower()
    requested = str(model_id or "").strip().lower()
    if not name or not requested:
        return False
    if name == requested:
        return True
    # Ollama may omit or add the implicit latest tag. A request for an exact
    # non-latest tag must never match another size from the same family.
    if ":" not in requested:
        return name == f"{requested}:latest" or name.split(":", 1)[0] == requested
    if requested.endswith(":latest") and ":" not in name:
        return name == requested.rsplit(":", 1)[0]
    return False


def _expected_placement(num_gpu: Optional[int]) -> Optional[str]:
    if num_gpu == 0:
        return "CPU"
    if num_gpu == _FORCE_GPU_LAYERS:
        return "GPU"
    return None


def _resolve_target(
    profile: Optional[str], model: Optional[str], backend: Optional[str]
) -> Tuple[str, Optional[str], Dict[str, Any]]:
    """Return (backend, model_id, profile_dict). Raises ValueError on bad input."""
    if profile:
        from api_server import load_config  # lazy: api_server owns config loading

        prof = load_config().get("profiles", {}).get(profile)
        if not prof:
            raise ValueError(f"Unknown profile '{profile}'.")
        return prof.get("backend", "ollama"), prof.get("model_id"), prof
    if not model:
        raise ValueError("Provide 'profile' or 'model'.")
    return (backend or "ollama"), model, {}


# --- Step 6: status ----------------------------------------------------------


@router.get("/system/status")
def system_status() -> Dict[str, Any]:
    running, run_error = _ollama.list_running()
    for m in running:
        m.update(_placement(m.get("size"), m.get("size_vram")))
        if isinstance(m.get("size"), int):
            m["size_mb"] = round(m["size"] / 1_000_000)
        if isinstance(m.get("size_vram"), int):
            m["size_vram_mb"] = round(m["size_vram"] / 1_000_000)
        m["activity"] = "loaded"
        m["activity_note"] = (
            "Ollama reports this model is loaded/warm. It does not expose per-request activity here."
        )
    hardware = detect_hardware()
    return {
        "success": True,
        "ollama": {"reachable": run_error is None, "running": running, "error": run_error},
        "served_models": [m.get("name") for m in running],
        "hardware": {
            "gpu_available": hardware["gpu_available"],
            "gpus": hardware["gpus"],
            "system": hardware.get("system"),
        },
        "require_gpu_only": require_gpu_only(),
    }


# --- GGUF import: local file / direct URL ------------------------------------
# Covers models in no registry (a file already on disk, or hosted on GitHub
# Releases or another direct host). A local file becomes a llama.cpp profile
# (no download); a URL is downloaded then registered with Ollama via
# /api/create so it gets the same serve/stop/delete lifecycle as a pull.


class LocalGgufCheckRequest(BaseModel):
    path: str


@router.post("/system/check-local-gguf")
def check_local_gguf(req: LocalGgufCheckRequest) -> Dict[str, Any]:
    """Validate a local GGUF path before it becomes a llama.cpp profile."""
    raw = (req.path or "").strip().strip('"')
    if not raw:
        return {"success": False, "error": "Enter a file path."}
    if not raw.lower().endswith(".gguf"):
        return {"success": False, "error": "Path must end in .gguf."}
    try:
        path = Path(raw).expanduser()
    except (OSError, ValueError) as exc:
        return {"success": False, "error": f"Invalid path: {exc}"}
    if not path.is_file():
        return {"success": False, "exists": False, "error": f"File not found: {path}"}
    return {"success": True, "exists": True, "path": str(path), "size_bytes": path.stat().st_size}


def _imports_dir() -> Path:
    directory = app_home() / "imports"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _safe_model_name(raw: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9._-]+", "-", (raw or "").strip()).strip("-.").lower()
    return name or "imported-model"


_MAX_IMPORT_GB = env_float("MAX_GGUF_IMPORT_GB", 128.0)
_DOWNLOAD_CHUNK = 8 * 1024 * 1024


class ImportUrlRequest(BaseModel):
    url: str
    model: Optional[str] = None  # target Ollama model name; derived from the URL filename if omitted


@router.post("/models/import-url")
def models_import_url(req: ImportUrlRequest):
    if require_gpu_only():
        return {"success": False, "error": "GPU-only mode is enabled; refusing GGUF import."}
    if offline_mode():
        return {"success": False, "error": "offline mode (OFFLINE=true): URL import skipped - no egress."}

    parsed = urlsplit(req.url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return {"success": False, "error": "Only http:// or https:// URLs are supported."}
    filename = unquote(Path(parsed.path).name) or "model.gguf"
    if not filename.lower().endswith(".gguf"):
        return {"success": False, "error": f"URL does not point at a .gguf file (got '{filename}')."}
    target_model = _safe_model_name(req.model or Path(filename).stem)
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", target_model):
        return {"success": False, "error": f"Could not derive a valid model name from '{req.model or filename}'."}

    def event_stream():
        tmp_path = _imports_dir() / f"{target_model}.{time.time_ns()}.part"
        try:
            yield _sse({"status": f"starting download for {target_model}", "model": target_model, "source_url": req.url})
            try:
                response = requests.get(req.url, stream=True, timeout=(10, 60))
            except requests.Timeout:
                yield _sse({"error": f"Connection to {req.url} timed out."})
                return
            except requests.ConnectionError as exc:
                yield _sse({"error": f"Could not reach {req.url}: {exc}"})
                return
            except requests.RequestException as exc:
                yield _sse({"error": f"Download failed: {exc}"})
                return
            with response:
                if not response.ok:
                    yield _sse({"error": f"Download failed with HTTP {response.status_code}."})
                    return
                try:
                    total = int(response.headers.get("content-length") or 0)
                except (TypeError, ValueError):
                    total = 0
                max_bytes = _MAX_IMPORT_GB * 1_000_000_000
                if total and total > max_bytes:
                    yield _sse(
                        {
                            "error": f"File is {total / 1e9:.1f} GB, over the {_MAX_IMPORT_GB:.0f} GB import "
                            "limit (set MAX_GGUF_IMPORT_GB to change)."
                        }
                    )
                    return
                hasher = hashlib.sha256()
                downloaded = 0
                last_emit = 0.0
                try:
                    with open(tmp_path, "wb") as fh:
                        for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK):
                            if not chunk:
                                continue
                            downloaded += len(chunk)
                            if downloaded > max_bytes:
                                raise BackendCallError(
                                    f"Download exceeded the {_MAX_IMPORT_GB:.0f} GB import limit."
                                )
                            hasher.update(chunk)
                            fh.write(chunk)
                            now = time.time()
                            if now - last_emit > 0.4:
                                last_emit = now
                                yield _sse(
                                    {
                                        "status": "downloading",
                                        "digest": "download",
                                        "total": total or None,
                                        "completed": downloaded,
                                    }
                                )
                except BackendCallError as exc:
                    yield _sse({"error": str(exc)})
                    return
                except requests.RequestException as exc:
                    yield _sse({"error": f"Download interrupted: {exc}"})
                    return
                except OSError as exc:
                    yield _sse({"error": f"Could not write downloaded file: {exc}"})
                    return
            digest = hasher.hexdigest()
            yield _sse(
                {"status": "downloaded", "digest": "download", "total": total or downloaded, "completed": downloaded}
            )

            yield _sse({"status": "uploading to Ollama"})
            try:
                if not _ollama.blob_exists(digest):
                    _ollama.push_blob(tmp_path, digest)
            except BackendCallError as exc:
                yield _sse({"error": str(exc)})
                return
            except requests.ConnectionError:
                yield _sse({"error": "Ollama is not running or is unreachable. Start Ollama and retry."})
                return
            except requests.RequestException as exc:
                yield _sse({"error": f"Could not upload the file to Ollama: {exc}"})
                return

            try:
                for event in _ollama.create_stream(target_model, {filename: f"sha256:{digest}"}):
                    yield _sse(event)
                    if event.get("error"):
                        return
            except requests.ConnectionError:
                yield _sse({"error": "Ollama is not running or is unreachable. Start Ollama and retry."})
                return
            except requests.RequestException as exc:
                yield _sse({"error": f"Ollama create failed: {exc}"})
                return

            profile_name, created, _err = ensure_profile_for_model(target_model)
            yield _sse(
                {
                    "status": "success",
                    "done": True,
                    "model": target_model,
                    "profile": profile_name,
                    "profile_created": created,
                }
            )
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# --- Step 5: pull (streamed, fit-gated) --------------------------------------


class PullRequest(BaseModel):
    model: Optional[str] = None
    profile: Optional[str] = None
    allow_override: bool = False
    free_vram_mb: Optional[int] = None


@router.post("/models/pull")
def models_pull(req: PullRequest):
    try:
        backend, model_id, _ = _resolve_target(req.profile, req.model, "ollama")
    except ValueError as exc:
        return {"success": False, "error": str(exc)}
    if backend != "ollama":
        return {
            "success": False,
            "error": "Pull is only supported for Ollama. llama.cpp uses local GGUF files.",
        }
    if require_gpu_only():
        return {"success": False, "error": "GPU-only mode is enabled; refusing Ollama pull."}

    fit = fit_check(FitRequest(profile=req.profile, model_id=model_id, free_vram_mb=req.free_vram_mb))
    # Hard-block only when the model fits *nowhere* (not GPU VRAM, not system RAM).
    # A model that won't fit VRAM but runs on CPU (tier "cpu_only", severity "soft")
    # is allowed through with a note - blocking it would contradict the tiered fit
    # warnings. When fit-check couldn't even determine the model's size (severity
    # absent, e.g. an unparseable tag like "llama3:latest"), that's neither a hard
    # nor soft verdict - never block on it, but the pull must not go through silently.
    # Backward compatibility: older fit responses had only WONT_FIT. Treat that
    # explicit verdict as a hard block; genuinely unknown/unparseable responses
    # use UNKNOWN and remain advisory.
    severity = fit.get("severity") or ("hard" if fit.get("verdict") == "WONT_FIT" else "unknown")
    hard_block = severity == "hard"
    if hard_block and not req.allow_override:
        return {
            "success": False,
            "blocked_by": "fit-check",
            "fit": fit,
            "message": fit.get("headline")
            or "Model is unlikely to fit available VRAM or system RAM. Re-run with allow_override=true to pull anyway.",
        }

    try:
        destination = _ollama.base_url()
    except BackendCallError:
        destination = None

    def event_stream():
        start: Dict[str, Any] = {
            "status": f"starting pull for {model_id}",
            "fit_verdict": fit.get("verdict"),
            "model": model_id,
            # Where the model is being pulled to, so the UI can say so explicitly.
            "destination": destination,
            "destination_label": f"Ollama · {destination}" if destination else "Ollama",
        }
        # Surface the soft "won't fit GPU but runs on CPU" case so the pull isn't silent about it.
        if severity == "soft" and fit.get("cpu_deployable"):
            start["note"] = fit.get("headline") or "Won't fit GPU VRAM - will run on CPU (slower)."
        elif severity in (None, "unknown"):
            start["note"] = fit.get("message") or fit.get("headline") or "Could not verify VRAM fit for this model before pulling."
        yield _sse(start)
        try:
            for event in _ollama.pull_stream(model_id):
                yield _sse(event)
            # Pull succeeded: keep config.json in sync by ensuring a profile exists
            # for this model. Best-effort - never turn a good pull into a failure.
            profile_name, created, _err = ensure_profile_for_model(model_id)
            yield _sse(
                {
                    "status": "success",
                    "done": True,
                    "model": model_id,
                    "profile": profile_name,
                    "profile_created": created,
                }
            )
        except BackendCallError as exc:
            yield _sse({"error": str(exc)})
        except requests.ConnectionError:
            yield _sse({"error": "Ollama is not running or is unreachable. Start Ollama and retry."})
        except requests.RequestException as exc:
            yield _sse({"error": f"Ollama pull failed: {exc}"})
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# --- Step 6: serve / stop / switch -------------------------------------------


class ServeRequest(BaseModel):
    model: Optional[str] = None
    profile: Optional[str] = None
    keep_alive: str = "60m"
    device: Optional[str] = None  # "auto" | "gpu" | "cpu"
    num_gpu: Optional[int] = None  # advanced override (layers offloaded to GPU)


class StopRequest(BaseModel):
    model: Optional[str] = None
    profile: Optional[str] = None


class SwitchRequest(BaseModel):
    to_model: Optional[str] = None
    to_profile: Optional[str] = None
    from_model: Optional[str] = None
    keep_alive: str = "60m"
    device: Optional[str] = None
    num_gpu: Optional[int] = None


def _serve_ollama(model_id: str, keep_alive: str, num_gpu: Optional[int] = None) -> Dict[str, Any]:
    if require_gpu_only():
        return {"success": False, "error": "GPU-only mode is enabled; refusing Ollama serve."}
    target = _target_label(num_gpu)
    if num_gpu is not None:
        try:
            _ollama.unload_model(model_id)
        except BackendCallError as exc:
            return {"success": False, "error": f"Could not unload '{model_id}' before switching to {target}: {exc}"}
        except requests.ConnectionError:
            return {"success": False, "error": "Ollama is not running or is unreachable. Start Ollama and retry."}
        except requests.Timeout:
            return {
                "success": False,
                "timeout": True,
                "error": f"Timed out unloading '{model_id}' before switching to {target}. Wait a moment, then retry.",
            }
        except requests.RequestException as exc:
            return {"success": False, "error": f"Failed to unload '{model_id}' before switching to {target}: {exc}"}
    try:
        _ollama.load_model(model_id, keep_alive, num_gpu=num_gpu)
    except BackendCallError as exc:
        return {"success": False, "error": str(exc)}
    except requests.ConnectionError:
        return {"success": False, "error": "Ollama is not running or is unreachable. Start Ollama and retry."}
    except requests.Timeout:
        # The load may still finish in the background - guide the user to re-check
        # rather than implying a hard failure (common with large CPU loads).
        return {
            "success": False,
            "timeout": True,
            "error": (
                f"Loading '{model_id}' took too long to respond. Large models "
                "(especially on CPU) can take several minutes - it may still be "
                "loading. Wait a moment, then click Refresh status."
            ),
        }
    except requests.RequestException as exc:
        return {"success": False, "error": f"Failed to load '{model_id}': {exc}"}
    running, _ = _ollama.list_running()
    for m in running:
        m.update(_placement(m.get("size"), m.get("size_vram")))
    # If the device request couldn't be honored (e.g. a model too big to fully
    # fit GPU lands on Split), warn but proceed: the run is still useful and is
    # labeled with the *actual* placement, so nothing is mislabeled. A hard
    # failure here just produced confusing "status failed" rows for a reasonable
    # device choice.
    expected = _expected_placement(num_gpu)
    warning = None
    match = next((m for m in running if _matches_model_name(m.get("name"), model_id)), None)
    if expected is not None:
        actual = match.get("placement") if match else None
        if actual and actual != expected:
            warning = (
                f"Requested {expected}, but Ollama placed '{model_id}' on {actual} "
                f"(it may not fully fit {expected}). Results are labeled with the actual placement."
            )
    result = {
        "success": True,
        "backend": "ollama",
        "served": model_id,
        "running": running,
        "device": target,
        "message": f"'{model_id}' reloaded on {target} and kept alive for {keep_alive}."
        if num_gpu is not None
        else f"'{model_id}' warmed on {target} and kept alive for {keep_alive}.",
    }
    if warning:
        result["warning"] = warning
    result["calibration_sample"] = _record_calibration_sample(model_id, match)
    monitor.note_serve(model_id, target if target in ("GPU", "CPU") else None)
    return result


def _record_calibration_sample(model_id: str, running_match: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Best-effort: compare this warm-up's actual VRAM footprint (Ollama's
    `size_vram` from /api/ps) against the pre-load fit estimate, and record
    the pair for calibration.py. Never raises - a warm-up must still succeed
    even if calibration bookkeeping fails.
    """
    if not running_match:
        return None
    size_vram = running_match.get("size_vram")
    if not isinstance(size_vram, int) or size_vram <= 0:
        return None
    try:
        estimate = fit_check(FitRequest(model_id=model_id))
        estimated_gb = (estimate.get("estimate_gb") or {}).get("required")
        if not estimate.get("success") or not isinstance(estimated_gb, (int, float)) or estimated_gb <= 0:
            return None
        observed_gb = size_vram / 1_000_000_000.0
        sample = calibration.record_sample(
            gpu=calibration.gpu_key(),
            runtime="ollama",
            family=_base_family(model_id),
            quant=_parse_quant(model_id),
            context=4096,  # matches fit_check's default context when unspecified
            estimated_gb=estimated_gb,
            observed_gb=observed_gb,
        )
        sample["estimated_gb"] = round(estimated_gb, 3)
        sample["observed_gb"] = round(observed_gb, 3)
        return sample
    except Exception as exc:  # calibration is a bonus signal, never a hard dependency
        return {"error": str(exc)}


def _llamacpp_status_message() -> Dict[str, Any]:
    health = llama_health(os.getenv("LLAMACPP_BASE_URL", "http://127.0.0.1:8080"))
    if health.get("reachable"):
        return {"success": True, "backend": "llamacpp", "message": "llama.cpp server is running and serving its GGUF model."}
    return {
        "success": False,
        "backend": "llamacpp",
        "message": "Start the llama.cpp server first (scripts/start_llamacpp.ps1). LocalDeploy does not spawn it automatically.",
    }


@router.post("/models/serve")
def models_serve(req: ServeRequest) -> Dict[str, Any]:
    try:
        backend, model_id, _ = _resolve_target(req.profile, req.model, "ollama")
    except ValueError as exc:
        return {"success": False, "error": str(exc)}
    if backend == "ollama":
        return _serve_ollama(model_id, req.keep_alive, _resolve_num_gpu(req.device, req.num_gpu))
    return _llamacpp_status_message()


@router.post("/models/stop")
def models_stop(req: StopRequest) -> Dict[str, Any]:
    try:
        backend, model_id, _ = _resolve_target(req.profile, req.model, "ollama")
    except ValueError as exc:
        return {"success": False, "error": str(exc)}
    if backend != "ollama":
        return {
            "success": False,
            "backend": "llamacpp",
            "message": "Stop the llama.cpp server with scripts/stop.ps1; LocalDeploy does not manage that process.",
        }
    try:
        _ollama.unload_model(model_id)
    except BackendCallError as exc:
        return {"success": False, "error": str(exc)}
    except requests.ConnectionError:
        return {"success": False, "error": "Ollama is not running or is unreachable."}
    except requests.RequestException as exc:
        return {"success": False, "error": f"Failed to stop '{model_id}': {exc}"}

    # keep_alive=0 is an unload request, but Ollama's process inventory can lag
    # behind the generate response. Confirm it disappeared before claiming the
    # memory was released. A pending response lets the UI keep polling instead
    # of flipping back to Deploy prematurely.
    inventory_error = None
    for attempt in range(8):
        running, inventory_error = _ollama.list_running()
        if inventory_error:
            break
        if not any(_matches_model_name(item.get("name"), model_id) for item in running):
            session = monitor.note_stop(model_id)
            return {
                "success": True,
                "backend": "ollama",
                "status": "unloaded",
                "confirmed": True,
                "stopped": model_id,
                "message": f"'{model_id}' unloaded.",
                "session_summary": session,
            }
        if attempt < 7:
            time.sleep(0.2)
    return {
        "success": True,
        "backend": "ollama",
        "status": "pending",
        "confirmed": False,
        "stopped": model_id,
        "message": (
            f"Unload requested for '{model_id}', but running status could not be verified: {inventory_error}"
            if inventory_error
            else f"Unload requested for '{model_id}'; Ollama still reports it as loaded."
        ),
    }


class DeleteRequest(BaseModel):
    model: Optional[str] = None
    profile: Optional[str] = None


@router.post("/models/delete")
def models_delete(req: DeleteRequest) -> Dict[str, Any]:
    """Delete a model from disk (frees disk space). Ollama only."""
    try:
        backend, model_id, _ = _resolve_target(req.profile, req.model, "ollama")
    except ValueError as exc:
        return {"success": False, "error": str(exc)}
    if backend != "ollama":
        return {
            "success": False,
            "backend": "llamacpp",
            "message": "Delete is Ollama-only. Remove llama.cpp GGUF files manually.",
        }
    try:
        _ollama.delete_model(model_id)
    except BackendCallError as exc:
        return {"success": False, "error": str(exc)}
    except requests.ConnectionError:
        return {"success": False, "error": "Ollama is not running or is unreachable."}
    except requests.RequestException as exc:
        return {"success": False, "error": f"Failed to delete '{model_id}': {exc}"}
    return {"success": True, "deleted": model_id, "message": f"'{model_id}' deleted from disk."}


@router.post("/models/free")
def models_free() -> Dict[str, Any]:
    """Unload all loaded models from memory/VRAM (the 'free memory' reset): Ollama's, and the llama.cpp server LocalDeploy manages.

    A person freeing the GPU for something else (a game) means every model LocalDeploy has loaded. The llama.cpp server was left running
    because only Ollama models were unloaded here; it starts again on the next request that needs it. When Ollama cannot be reached but a
    llama.cpp server was stopped, memory was freed and that is a success.
    """
    running_before, _ = _ollama.list_running()
    llamacpp = llamacpp_managed.stop_all()
    try:
        count, err = _ollama.unload_all()
    except BackendCallError as exc:
        count, err = 0, str(exc)
    if err is not None and llamacpp is None:
        return {"success": False, "error": err}
    if err is not None:
        count = 0
    for m in running_before:
        name = m.get("name")
        if name:
            monitor.note_stop(name)
    unloaded = count + (1 if llamacpp else 0)
    return {
        "success": True,
        "unloaded": unloaded,
        "llamacpp": llamacpp,
        "message": f"Freed memory: unloaded {unloaded} model(s)." if unloaded else "No models were loaded.",
    }


@router.post("/models/switch")
def models_switch(req: SwitchRequest) -> Dict[str, Any]:
    try:
        backend, to_model, _ = _resolve_target(req.to_profile, req.to_model, "ollama")
    except ValueError as exc:
        return {"success": False, "error": str(exc)}
    if backend != "ollama":
        return _llamacpp_status_message()

    unloaded = None
    if req.from_model:
        try:
            _ollama.unload_model(req.from_model)
            unloaded = req.from_model
            monitor.note_stop(req.from_model)
        except (BackendCallError, requests.RequestException):
            unloaded = None  # best-effort; serving the new model is what matters

    result = _serve_ollama(to_model, req.keep_alive, _resolve_num_gpu(req.device, req.num_gpu))
    result["switched_from"] = unloaded
    if result.get("success"):
        result["message"] = f"Switched to '{to_model}'" + (f" (unloaded '{unloaded}')" if unloaded else "") + "."
    return result
