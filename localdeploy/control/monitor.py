"""Deployment monitoring (Release R3).

GET /system/monitor answers "what is happening after a model has been
loaded and while it is serving requests?" - live placement, VRAM/RAM/CPU,
per-model request stats, rolling history for charts, and rule-based alerts.

No background sampler thread: each call to this endpoint samples hardware
fresh (the response needs a fresh reading anyway) and appends it to a bounded
in-memory ring buffer, so a few minutes of client-side polling - while the
Monitor tab is open - builds enough history for rolling charts and
sustained-threshold alerts ("VRAM >95% for 3 minutes"). History does not
accumulate while nobody is watching; an independent background sampler would
close that gap but isn't needed for a first cut.

Privacy: everything recorded here is numeric metadata (token counts,
latencies, placement) - prompts and responses are never stored, matching
LocalDeploy's no-telemetry stance.
"""
from __future__ import annotations

import statistics
import threading
import time
from collections import deque
from itertools import count
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set

from fastapi import APIRouter

router = APIRouter()

_HW_HISTORY_MAXLEN = 720  # ~1 hour at a 5s client poll cadence
_REQUEST_HISTORY_MAXLEN = 500
_SUSTAINED_VRAM_ALERT_PCT = 95.0
_SUSTAINED_VRAM_ALERT_SECONDS = 180  # 3 minutes
_SLOW_GENERATION_ALERT_RATIO = 0.72  # current tok/s below 72% of this model's own median

_hw_history: Deque[Dict[str, Any]] = deque(maxlen=_HW_HISTORY_MAXLEN)
_request_history: Deque[Dict[str, Any]] = deque(maxlen=_REQUEST_HISTORY_MAXLEN)
# model_id -> {"since": epoch_seconds, "requested_device": "GPU"|"CPU"|"auto"|None}
_serve_state: Dict[str, Dict[str, Any]] = {}
# model_id -> unique IDs for backend calls that have started but not finished.
_active_requests: Dict[str, Set[int]] = {}
_request_ids = count(1)
_state_lock = threading.RLock()


def reset_state() -> None:
    """Test-only: clear all in-memory monitoring state."""
    with _state_lock:
        _hw_history.clear()
        _request_history.clear()
        _serve_state.clear()
        _active_requests.clear()


def note_serve(model_id: str, requested_device: Optional[str]) -> None:
    """Record that a model just (re)started serving - resets its uptime clock."""
    with _state_lock:
        _serve_state[model_id] = {"since": time.time(), "requested_device": requested_device}


def _model_names_match(left: str, right: str) -> bool:
    if str(left).casefold() == str(right).casefold():
        return True
    try:
        from .models import _matches_model_name

        return _matches_model_name(left, right)
    except Exception:
        return False


def _matching_key(mapping: Dict[str, Any], model_id: str) -> Optional[str]:
    if model_id in mapping:
        return model_id
    return next((key for key in mapping if _model_names_match(model_id, key)), None)


def note_stop(model_id: str) -> Optional[Dict[str, Any]]:
    """Pop this model's serve state and persist a session summary.

    Returns the summary, or None if there was no tracked serve state (e.g. it
    was already stopped, or the server restarted since it was loaded).
    """
    with _state_lock:
        tracked_name = _matching_key(_serve_state, model_id)
        state = _serve_state.pop(tracked_name, None) if tracked_name else None
    if state is None:
        return None
    summary = _session_summary(tracked_name or model_id, state)
    _persist_session(summary)
    return summary


def note_request_start(model_id: Optional[str]) -> Optional[int]:
    """Track a backend call as in flight for truthful concurrency reporting."""
    if not model_id:
        return None
    with _state_lock:
        key = _matching_key(_active_requests, model_id) or model_id
        request_id = next(_request_ids)
        _active_requests.setdefault(key, set()).add(request_id)
        return request_id


def note_request_end(model_id: Optional[str], request_id: Optional[int] = None) -> None:
    """Finish one in-flight backend call; unmatched/duplicate finishes are safe."""
    if not model_id:
        return
    with _state_lock:
        key = _matching_key(_active_requests, model_id)
        if key is None:
            return
        active = _active_requests[key]
        if request_id is None:
            if active:
                active.pop()
        else:
            active.discard(request_id)
        if not active:
            _active_requests.pop(key, None)


def record_request(
    *,
    profile: Optional[str],
    model: Optional[str],
    backend: Optional[str],
    kind: str,
    success: bool,
    elapsed_seconds: float,
    metrics: Optional[Dict[str, Any]],
    context_limit: Optional[int],
    error: Optional[str] = None,
) -> None:
    """Numeric-only request record for the Monitor request log and alerts."""
    metrics = metrics or {}
    with _state_lock:
        _request_history.append({
            "ts": time.time(),
            "profile": profile,
            "model": model,
            "backend": backend,
            "source": kind,  # "chat" | "vision" | "benchmark" | ...
            "success": success,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "prompt_tokens": metrics.get("prompt_eval_count"),
            "output_tokens": metrics.get("eval_count"),
            "ttft_ms": metrics.get("ttft_ms"),
            "tokens_per_second": metrics.get("tokens_per_second"),
            "context_limit": context_limit,
            "error": (str(error)[:200] or None) if error else None,
        })


def _primary_gpus(hw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """GPUs in the largest compatible pool, the same pool the fit checks use.

    Adding an integrated GPU's small memory to a discrete card's would make the
    Monitor disagree with the System card (8.5 GB against 8.0 GB on a laptop
    with an RTX 3080 and a Radeon iGPU).
    """
    gpus = hw.get("gpus") or []
    groups = (hw.get("gpu_summary") or {}).get("compatible_groups") or []
    if groups:
        indexes = set(groups[0].get("gpu_indexes") or [])
        picked = [gpu for index, gpu in enumerate(gpus) if index in indexes]
        if picked:
            return picked
    return gpus


def _sample_hardware(hw: Dict[str, Any]) -> Dict[str, Any]:
    gpus = _primary_gpus(hw)
    vram_total = sum(g.get("vram_total_mb") or 0 for g in gpus) or None
    vram_used = sum(g.get("vram_used_mb") or 0 for g in gpus) or None
    utils = [g.get("utilization_pct") for g in gpus if g.get("utilization_pct") is not None]
    system = hw.get("system") or {}
    ram_total = system.get("ram_total_mb")
    ram_available = system.get("ram_available_mb")
    ram_used = (ram_total - ram_available) if ram_total is not None and ram_available is not None else None
    sample = {
        "ts": time.time(),
        "vram_used_mb": vram_used,
        "vram_total_mb": vram_total,
        "vram_pct": round(vram_used / vram_total * 100, 1) if vram_used is not None and vram_total else None,
        "gpu_utilization_pct": round(statistics.mean(utils), 1) if utils else None,
        "ram_used_mb": ram_used,
        "ram_total_mb": ram_total,
        "cpu_percent": system.get("cpu_percent"),
    }
    with _state_lock:
        _hw_history.append(sample)
    return sample


def _alerts(current: Dict[str, Any], model_cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []
    now = time.time()

    # Sustained high VRAM: every sample in the trailing window must be over threshold.
    window_start = now - _SUSTAINED_VRAM_ALERT_SECONDS
    with _state_lock:
        hardware_history = list(_hw_history)
    window = [h for h in hardware_history if h["ts"] >= window_start and h.get("vram_pct") is not None]
    if window and len(window) >= 2 and all(h["vram_pct"] >= _SUSTAINED_VRAM_ALERT_PCT for h in window):
        span = now - window[0]["ts"]
        if span >= _SUSTAINED_VRAM_ALERT_SECONDS * 0.8:  # enough coverage to trust the window
            alerts.append(
                {
                    "level": "warning",
                    "text": f"VRAM usage has remained above {_SUSTAINED_VRAM_ALERT_PCT:.0f}% for "
                    f"{int(span // 60)} minute(s).",
                }
            )

    for card in model_cards:
        # Requested a specific device but Ollama placed it elsewhere.
        requested = card.get("requested_device")
        actual = card.get("placement")
        if requested in ("GPU", "CPU") and actual and actual.upper() != requested.upper():
            alerts.append(
                {
                    "level": "warning",
                    "text": f"'{card['name']}' is using {actual} even though {requested}-only placement was requested.",
                }
            )
        # Generation speed well below this model's own recent median.
        median_tps = card.get("median_tokens_per_second")
        recent_tps = card.get("recent_tokens_per_second")
        if median_tps and recent_tps and recent_tps < median_tps * _SLOW_GENERATION_ALERT_RATIO:
            drop_pct = round((1 - recent_tps / median_tps) * 100)
            alerts.append(
                {
                    "level": "info",
                    "text": f"'{card['name']}' generation speed is {drop_pct}% below its recent median "
                    f"({recent_tps} vs ~{median_tps} tok/s).",
                }
            )
        if card.get("active_requests", 0) >= 2:
            alerts.append(
                {
                    "level": "info",
                    "text": f"{card['active_requests']} simultaneous requests to '{card['name']}' may exceed "
                    "available KV-cache headroom.",
                }
            )
    return alerts


def _model_requests(model_id: str, since: Optional[float] = None) -> List[Dict[str, Any]]:
    with _state_lock:
        requests = list(_request_history)
    return [r for r in requests if _model_names_match(str(r.get("model") or ""), model_id)
            and (since is None or r["ts"] >= since)]


def _find_serve_state(name: str) -> Optional[Dict[str, Any]]:
    """Look up `_serve_state` by the name Ollama reports for a running model.

    Exact match covers the common case (profiles always carry a fully
    qualified model_id), but a bare/tagless serve request can leave the
    two spellings slightly different (e.g. 'llama3' vs Ollama's reported
    'llama3:latest') - fall back to the same fuzzy matcher models.py uses
    everywhere else for this instead of silently showing blank uptime.
    """
    with _state_lock:
        key = _matching_key(_serve_state, name)
        return dict(_serve_state[key]) if key is not None else None


def _active_request_count(name: str) -> int:
    with _state_lock:
        return sum(len(request_ids) for model_id, request_ids in _active_requests.items()
                   if _model_names_match(name, model_id))


def _model_card(running: Dict[str, Any]) -> Dict[str, Any]:
    name = str(running.get("name") or "")
    state = _find_serve_state(name)
    reqs = _model_requests(name, since=state["since"] if state else None)
    recent_reqs = [r for r in reqs if time.time() - r["ts"] <= 120]
    ok = [r for r in reqs if r["success"]]
    recent_ok = [r for r in recent_reqs if r["success"]]
    tps_all = [r["tokens_per_second"] for r in ok if r.get("tokens_per_second") is not None]
    tps_recent = [r["tokens_per_second"] for r in recent_ok if r.get("tokens_per_second") is not None]
    ttft_all = [r["ttft_ms"] for r in ok if r.get("ttft_ms") is not None]
    return {
        "name": name,
        "placement": running.get("placement"),
        "gpu_percent": running.get("gpu_percent"),
        "size_mb": running.get("size_mb"),
        "size_vram_mb": running.get("size_vram_mb"),
        "expires_at": running.get("expires_at"),
        "requested_device": state.get("requested_device") if state else None,
        "uptime_seconds": round(time.time() - state["since"], 1) if state else None,
        "request_count": len(reqs),
        "active_requests": _active_request_count(name),
        "failure_count": len(reqs) - len(ok),
        "median_tokens_per_second": round(statistics.median(tps_all), 2) if tps_all else None,
        "recent_tokens_per_second": round(statistics.median(tps_recent), 2) if tps_recent else None,
        "median_ttft_ms": round(statistics.median(ttft_all), 1) if ttft_all else None,
    }


def _session_summary(model_id: str, state: Dict[str, Any]) -> Dict[str, Any]:
    since = state["since"]
    reqs = _model_requests(model_id, since=since)
    ok = [r for r in reqs if r["success"]]
    tps = [r["tokens_per_second"] for r in ok if r.get("tokens_per_second") is not None]
    ttft = [r["ttft_ms"] for r in ok if r.get("ttft_ms") is not None]
    with _state_lock:
        hardware_history = list(_hw_history)
    hw_samples = [h for h in hardware_history if h["ts"] >= since and h.get("vram_used_mb") is not None]
    peak_vram_mb = max((h["vram_used_mb"] for h in hw_samples), default=None)
    from .. import __version__

    return {
        "schema_version": 1,
        "model": model_id,
        "requested_device": state.get("requested_device"),
        "started_at": since,
        "ended_at": time.time(),
        "uptime_seconds": round(time.time() - since, 1),
        "request_count": len(reqs),
        "failure_count": len(reqs) - len(ok),
        "peak_vram_mb": peak_vram_mb,
        "median_tokens_per_second": round(statistics.median(tps), 2) if tps else None,
        "median_ttft_ms": round(statistics.median(ttft), 1) if ttft else None,
        "localdeploy_version": __version__,
    }


def _sessions_dir() -> Path:
    from ..utils import app_home

    return app_home() / "reports" / "monitor-sessions"


def _persist_session(summary: Dict[str, Any]) -> Optional[str]:
    """Best-effort JSON write, mirroring history.py's benchmark-history pattern.
    Never raises - losing a session summary must not break a model stop."""
    try:
        import json
        import re

        directory = _sessions_dir()
        directory.mkdir(parents=True, exist_ok=True)
        safe_model = re.sub(r"[^A-Za-z0-9_.-]", "_", summary.get("model") or "model")
        path = directory / f"{safe_model}-{int(summary['ended_at'] * 1000)}.json"
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(path)
    except Exception:
        return None


@router.get("/system/monitor")
def system_monitor() -> Dict[str, Any]:
    from . import _ollama
    from .hardware import detect_hardware
    from .models import _placement

    hw = detect_hardware()
    hw_sample = _sample_hardware(hw)

    running, run_error = _ollama.list_running()
    for m in running:
        m.update(_placement(m.get("size"), m.get("size_vram")))
        if isinstance(m.get("size"), int):
            m["size_mb"] = round(m["size"] / 1_000_000)
        if isinstance(m.get("size_vram"), int):
            m["size_vram_mb"] = round(m["size_vram"] / 1_000_000)

    model_cards = [_model_card(m) for m in running]
    alerts = _alerts(hw_sample, model_cards)

    with _state_lock:
        recent_requests = list(_request_history)[-50:]
        hardware_history = list(_hw_history)[-180:]
    recent_requests.reverse()  # newest first, matching the roadmap's "recent requests" framing

    return {
        "success": True,
        "ollama_reachable": run_error is None,
        "hardware": hw_sample,
        "history": {
            "hardware": hardware_history,  # last ~15 minutes at 5s cadence
        },
        "models": model_cards,
        "requests": recent_requests,
        "alerts": alerts,
        "note": "Active requests are backend calls that have started but have not yet finished.",
    }
