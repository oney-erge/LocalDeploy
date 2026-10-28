"""Automated model bakeoff (Release R6) - "Compare top models for me."

POST /system/bakeoff/run streams the strongest end-to-end LocalDeploy workflow
as one operation: pick fit-safe candidates for a workload within a download
budget, pull + serve + benchmark each one in turn (sequentially, to avoid
VRAM contention - same discipline as /benchmark/run), rank them with the
existing recommend.py scoring formula, then re-serve the winner.

This is orchestration only. Every step reuses an existing engine:
  - candidate selection: starter.py's fit-filtered, workload-biased catalog
  - pull: _ollama.pull_stream (same as /models/pull)
  - serve + placement: models._serve_ollama / models._placement
  - benchmark: benchmark.execute_test + bench.py's BENCHMARK_PACKS
  - ranking: recommend.py's rank_candidates (quality x speed x headroom)
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import _ollama
from ._config import ensure_profile_for_model
from .bench import BENCHMARK_PACKS
from .bench import _bench  # memoized, race-safe import of the top-level `benchmark` module
from .fit import _weight_gb_per_b
from .models import _matches_model_name, _serve_ollama
from .recommend import rank_candidates
from .starter import _fitting_candidates, _rank, _resolve_budget, _with_workload_bias

router = APIRouter()

# use_case (from the guided-recommend vocabulary) -> default benchmark pack.
_USE_CASE_PACK = {
    "coding": "coding",
    "structured_extraction": "structured",
    "reasoning": "reasoning",
    "tool_calling": "structured",
}
_DEFAULT_PACK = "general"


class BakeoffRequest(BaseModel):
    use_case: Optional[str] = None
    priority: str = Field(default="balanced")
    expected_context: int = Field(default=8192, gt=0, le=1_000_000)
    download_budget_gb: float = Field(default=10.0, gt=0, le=1_000)
    pack: Optional[str] = None
    max_candidates: int = Field(default=3, ge=1, le=5)
    sample_size: int = Field(default=3, ge=1, le=10)
    timeout: int = Field(default=120, ge=1, le=3_600)
    free_vram_mb: Optional[int] = Field(default=None, ge=0, le=100_000_000)
    margin_gb: float = Field(default=2.0, ge=0, le=1_024)


def _select_candidates(
    use_case: Optional[str], priority: str, budget_gb: float, download_budget_gb: float,
    max_candidates: int, expected_context: int = 4096,
) -> List[Dict[str, Any]]:
    """Fit-safe, workload-biased, download-budget-capped picks - the same
    catalog and ranking recommend.py's guided-recommend endpoint uses, just
    greedily packed against a cumulative download-size cap instead of split
    into three labeled buckets."""
    fitting = _fitting_candidates(budget_gb, context=expected_context)
    biased = _with_workload_bias(fitting, use_case)
    ranked = _rank(biased, budget_gb, priority)

    picked: List[Dict[str, Any]] = []
    cumulative_download_gb = 0.0
    for entry in ranked:
        download_gb = round(entry["params_b"] * _weight_gb_per_b("q4"), 1)
        if cumulative_download_gb + download_gb > download_budget_gb:
            continue
        entry = dict(entry)
        entry["download_gb"] = download_gb
        picked.append(entry)
        cumulative_download_gb += download_gb
        if len(picked) >= max_candidates:
            break
    return picked


@router.post("/system/bakeoff/run")
def bakeoff_run(req: BakeoffRequest):
    def sse(event: Dict[str, Any]) -> str:
        return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    def event_stream():
        raw_budget_gb, budget_gb, source, hw = _resolve_budget(req.free_vram_mb, req.margin_gb)
        if budget_gb is None:
            yield sse({"event": "error", "error": "Could not determine VRAM or system RAM budget."})
            yield "data: [DONE]\n\n"
            return

        candidates = _select_candidates(
            req.use_case, req.priority, budget_gb, req.download_budget_gb,
            req.max_candidates, expected_context=req.expected_context,
        )
        if not candidates:
            yield sse({"event": "error", "error": "No catalog model fits your budget and download cap."})
            yield "data: [DONE]\n\n"
            return

        bench = _bench()
        pack_id = req.pack or _USE_CASE_PACK.get(req.use_case or "", _DEFAULT_PACK)
        pack_categories = (BENCHMARK_PACKS.get(pack_id) or BENCHMARK_PACKS[_DEFAULT_PACK])["categories"]
        tests = [t for t in bench.builtin_test_cases() if t.category in pack_categories]
        tests = sorted(tests, key=lambda t: t.max_output_tokens)[: max(1, req.sample_size)]
        base_url = bench.api_base_url()

        yield sse(
            {
                "event": "bakeoff_start",
                "candidates": [c["id"] for c in candidates],
                "pack": pack_id,
                "sample_tests": [t.name for t in tests],
                "budget_source": source,
                "budget_gb": round(budget_gb, 2),
                "download_budget_gb": req.download_budget_gb,
            }
        )

        scored: List[Dict[str, Any]] = []
        installed, _ = _ollama.list_installed()
        installed_names = {str(m.get("name")) for m in installed if m.get("name")}

        for candidate in candidates:
            model_id = candidate["id"]
            yield sse({"event": "candidate_start", "model": model_id, "download_gb": candidate["download_gb"]})

            if not any(_matches_model_name(name, model_id) for name in installed_names):
                yield sse({"event": "pull_start", "model": model_id})
                try:
                    for progress in _ollama.pull_stream(model_id):
                        yield sse({"event": "pull_progress", "model": model_id, **progress})
                except Exception as exc:
                    yield sse({"event": "candidate_failed", "model": model_id, "reason": f"pull failed: {exc}"})
                    scored.append({"profile": model_id, "avg_accuracy": 0.0, "avg_latency_s": 0.0, "passed": 0, "tests": 0, "margin_gb": None})
                    continue
                yield sse({"event": "pull_end", "model": model_id})

            profile_name, _created, profile_err = ensure_profile_for_model(model_id)
            if not profile_name:
                yield sse({"event": "candidate_failed", "model": model_id, "reason": profile_err or "could not create a profile"})
                scored.append({"profile": model_id, "avg_accuracy": 0.0, "avg_latency_s": 0.0, "passed": 0, "tests": 0, "margin_gb": None})
                continue

            yield sse({"event": "deploy_start", "model": model_id})
            served = _serve_ollama(model_id, "10m")
            if not served.get("success"):
                yield sse({"event": "candidate_failed", "model": model_id, "reason": served.get("error") or "deploy failed"})
                scored.append({"profile": profile_name, "avg_accuracy": 0.0, "avg_latency_s": 0.0, "passed": 0, "tests": 0, "margin_gb": candidate.get("margin_gb")})
                continue
            running = served.get("running") or []
            match = next((m for m in running if _matches_model_name(m.get("name"), model_id)), None)
            placement = match.get("placement") if match else None
            yield sse({"event": "deploy_end", "model": model_id, "placement": placement, "warning": served.get("warning")})

            from api_server import load_config  # lazy: api_server owns config loading

            profile = load_config().get("profiles", {}).get(profile_name, {})
            results = []
            for t in tests:
                yield sse({"event": "test_start", "model": model_id, "name": t.name})
                try:
                    item = bench.execute_test(base_url, profile_name, profile, t, req.timeout)
                except Exception:
                    item = bench.TestResult(
                        name=t.name, category=t.category, success=False, elapsed_seconds=0.0,
                        response_length=0, response_preview="", accuracy=0.0, error="execute_test error",
                    )
                results.append(item)
                yield sse({"event": "test_result", "model": model_id, "name": item.name, "success": item.success, "accuracy": item.accuracy, "elapsed_seconds": item.elapsed_seconds})

            accs = [r.accuracy for r in results]
            lats = [r.elapsed_seconds for r in results if r.elapsed_seconds]
            row = {
                "profile": model_id,
                "avg_accuracy": round(sum(accs) / len(accs), 3) if accs else 0.0,
                "avg_latency_s": round(sum(lats) / len(lats), 3) if lats else 0.0,
                "passed": sum(1 for r in results if r.success),
                "tests": len(results),
                "margin_gb": candidate.get("margin_gb"),
                "placement": placement,
            }
            scored.append(row)
            yield sse({"event": "candidate_end", "model": model_id, **row})

            # Sequential VRAM discipline (same as /benchmark/run): unload before
            # the next candidate. The overall winner gets explicitly re-served below.
            try:
                _ollama.unload_model(model_id)
            except Exception:
                pass

        ranked = rank_candidates([s for s in scored if s["tests"] > 0]) if any(s["tests"] > 0 for s in scored) else []
        if not ranked:
            yield sse({"event": "error", "error": "Every candidate failed to pull, deploy, or benchmark."})
            yield "data: [DONE]\n\n"
            return

        winner_model = ranked[0]["profile"]
        yield sse({"event": "deploy_start", "model": winner_model, "final": True})
        winner_served = _serve_ollama(winner_model, "60m")
        yield sse(
            {
                "event": "bakeoff_end",
                "pack": pack_id,
                "ranked": ranked,
                "winner": winner_model,
                "losers": [s["profile"] for s in ranked[1:]],
                "winner_deployed": bool(winner_served.get("success")),
            }
        )
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
