"""Step 15 - "Starter pack" - one-click curated model picks for new users.

POST /registry/starter-pack takes the detected free VRAM (or system RAM when
there is no GPU), subtracts a safety margin, and returns the best-fitting
handful of well-known Ollama models from a small curated catalog. This is
deliberately a static catalog rather than a live Hugging Face search - a
brand-new user's first pull should be something known-good, not whatever is
newest/most-downloaded this week (that's what /registry/check-updates is for).

Each candidate's VRAM estimate reuses the same weights+KV+overhead formula as
fit.py, so the numbers agree with the rest of the app.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter
from pydantic import BaseModel, Field

from .fit import _KV_MB_PER_TOKEN_PER_B, _OVERHEAD_GB, _kv_quant_factor, _weight_gb_per_b
from .hardware import detect_hardware

router = APIRouter()

# A small, curated catalog of well-known dense instruct models spanning
# ~0.5B -> 72B. Deliberately short so recommendations stay trustworthy. `tier`
# is a hand-curated 1-5 quality rating (5 = best-in-class for its size class),
# informed by this repo's own benchmark results (docs/MODELS.md) plus general
# community consensus for sizes not benchmarked here. `params_b` is set
# explicitly (not parsed from the tag) so MoE / odd naming can't throw the
# VRAM estimate off; every entry below is a dense model.
# Guided-recommend workload tags (Release R1). Matches the "Use case" options
# in the Recommended-models UI. Hand-curated from each family's public model
# card/announcement, not measured on any specific machine - treated as
# "published" provenance in /registry/recommend's explainability, same as
# `context_native` below.
WORKLOAD_TAGS = (
    "general", "coding", "document_analysis", "structured_extraction",
    "reasoning", "vision", "multilingual", "tool_calling",
)

# Published native context length (tokens), hand-curated from each family's
# model card. Deliberately rounded to a coarse tier (32768, 131072, or 262144) rather
# than an exact figure this repo can't verify per release - "expected
# context" filtering only needs to know roughly which bucket a model reaches.
STARTER_CATALOG: List[Dict[str, Any]] = [
    {"id": "qwen2.5:0.5b", "family": "qwen2.5", "params_b": 0.5, "tier": 2, "vision": False,
     "use_case": "tiny/edge", "workload_tags": ["general"], "context_native": 32768,
     "description": "Smallest usable Qwen2.5 instruct model. Fits almost anywhere, but expect basic quality."},
    {"id": "qwen2.5:1.5b", "family": "qwen2.5", "params_b": 1.5, "tier": 3, "vision": False,
     "use_case": "tiny/edge", "workload_tags": ["general"], "context_native": 32768,
     "description": "Small Qwen2.5 instruct model with noticeably better quality than the 0.5B."},
    {"id": "llama3.2:1b", "family": "llama3.2", "params_b": 1.0, "tier": 3, "vision": False,
     "use_case": "tiny/edge", "workload_tags": ["general"], "context_native": 131072,
     "description": "Meta's smallest Llama 3.2 instruct model. Fast and lightweight for simple tasks."},
    {"id": "gemma3:1b", "family": "gemma3", "params_b": 1.0, "tier": 3, "vision": False,
     "use_case": "tiny/edge", "workload_tags": ["general"], "context_native": 32768,
     "description": "Smallest Gemma 3 model, text-only. Good for very constrained hardware."},
    {"id": "qwen3.5:2b", "family": "qwen3.5", "params_b": 2.0, "tier": 4, "vision": True,
     "use_case": "tiny/vision", "workload_tags": ["general", "vision", "reasoning", "tool_calling", "multilingual"],
     "context_native": 262144,
     "description": "Compact multimodal Qwen 3.5 model with reasoning and tool support."},
    {"id": "qwen2.5:3b", "family": "qwen2.5", "params_b": 3.0, "tier": 3, "vision": False,
     "use_case": "small/general", "workload_tags": ["general"], "context_native": 32768,
     "description": "Solid small general-purpose Qwen2.5 model."},
    {"id": "llama3.2:3b", "family": "llama3.2", "params_b": 3.0, "tier": 4, "vision": False,
     "use_case": "small/fast", "workload_tags": ["general"], "context_native": 131072,
     "description": "Fastest profile in this repo's own benchmarks (~4s/test) with acceptable quality."},
    {"id": "phi4-mini", "family": "phi4", "params_b": 3.8, "tier": 3, "vision": False,
     "use_case": "small/general", "workload_tags": ["general", "reasoning"], "context_native": 131072,
     "description": "Compact Microsoft Phi-4 Mini instruct model."},
    {"id": "gemma3:4b", "family": "gemma3", "params_b": 4.0, "tier": 4, "vision": True,
     "use_case": "small/vision", "workload_tags": ["general", "vision"], "context_native": 131072,
     "description": "This repo's safe default profile - small, multimodal, dependable."},
    {"id": "qwen3-vl:4b-instruct", "family": "qwen3-vl", "params_b": 4.0, "tier": 4, "vision": True,
     "use_case": "small/vision", "workload_tags": ["vision", "general"], "context_native": 32768,
     "description": "Fast small vision-language model, ~2x faster than the 8B at somewhat lower quality."},
    {"id": "qwen2.5:7b", "family": "qwen2.5", "params_b": 7.0, "tier": 5, "vision": False,
     "use_case": "mid/general", "workload_tags": ["general", "multilingual"], "context_native": 32768,
     "description": "Fastest strong generalist measured in this repo (0.79 overall @ 5.4s/test)."},
    {"id": "qwen2.5-coder:7b", "family": "qwen2.5-coder", "params_b": 7.0, "tier": 4, "vision": False,
     "use_case": "mid/coding", "workload_tags": ["coding"], "context_native": 32768,
     "description": "Strongest small coding model measured here (0.93 on code tasks)."},
    {"id": "mistral:7b", "family": "mistral", "params_b": 7.0, "tier": 3, "vision": False,
     "use_case": "mid/general", "workload_tags": ["general"], "context_native": 32768,
     "description": "Solid general model, but weaker on math/classification than the Qwen alternatives at this size."},
    {"id": "deepseek-r1:7b", "family": "deepseek-r1", "params_b": 7.0, "tier": 3, "vision": False,
     "use_case": "mid/reasoning", "workload_tags": ["reasoning"], "context_native": 32768,
     "description": "Chain-of-thought reasoning model; best for math when 'think' is enabled."},
    {"id": "qwen3:8b", "family": "qwen3", "params_b": 8.0, "tier": 5, "vision": False,
     "use_case": "mid/structured", "workload_tags": ["structured_extraction", "tool_calling", "general"],
     "context_native": 32768,
     "description": "Structured-output champion measured here (0.91 hard-JSON); set think:false."},
    {"id": "qwen3-vl:8b-instruct", "family": "qwen3-vl", "params_b": 8.0, "tier": 5, "vision": True,
     "use_case": "mid/vision", "workload_tags": ["vision", "general", "document_analysis"],
     "context_native": 32768,
     "description": "Best overall all-rounder measured in this repo (0.84 overall, strong on every category)."},
    {"id": "llama3.1:8b", "family": "llama3.1", "params_b": 8.0, "tier": 4, "vision": False,
     "use_case": "mid/general", "workload_tags": ["structured_extraction", "general", "tool_calling"],
     "context_native": 131072,
     "description": "Strong sleeper pick for JSON pipelines (0.89 hard-JSON) despite a modest overall score."},
    {"id": "granite3.3:8b", "family": "granite3.3", "params_b": 8.0, "tier": 3, "vision": False,
     "use_case": "mid/general", "workload_tags": ["general", "document_analysis"], "context_native": 131072,
     "description": "IBM Granite 3.3 general instruct model."},
    {"id": "qwen3.5:9b", "family": "qwen3.5", "params_b": 9.0, "tier": 4, "vision": True,
     "use_case": "mid/vision-agent",
     "workload_tags": ["general", "vision", "reasoning", "coding", "tool_calling", "multilingual"],
     "context_native": 262144,
     "description": "Mid-sized multimodal Qwen 3.5 model for local reasoning, coding, and tool use."},
    {"id": "gemma3:12b-it-qat", "family": "gemma3", "params_b": 12.0, "tier": 4, "vision": True,
     "use_case": "large/vision", "workload_tags": ["vision", "general"], "context_native": 131072,
     "description": "Quantization-aware-trained Gemma 3 12B - beats the default Q4 tag at the same speed/VRAM."},
    {"id": "qwen3:14b", "family": "qwen3", "params_b": 14.0, "tier": 5, "vision": False,
     "use_case": "large/general", "workload_tags": ["general", "reasoning", "multilingual", "tool_calling"],
     "context_native": 32768,
     "description": "Larger dense Qwen3 generalist for hardware with headroom beyond the 8B class."},
    {"id": "gemma3:27b", "family": "gemma3", "params_b": 27.0, "tier": 4, "vision": True,
     "use_case": "xlarge/vision", "workload_tags": ["vision", "general", "document_analysis"],
     "context_native": 131072,
     "description": "Large multimodal Gemma 3 for high-VRAM cards."},
    {"id": "qwen3.6:27b", "family": "qwen3.6", "params_b": 27.0, "tier": 5, "vision": True,
     "use_case": "xlarge/vision-agent",
     "workload_tags": ["vision", "tool_calling", "document_analysis", "multilingual"],
     "context_native": 262144,
     "description": "Dense native-multimodal Qwen3.6 for high-VRAM visual reasoning and structured agent work."},
    {"id": "qwen3:32b", "family": "qwen3", "params_b": 32.0, "tier": 5, "vision": False,
     "use_case": "xlarge/general", "workload_tags": ["general", "reasoning", "multilingual", "tool_calling"],
     "context_native": 32768,
     "description": "Flagship dense Qwen3 size - strong all-around quality for 24GB+ cards."},
    {"id": "llama3.1:70b", "family": "llama3.1", "params_b": 70.0, "tier": 5, "vision": False,
     "use_case": "huge/flagship", "workload_tags": ["general", "reasoning", "tool_calling"],
     "context_native": 131072,
     "description": "Meta's flagship dense Llama 3.1 - needs a very high-VRAM card or multi-GPU."},
    {"id": "qwen2.5:72b", "family": "qwen2.5", "params_b": 72.0, "tier": 5, "vision": False,
     "use_case": "huge/flagship", "workload_tags": ["general", "multilingual"], "context_native": 32768,
     "description": "Flagship dense Qwen2.5 size - top-tier quality, needs a very high-VRAM card."},
]


def _required_gb(params_b: float, quant: Optional[str] = None, context: int = 4096) -> float:
    """Same estimator as fit.py: weights + KV cache + fixed overhead."""
    weights_gb = params_b * _weight_gb_per_b(quant)
    kv_gb = _KV_MB_PER_TOKEN_PER_B * params_b * _kv_quant_factor(quant) * context / 1024.0
    return round(weights_gb + kv_gb + _OVERHEAD_GB, 2)


def _resolve_budget(
    free_vram_mb: Optional[int], margin_gb: float
) -> Tuple[Optional[float], Optional[float], Optional[str], Dict[str, Any]]:
    """Return (raw_budget_gb, margin_budget_gb, source, hardware)."""
    hw = detect_hardware()
    vram_mb = free_vram_mb
    if vram_mb is None:
        vram_mb = (hw.get("gpu_summary") or {}).get("best_pool_free_mb")
    if vram_mb is None and hw["gpu_available"] and hw["gpus"]:
        vram_mb = hw["gpus"][0].get("vram_free_mb") or hw["gpus"][0].get("vram_total_mb")
    if vram_mb is not None:
        raw = vram_mb / 1024.0
        return raw, max(raw - margin_gb, 0.0), "vram", hw
    ram_mb = (hw.get("system") or {}).get("ram_available_mb")
    if ram_mb is not None:
        raw = ram_mb / 1024.0
        return raw, max(raw - margin_gb, 0.0), "ram", hw
    return None, None, None, hw


def _pick(catalog: List[Dict[str, Any]], budget_gb: float, limit: int) -> List[Dict[str, Any]]:
    """Fit-filter the catalog against `budget_gb`, then rank + cap per family.

    Ranking is quality tier first (desc), model size as a tie-break (desc) -
    among equally-rated models, prefer the one that uses more of the
    available headroom. At most 2 picks per family so the top-5 isn't just
    one family's whole size ladder.
    """
    fitting: List[Dict[str, Any]] = []
    for entry in catalog:
        required = _required_gb(entry["params_b"])
        if required <= budget_gb:
            fitting.append({**entry, "required_gb": required, "margin_gb": round(budget_gb - required, 2)})
    fitting.sort(key=lambda e: (e["tier"], e["params_b"]), reverse=True)

    picked: List[Dict[str, Any]] = []
    per_family: Dict[str, int] = {}
    for entry in fitting:
        if per_family.get(entry["family"], 0) >= 2:
            continue
        picked.append(entry)
        per_family[entry["family"]] = per_family.get(entry["family"], 0) + 1
        if len(picked) >= limit:
            break
    if len(picked) < limit:
        # Family cap left the list short (e.g. a narrow budget) - top up with
        # the next-best fitting entries regardless of family.
        picked_ids = {p["id"] for p in picked}
        for entry in fitting:
            if entry["id"] in picked_ids:
                continue
            picked.append(entry)
            if len(picked) >= limit:
                break
    return picked


class StarterPackRequest(BaseModel):
    free_vram_mb: Optional[int] = Field(default=None, ge=0, le=100_000_000)
    margin_gb: float = Field(default=2.0, ge=0, le=1_024)
    limit: int = Field(default=5, ge=1, le=20)


@router.post("/registry/starter-pack")
def starter_pack(req: StarterPackRequest) -> Dict[str, Any]:
    raw_budget_gb, budget_gb, source, hw = _resolve_budget(req.free_vram_mb, req.margin_gb)

    if budget_gb is None:
        return {
            "success": True,
            "budget_source": None,
            "budget_gb": None,
            "margin_gb": req.margin_gb,
            "candidates": [],
            "hardware": {"gpu_available": hw["gpu_available"], "gpus": hw["gpus"]},
            "message": (
                "Could not determine VRAM or system RAM. Pass "
                "'free_vram_mb' explicitly to request a manual estimate."
            ),
        }

    limit = max(1, min(req.limit, len(STARTER_CATALOG)))
    candidates = _pick(STARTER_CATALOG, budget_gb, limit)
    margin_relaxed = False
    note: Optional[str] = None

    if not candidates:
        # Nothing fits within the margin - retry against the raw budget so a
        # user on tight hardware still gets *something*, flagged as tight.
        candidates = _pick(STARTER_CATALOG, raw_budget_gb, limit)
        margin_relaxed = bool(candidates)
        if candidates:
            note = (
                f"Nothing fit within the {req.margin_gb} GB safety margin, so this list "
                "uses your full detected budget instead - treat these as tight fits."
            )
        else:
            # Still nothing: hand back the smallest catalog entry as a last resort.
            smallest = min(STARTER_CATALOG, key=lambda e: e["params_b"])
            required = _required_gb(smallest["params_b"])
            candidates = [
                {**smallest, "required_gb": required, "margin_gb": round(raw_budget_gb - required, 2)}
            ]
            note = (
                "Even the smallest starter model may not comfortably fit your detected "
                "memory. Listing it anyway as a best-effort pick - consider CPU "
                "deployment or freeing memory first."
            )

    effective_budget_gb = raw_budget_gb if margin_relaxed else budget_gb
    for c in candidates:
        c["pull_name"] = c["id"]
        unit = "VRAM" if source == "vram" else "RAM"
        c["reasoning"] = (
            f"tier {c['tier']}/5, ~{c['required_gb']} GB estimated, "
            f"~{c['margin_gb']} GB headroom left in your {unit} budget"
        )

    return {
        "success": True,
        "budget_source": source,
        "raw_budget_gb": round(raw_budget_gb, 2),
        "budget_gb": round(effective_budget_gb, 2),
        "margin_gb": req.margin_gb,
        "margin_relaxed": margin_relaxed,
        "candidates": candidates,
        "hardware": {"gpu_available": hw["gpu_available"], "gpus": hw["gpus"]},
        "message": note,
    }


# --- guided recommend (Release R1) -------------------------------------------
# POST /registry/recommend turns "which model should I install for what I
# actually want to do?" into three explained picks. It reuses the same
# catalog, fit formula, and budget resolution as /registry/starter-pack above
# - this is a second view (workload + priority aware, bucketed into
# Recommended/Faster/Higher quality) over the same data, not a new engine.

_PRIORITIES = ("balanced", "best_quality", "fastest", "lowest_memory", "longest_context")
_USAGE_MODES = ("single_user_chat", "local_api", "multi_request")


class RecommendModelsRequest(BaseModel):
    use_case: Optional[str] = None  # one of WORKLOAD_TAGS; None/unrecognized = no workload bias
    priority: str = Field(default="balanced")
    expected_context: int = Field(default=8192, gt=0, le=1_000_000)
    usage_mode: str = Field(default="single_user_chat")
    free_vram_mb: Optional[int] = Field(default=None, ge=0, le=100_000_000)
    margin_gb: float = Field(default=2.0, ge=0, le=1_024)


def _priority_score(entry: Dict[str, Any], required_gb: float, budget_gb: float, priority: str) -> float:
    """Higher is better. Each priority is a different lens on the same fitting
    catalog - none of them re-estimate memory, they just re-rank it."""
    margin = budget_gb - required_gb
    if priority == "fastest":
        return -entry["params_b"] * 10 + entry["tier"]
    if priority == "lowest_memory":
        return -required_gb * 10 + entry["tier"]
    if priority == "longest_context":
        return entry.get("context_native", 0) / 1000.0 + entry["tier"]
    if priority == "best_quality":
        return entry["tier"] * 10 + entry["params_b"] * 0.1
    # balanced (default): quality-led, softened by a size/headroom tie-break
    # so it doesn't always reach for the single largest thing that fits.
    return entry["tier"] * 3 - entry["params_b"] * 0.05 + margin * 0.02


def _measured_stats(model_id: str) -> Optional[Dict[str, Any]]:
    """Best-effort lookup into saved benchmark history (server.py's opt-in
    store) for this exact model id, across any backend - this is the
    "measured on this machine" signal in a recommendation's explainability."""
    try:
        from .registry import _benchmark_rates  # lazy: avoid import-order coupling

        rates = _benchmark_rates()
    except Exception:
        return None
    matches = [v for (_backend, name), v in rates.items() if name == model_id]
    if not matches:
        return None
    best = max(matches, key=lambda m: m.get("sample_count", 0))
    return best


def _candidate_reasons(
    entry: Dict[str, Any], *, unit: str, expected_context: int, use_case: Optional[str], measured: Optional[Dict[str, Any]]
) -> List[Dict[str, str]]:
    reasons: List[Dict[str, str]] = [
        {
            "text": f"Fits your {unit} budget with ~{entry['margin_gb']} GB headroom (~{entry['required_gb']} GB estimated)",
            "kind": "estimated",
        }
    ]
    context_native = entry.get("context_native")
    if context_native:
        if expected_context <= context_native:
            reasons.append({"text": f"Published context window: {context_native // 1024}K tokens", "kind": "published"})
        else:
            reasons.append(
                {
                    "text": (
                        f"Requested {expected_context}-token context exceeds this model's published "
                        f"{context_native // 1024}K window - behavior beyond that point is not guaranteed."
                    ),
                    "kind": "published",
                }
            )
    if use_case and use_case in (entry.get("workload_tags") or []):
        reasons.append({"text": f"Tagged for {use_case.replace('_', ' ')}", "kind": "published"})
    download_gb = round(entry["params_b"] * _weight_gb_per_b("q4"), 1)
    reasons.append({"text": f"~{download_gb} GB estimated download (Q4 default quant)", "kind": "estimated"})
    if measured:
        reasons.append(
            {
                "text": f"Measured on this machine: ~{measured['tokens_per_second']} tok/s "
                f"across {measured['sample_count']} saved benchmark sample(s)",
                "kind": "measured",
            }
        )
    return reasons


def _confidence_for_candidate(entry: Dict[str, Any], expected_context: int, measured: Optional[Dict[str, Any]]) -> str:
    if measured:
        return "high"
    context_native = entry.get("context_native")
    if context_native and expected_context <= context_native:
        return "medium"
    return "low"


def _rank(fitting: List[Dict[str, Any]], budget_gb: float, priority: str) -> List[Dict[str, Any]]:
    return sorted(fitting, key=lambda e: _priority_score(e, e["required_gb"], budget_gb, priority), reverse=True)


def _fitting_candidates(budget: float, context: int = 4096) -> List[Dict[str, Any]]:
    """Catalog entries whose estimated requirement fits within `budget` GB,
    each annotated with required_gb/margin_gb. Shared by the guided-recommend
    endpoint below and the automated bakeoff (bakeoff.py)."""
    out = []
    for entry in STARTER_CATALOG:
        required = _required_gb(entry["params_b"], context=context)
        if required <= budget:
            out.append({**entry, "required_gb": required, "margin_gb": round(budget - required, 2)})
    return out


def _with_workload_bias(items: List[Dict[str, Any]], use_case: Optional[str]) -> List[Dict[str, Any]]:
    """Soft workload bias: matching entries sort first, but a non-matching
    entry is never hidden outright - a user on tight hardware, or a bakeoff
    with no exact-match candidate, should still see the best available option."""
    if not use_case:
        return items
    matched = [e for e in items if use_case in (e.get("workload_tags") or [])]
    return matched or items


def _bucket_candidate(
    entry: Dict[str, Any], *, label: str, why_summary: str, unit: str, expected_context: int,
    use_case: Optional[str],
) -> Dict[str, Any]:
    measured = _measured_stats(entry["id"])
    out = dict(entry)
    out["pull_name"] = entry["id"]
    out["bucket"] = label
    out["why_summary"] = why_summary
    out["reasons"] = _candidate_reasons(
        entry, unit=unit, expected_context=expected_context, use_case=use_case, measured=measured
    )
    out["confidence"] = _confidence_for_candidate(entry, expected_context, measured)
    if measured:
        out["measured_tokens_per_second"] = measured["tokens_per_second"]
    return out


@router.post("/registry/recommend")
def recommend_models(req: RecommendModelsRequest) -> Dict[str, Any]:
    priority = req.priority if req.priority in _PRIORITIES else "balanced"
    use_case = req.use_case if req.use_case in WORKLOAD_TAGS else None

    raw_budget_gb, budget_gb, source, hw = _resolve_budget(req.free_vram_mb, req.margin_gb)
    if budget_gb is None:
        return {
            "success": True,
            "budget_source": None,
            "recommended": None,
            "faster": None,
            "higher_quality": None,
            "hardware": {"gpu_available": hw["gpu_available"], "gpus": hw["gpus"]},
            "message": (
                "Could not determine VRAM or system RAM. Pass "
                "'free_vram_mb' explicitly to request a manual estimate."
            ),
        }

    fitting = _fitting_candidates(budget_gb, context=req.expected_context)
    margin_relaxed = False
    note: Optional[str] = None
    if not fitting:
        fitting = _fitting_candidates(raw_budget_gb, context=req.expected_context)
        margin_relaxed = bool(fitting)
        if fitting:
            note = (
                f"Nothing fit within the {req.margin_gb} GB safety margin, so these use your full "
                "detected budget instead - treat them as tight fits."
            )
        else:
            return {
                "success": True,
                "budget_source": source,
                "recommended": None,
                "faster": None,
                "higher_quality": None,
                "hardware": {"gpu_available": hw["gpu_available"], "gpus": hw["gpus"]},
                "message": "Even the smallest starter model may not comfortably fit your detected memory.",
            }

    effective_budget_gb = raw_budget_gb if margin_relaxed else budget_gb
    unit = "VRAM" if source == "vram" else "RAM"

    biased = _with_workload_bias(fitting, use_case)

    picks: List[Dict[str, Any]] = []
    chosen_ids: set = set()

    def _take(primary: List[Dict[str, Any]], fallback: List[Dict[str, Any]], label: str, why_summary: str) -> None:
        # Workload-matching candidates first; once those are exhausted (e.g. only
        # one model in the catalog is tagged for this use case), fall back to the
        # full fitting list so "faster"/"higher quality" still surface a real
        # alternative instead of an empty slot.
        for pool in (primary, fallback):
            for entry in pool:
                if entry["id"] in chosen_ids:
                    continue
                chosen_ids.add(entry["id"])
                picks.append(
                    _bucket_candidate(
                        entry, label=label, why_summary=why_summary, unit=unit,
                        expected_context=req.expected_context, use_case=use_case,
                    )
                )
                return

    _take(
        _rank(biased, effective_budget_gb, priority), _rank(fitting, effective_budget_gb, priority),
        "recommended", "Best balance for what you asked for.",
    )
    _take(
        _rank(biased, effective_budget_gb, "fastest"), _rank(fitting, effective_budget_gb, "fastest"),
        "faster", "Smaller and faster, some quality trade-off.",
    )
    _take(
        _rank(biased, effective_budget_gb, "best_quality"), _rank(fitting, effective_budget_gb, "best_quality"),
        "higher_quality", "Strongest quality that still fits.",
    )

    by_label = {p["bucket"]: p for p in picks}
    return {
        "success": True,
        "budget_source": source,
        "raw_budget_gb": round(raw_budget_gb, 2),
        "budget_gb": round(effective_budget_gb, 2),
        "margin_relaxed": margin_relaxed,
        "use_case": use_case,
        "priority": priority,
        "expected_context": req.expected_context,
        "usage_mode": req.usage_mode if req.usage_mode in _USAGE_MODES else "single_user_chat",
        "recommended": by_label.get("recommended"),
        "faster": by_label.get("faster"),
        "higher_quality": by_label.get("higher_quality"),
        "hardware": {"gpu_available": hw["gpu_available"], "gpus": hw["gpus"]},
        "message": note,
    }
