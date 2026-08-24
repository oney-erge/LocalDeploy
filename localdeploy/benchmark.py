"""Capability benchmark for local LLM profiles (v2 - harder tests).

Tests each enabled profile against five task categories:
    planning        software-project decomposition (not domestic tasks)
    classification  multi-class label with ambiguity
    code            non-trivial Python (DP, intervals, LRU cache, SQL)
    math            multi-step arithmetic, algebra, probability, matrices
    structured      JSON-schema-shaped output with field-level grading

Captures: success, elapsed seconds, response length, approx tokens/sec, accuracy
per test, and VRAM snapshots (via nvidia-smi) before/peak/after each profile.

Writes:
    reports/benchmark_<UTC-timestamp>.json   detailed results
    reports/benchmark_<UTC-timestamp>.md     human-readable summary

Run:
    python benchmark.py                      # all enabled profiles
    python benchmark.py --profile gemma3_4b_ollama_safe --profile qwen3vl_4b_ollama
    python benchmark.py --skip-categories math
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import requests
from dotenv import load_dotenv

from api_server import load_config
from localdeploy.grader_sandbox import run_code_fraction

from localdeploy.utils import api_auth_headers, api_client_base_url, app_home

APP_DIR = Path(__file__).resolve().parents[1]  # repo root in a checkout
APP_HOME = app_home()
load_dotenv(APP_HOME / ".env")

REPORTS_DIR = APP_HOME / "reports"


def api_base_url() -> str:
    return api_client_base_url()


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


@dataclass
class TestCase:
    name: str
    category: str
    prompt: str
    grader: Callable[[str], float]
    grader_explainer: str
    max_output_tokens: int = 768


# ----- helpers ----------------------------------------------------------------


def _extract_json(text: str) -> Optional[Any]:
    """Best-effort JSON extraction: fenced, bracketed, or raw."""
    candidates: List[str] = []
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        candidates.append(fence.group(1))
    # Largest balanced object or array
    obj = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if obj:
        candidates.append(obj.group(1))
    arr = re.search(r"(\[.*\])", text, flags=re.DOTALL)
    if arr:
        candidates.append(arr.group(1))
    candidates.append(text.strip())
    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception:
            continue
    return None


def _extract_python_code(text: str) -> str:
    fence = re.search(r"```(?:python|py)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    return fence.group(1) if fence else text


def _first_word(text: str) -> str:
    line = text.strip().splitlines()[0] if text.strip() else ""
    cleaned = re.sub(r"[^a-z0-9_+-]", " ", line.lower()).split()
    return cleaned[0] if cleaned else ""


# ----- planning graders -------------------------------------------------------


def _grade_chat_backend_plan(text: str) -> float:
    """Looks for 6 numbered milestones, each with name + deliverables + dependencies + effort."""
    score = 0.0
    # 6 top-level numbered items
    numbers = re.findall(r"^\s*(\d+)\.\s", text, flags=re.MULTILINE)
    if len(numbers) >= 6:
        score += 0.4
    elif len(numbers) >= 4:
        score += 0.2
    # Field markers
    lowered = text.lower()
    field_hits = sum(1 for kw in ["deliverable", "depend", "effort", "milestone"] if kw in lowered)
    score += 0.15 * (field_hits / 4)
    # T-shirt sizing presence
    if re.search(r"\b(xs|s|m|l|xl)\b", lowered):
        score += 0.15
    # Concrete tech topics (auth, websocket, db, persistence, scaling, deploy)
    topic_hits = sum(1 for kw in ["websocket", "database", "auth", "scal", "deploy", "persist", "message", "presence"] if kw in lowered)
    score += 0.3 * min(1.0, topic_hits / 4)
    return min(1.0, score)


def _grade_refactor_json(text: str) -> float:
    data = _extract_json(text)
    if not isinstance(data, dict):
        return 0.0
    score = 0.2  # parsed
    phases = data.get("phases")
    if isinstance(phases, list) and len(phases) == 4:
        score += 0.25
        # Each phase has name/description/risks/success_criteria
        per_phase = []
        for p in phases:
            if not isinstance(p, dict):
                per_phase.append(0)
                continue
            ok = sum(
                1
                for key in ("name", "description", "risks", "success_criteria")
                if key in p
            )
            # risks and success_criteria are lists >= 2
            list_ok = sum(
                1
                for key in ("risks", "success_criteria")
                if isinstance(p.get(key), list) and len(p[key]) >= 2
            )
            per_phase.append((ok / 4) * 0.7 + (list_ok / 2) * 0.3)
        score += 0.3 * (sum(per_phase) / 4)
    if isinstance(data.get("rollback_plan"), str) and len(data["rollback_plan"]) > 30:
        score += 0.1
    if isinstance(data.get("estimated_duration_days"), int):
        score += 0.15
    return min(1.0, score)


def _grade_p99_plan(text: str) -> float:
    score = 0.0
    numbers = re.findall(r"^\s*(\d+)\.\s", text, flags=re.MULTILINE)
    if len(numbers) >= 5:
        score += 0.4
    elif len(numbers) >= 3:
        score += 0.2
    lowered = text.lower()
    # Field markers required per step
    field_hits = sum(1 for kw in ["expected", "time", "access", "signal", "check"] if kw in lowered)
    score += 0.2 * (field_hits / 5)
    # Cheap-to-invasive ordering signals (check dashboards / logs / traces before code)
    cheap = any(kw in lowered for kw in ["dashboard", "metric", "log", "grafana", "alert"])
    invasive = any(kw in lowered for kw in ["redeploy", "rollback", "ssh", "production debug", "code change", "git bisect"])
    if cheap and invasive:
        cheap_idx = min((lowered.find(kw) for kw in ["dashboard", "metric", "log"] if kw in lowered), default=10_000)
        invasive_idx = min((lowered.find(kw) for kw in ["redeploy", "rollback", "ssh", "git bisect"] if kw in lowered), default=10_000)
        if cheap_idx < invasive_idx:
            score += 0.2
    # Plausible diagnostic terms
    diag_hits = sum(1 for kw in ["latency", "p99", "trace", "noisy neighbor", "gc", "memory", "cpu", "downstream", "throughput"] if kw in lowered)
    score += 0.2 * min(1.0, diag_hits / 5)
    return min(1.0, score)


# ----- classification graders -------------------------------------------------


def _grade_first_word(answer: str) -> Callable[[str], float]:
    def grader(text: str) -> float:
        word = _first_word(text)
        if word == answer:
            return 1.0
        if answer in text.lower():
            return 0.5
        return 0.0

    return grader


# ----- code graders -----------------------------------------------------------


def _grade_levenshtein(text: str) -> float:
    code = _extract_python_code(text)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0.0
    fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    if not any(f.name == "levenshtein" for f in fns):
        return 0.1
    score = 0.4
    # Run the candidate code in an isolated subprocess (see grader_sandbox).
    score += 0.6 * run_code_fraction(code, "levenshtein")
    return min(1.0, score)


def _grade_merge_intervals(text: str) -> float:
    code = _extract_python_code(text)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0.0
    if not any(isinstance(n, ast.FunctionDef) and n.name == "merge_intervals" for n in ast.walk(tree)):
        return 0.1
    score = 0.4
    score += 0.6 * run_code_fraction(code, "merge_intervals")
    return min(1.0, score)


def _grade_lru_cache(text: str) -> float:
    code = _extract_python_code(text)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0.0
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "LRUCache"]
    if not classes:
        return 0.1
    score = 0.3
    methods = {m.name for m in ast.walk(classes[0]) if isinstance(m, ast.FunctionDef)}
    if {"__init__", "get", "put"}.issubset(methods):
        score += 0.2
    score += 0.5 * run_code_fraction(code, "lru_cache")
    return min(1.0, score)


def _grade_sql_second_highest(text: str) -> float:
    cleaned = text.strip().lower()
    score = 0.0
    if "select" in cleaned:
        score += 0.2
    # Acceptable patterns
    if "limit 1" in cleaned and "offset 1" in cleaned:
        score += 0.5
    elif re.search(r"dense_rank|rank\(\)\s*over", cleaned):
        score += 0.5
    elif "distinct" in cleaned and re.search(r"order\s+by\s+salary\s+desc", cleaned):
        score += 0.4
    elif "max" in cleaned and "where" in cleaned and "<" in cleaned:
        score += 0.4
    if "null" in cleaned or "ifnull" in cleaned or "coalesce" in cleaned:
        score += 0.2
    if "employee" in cleaned:
        score += 0.1
    return min(1.0, score)


# ----- math graders -----------------------------------------------------------


def _grade_number(expected: float, tolerance: float = 0.5) -> Callable[[str], float]:
    def grader(text: str) -> float:
        for cand in re.findall(r"-?\d+(?:\.\d+)?", text):
            try:
                if abs(float(cand) - expected) <= tolerance:
                    return 1.0
            except ValueError:
                continue
        return 0.0

    return grader


def _grade_time_953(text: str) -> float:
    # Accept 9:52, 9:53, 9:54 AM. The correct answer is unambiguously AM (both
    # trains depart in the morning), so a matching "9:53 PM" must be rejected -
    # an explicit PM marker on the match disqualifies it; no marker or an
    # explicit AM marker is accepted.
    for hr, mn, meridiem in re.findall(r"(\d{1,2}):(\d{2})\s*([AaPp]\.?[Mm]\.?)?", text):
        if meridiem and meridiem.strip().lower().startswith("p"):
            continue
        try:
            h = int(hr)
            m = int(mn)
        except ValueError:
            continue
        if h == 9 and abs(m - 53) <= 1:
            return 1.0
    return 0.0


def _grade_fraction(target_num: int, target_den: int) -> Callable[[str], float]:
    def grader(text: str) -> float:
        for num, den in re.findall(r"(\d+)\s*/\s*(\d+)", text):
            try:
                n, d = int(num), int(den)
                if d != 0 and n * target_den == target_num * d:
                    return 1.0
            except ValueError:
                continue
        return 0.0

    return grader


# ----- structured output graders ----------------------------------------------


def _grade_request_parse(text: str) -> float:
    data = _extract_json(text)
    if not isinstance(data, dict):
        return 0.0
    score = 0.2
    if data.get("action") == "read":
        score += 0.2
    entity = str(data.get("entity") or "").lower()
    if "invoice" in entity:
        score += 0.15
    filters = data.get("filters")
    if isinstance(filters, dict):
        f_lower = json.dumps(filters).lower()
        if "overdue" in f_lower:
            score += 0.15
        if "acme-42" in f_lower or "acme_42" in f_lower or "acme" in f_lower:
            score += 0.15
    if str(data.get("output_format") or "").lower() == "csv":
        score += 0.15
    return min(1.0, score)


def _grade_flight_extract(text: str) -> float:
    data = _extract_json(text)
    if not isinstance(data, dict):
        return 0.0
    score = 0.2
    if "united" in str(data.get("airline", "")).lower():
        score += 0.15
    flight = data.get("flight", {})
    if isinstance(flight, dict):
        if str(flight.get("origin", "")).upper() == "ORD":
            score += 0.15
        if str(flight.get("destination", "")).upper() == "LAX":
            score += 0.15
        if re.match(r"^\d{1,2}:\d{2}", str(flight.get("departure_time", ""))):
            score += 0.1
    pax = data.get("passenger", {})
    if isinstance(pax, dict):
        if "smith" in str(pax.get("last_name", "")).lower():
            score += 0.1
        if "4429" in str(pax.get("ff_number", "")):
            score += 0.15
    return min(1.0, score)


def _grade_function_schema(text: str) -> float:
    data = _extract_json(text)
    if not isinstance(data, dict):
        return 0.0
    score = 0.15
    params = data.get("parameters") or data
    if not isinstance(params, dict):
        return score
    if params.get("type") == "object":
        score += 0.15
    props = params.get("properties") or {}
    if not isinstance(props, dict):
        return score
    needed = {"title", "participants", "duration_minutes", "location"}
    score += 0.3 * (len(needed & set(props.keys())) / len(needed))
    if isinstance(props.get("participants"), dict) and props["participants"].get("type") == "array":
        score += 0.1
    if isinstance(props.get("duration_minutes"), dict) and props["duration_minutes"].get("type") == "integer":
        score += 0.1
    required = params.get("required") or []
    if isinstance(required, list) and "title" in required and "participants" in required:
        score += 0.2
    return min(1.0, score)


# ----- HARD structured graders (modeled on YBM pydantic schemas) --------------
# These exercise nested enums, conditional fields, capability-subset constraints,
# multi-entity extraction, and reasoning-then-output - patterns the easy tests miss.


_INTENT_ROUTES = {
    "conversation", "status", "desktop.observe", "computer.use", "browser.open",
    "browser.control", "filesystem.manage", "document.manage", "artifact.deliver",
    "code.interpreter", "coding.agent", "schedule.manage", "configuration", "unknown",
}
_TASK_TYPES = {"development", "configuration", "admin_control", "desktop_observation", "question", "status_request", "other"}
_RISK_LEVELS = {"low", "medium", "high", "critical"}
_CAPABILITIES = {
    "telegram.send", "telegram.receive", "llm.generate", "stt.transcribe", "tts.synthesize",
    "vscode.read_state", "vscode.write_files", "terminal.run", "filesystem.read", "filesystem.write",
    "desktop.screenshot", "desktop.control", "browser.open", "browser.control", "schedule.manage",
    "github.read", "github.push", "dependencies.install",
}
_DELIVERY_KINDS = {"none", "latest", "file", "screenshot"}


def _grade_intent_classify(text: str) -> float:
    """User said: 'Open chrome go to news.ycombinator.com take a screenshot of the front page and send it to me.'

    Expected MessageClassification + nested OrchestrationIntent.
    """
    data = _extract_json(text)
    if not isinstance(data, dict):
        return 0.0
    score = 0.0
    # is_task=true
    if data.get("is_task") is True:
        score += 0.1
    # task_type in valid enum, prefer "other" (browser→other per YBM rule)
    tt = str(data.get("task_type", "")).lower()
    if tt in _TASK_TYPES:
        score += 0.05
        if tt == "other":
            score += 0.05
    # confidence is number in [0,1]
    conf = data.get("confidence")
    if isinstance(conf, (int, float)) and 0.0 <= float(conf) <= 1.0:
        score += 0.05
        if float(conf) >= 0.7:
            score += 0.05
    # normalized_objective mentions key entities
    obj = str(data.get("normalized_objective") or "").lower()
    if "screenshot" in obj and ("ycombinator" in obj or "hacker news" in obj or "front page" in obj):
        score += 0.1
    elif "screenshot" in obj or "ycombinator" in obj:
        score += 0.05
    # reason is a non-empty string
    reason = data.get("reason")
    if isinstance(reason, str) and len(reason.strip()) >= 5:
        score += 0.05
    # nested intent
    intent = data.get("intent")
    if isinstance(intent, dict):
        score += 0.05
        route = str(intent.get("route", "")).lower()
        if route in _INTENT_ROUTES:
            score += 0.05
            if route in {"browser.open", "browser.control"}:
                score += 0.1
        url = str(intent.get("url") or "")
        if "ycombinator" in url.lower():
            score += 0.1
        delivery = str(intent.get("delivery", "")).lower()
        if delivery in _DELIVERY_KINDS:
            score += 0.05
            if delivery in {"screenshot", "file", "latest"}:
                score += 0.05
        if isinstance(intent.get("needs_plan_first"), bool):
            score += 0.05
    return min(1.0, score)


def _grade_plan_orchestration(text: str) -> float:
    """Task: 'Take a screenshot of the current desktop and send it to me.'

    Expected PlanModel-like JSON with desktop.screenshot + telegram.send capabilities
    and the capability-subset constraint between top-level and per-step.
    """
    data = _extract_json(text)
    if not isinstance(data, dict):
        return 0.0
    score = 0.05  # parsed
    # objective mentions screenshot
    obj = str(data.get("objective") or "").lower()
    if "screenshot" in obj or "desktop" in obj:
        score += 0.1
    # required_capabilities is array of valid enum values
    top_caps = data.get("required_capabilities")
    if isinstance(top_caps, list):
        score += 0.05
        cap_set = {str(c).lower() for c in top_caps if isinstance(c, str)}
        valid_caps = cap_set & _CAPABILITIES
        if "desktop.screenshot" in valid_caps:
            score += 0.1
        if "telegram.send" in valid_caps or "artifact.deliver" in {str(c).lower() for c in top_caps}:
            score += 0.1
    # steps is array of objects with required keys
    steps = data.get("steps")
    if isinstance(steps, list) and len(steps) >= 2:
        score += 0.1
        # Each step has title, description, required_capabilities, risk_level
        per_step_ok = 0
        capability_subset_ok = True
        top_cap_set = {str(c).lower() for c in (top_caps or []) if isinstance(c, str)}
        for step in steps:
            if not isinstance(step, dict):
                capability_subset_ok = False
                continue
            field_hits = sum(1 for k in ("title", "description", "required_capabilities", "risk_level") if k in step)
            if field_hits >= 3:
                per_step_ok += 1
            # capability subset check
            step_caps = step.get("required_capabilities") or []
            if isinstance(step_caps, list) and top_cap_set:
                if not {str(c).lower() for c in step_caps if isinstance(c, str)}.issubset(top_cap_set):
                    capability_subset_ok = False
            # risk_level is valid enum
            if str(step.get("risk_level", "")).lower() in _RISK_LEVELS:
                pass  # rolled into field_hits
        score += 0.1 * (per_step_ok / max(1, len(steps)))
        if capability_subset_ok and top_cap_set:
            score += 0.15
    # success_criteria is array
    sc = data.get("success_criteria")
    if isinstance(sc, list) and len(sc) >= 1:
        score += 0.1
    # approval_gates is array (may be empty)
    ag = data.get("approval_gates")
    if isinstance(ag, list):
        score += 0.05
    return min(1.0, score)


def _grade_approval_decision(text: str) -> float:
    """Action: delete user's document. Expected: cautious decision with conditional field.

    Conditional constraint: confirmation_prompt is required iff needs_user_confirmation=True.
    Cautious decision: 'reject' or 'needs_clarification', NOT auto-approve.
    """
    data = _extract_json(text)
    if not isinstance(data, dict):
        return 0.0
    score = 0.05  # parsed
    decision = str(data.get("decision", "")).lower()
    if decision in {"reject", "needs_clarification"}:
        score += 0.2
    elif decision == "approve":
        score += 0.0  # wrong call for destructive action
    risk = str(data.get("risk_level", "")).lower()
    if risk in {"high", "critical"}:
        score += 0.15
    elif risk == "medium":
        score += 0.05
    reasons = data.get("reasons")
    if isinstance(reasons, list) and len(reasons) >= 2:
        score += 0.1
    needs_conf = data.get("needs_user_confirmation")
    conf_prompt = data.get("confirmation_prompt")
    # Conditional: confirmation_prompt iff needs_user_confirmation
    if isinstance(needs_conf, bool):
        score += 0.05
        if needs_conf is True and isinstance(conf_prompt, str) and len(conf_prompt.strip()) >= 5:
            score += 0.15
        elif needs_conf is False and (conf_prompt is None or conf_prompt == ""):
            score += 0.1
        elif needs_conf is True and not isinstance(conf_prompt, str):
            pass  # constraint violation
    caps = data.get("required_capabilities")
    if isinstance(caps, list):
        cap_set = {str(c).lower() for c in caps if isinstance(c, str)}
        if "filesystem.delete" in cap_set or "filesystem.write" in cap_set:
            score += 0.1
    alt = data.get("alternative_action")
    if isinstance(alt, dict) and "action" in alt and "rationale" in alt:
        score += 0.1
    elif alt is None:
        score += 0.05
    return min(1.0, score)


def _grade_multi_task_extract(text: str) -> float:
    """User asked for 2 tasks: watch Downloads for PDFs, schedule daily 9am report.

    Plus a negative constraint about Codex.
    """
    data = _extract_json(text)
    if not isinstance(data, dict):
        return 0.0
    score = 0.05  # parsed
    tasks = data.get("tasks")
    if not isinstance(tasks, list):
        return score
    if len(tasks) == 2:
        score += 0.2
    elif len(tasks) == 1 or len(tasks) == 3:
        score += 0.1
    # Find task with filesystem/document route and watch trigger
    found_watch = False
    found_schedule = False
    codex_constraint = False
    for t in tasks:
        if not isinstance(t, dict):
            continue
        route = str(t.get("route", "")).lower()
        trigger = t.get("trigger")
        trigger_kind = ""
        if isinstance(trigger, dict):
            trigger_kind = str(trigger.get("kind", "")).lower()
        if route in {"filesystem.manage", "document.manage"} and trigger_kind == "watch":
            found_watch = True
        if route == "schedule.manage" and trigger_kind == "schedule":
            found_schedule = True
        constraints = t.get("constraints")
        if isinstance(constraints, list):
            joined = " ".join(str(c).lower() for c in constraints)
            if "codex" in joined or "copilot" in joined or "no advanced" in joined or "advanced" in joined:
                codex_constraint = True
    if found_watch:
        score += 0.2
    if found_schedule:
        score += 0.2
    # Schedule task should mention 9am or daily
    for t in tasks:
        if not isinstance(t, dict):
            continue
        if str(t.get("route", "")).lower() == "schedule.manage":
            trigger = t.get("trigger") or {}
            details = str(trigger.get("details", "") if isinstance(trigger, dict) else "").lower()
            if "9am" in details or "9:00" in details or "0900" in details or "daily" in details:
                score += 0.1
                break
    if codex_constraint:
        score += 0.15
    # ignored_instructions is array (can be empty)
    if isinstance(data.get("ignored_instructions"), list):
        score += 0.05
    # Sequential ids
    ids = [t.get("id") for t in tasks if isinstance(t, dict)]
    if ids and all(isinstance(i, int) for i in ids) and ids == list(range(1, len(ids) + 1)):
        score += 0.05
    return min(1.0, score)


def _grade_review_extract(text: str) -> float:
    data = _extract_json(text)
    if not isinstance(data, dict):
        return 0.0
    score = 0.2
    if "eternal drift" in str(data.get("title", "")).lower():
        score += 0.2
    try:
        if abs(float(data.get("score", -1)) - 7) <= 0.01:
            score += 0.2
    except (TypeError, ValueError):
        pass
    pros = data.get("pros")
    cons = data.get("cons")
    if isinstance(pros, list) and len(pros) >= 1:
        score += 0.2
    if isinstance(cons, list) and len(cons) >= 1:
        score += 0.2
    return min(1.0, score)


# ---------------------------------------------------------------------------
# Test catalog
# ---------------------------------------------------------------------------


TEST_CASES: List[TestCase] = [
    # --- Planning (software/project oriented) ---
    TestCase(
        name="plan_chat_backend",
        category="planning",
        prompt=(
            "Plan the implementation of a real-time chat application backend (1-on-1 + group rooms, presence, "
            "message history, scalable to 50k concurrent users). Output 6 ordered milestones as a numbered "
            "Markdown list. For each milestone include: name, 2-3 concrete deliverables, dependencies on earlier "
            "milestones, and effort estimate as XS/S/M/L/XL. Use indented sub-items for the deliverables."
        ),
        grader=_grade_chat_backend_plan,
        grader_explainer="6 numbered milestones with deliverables/dependencies/effort fields + concrete tech topics",
        max_output_tokens=1024,
    ),
    TestCase(
        name="plan_refactor_json",
        category="planning",
        prompt=(
            "I need to refactor a 5000-line monolithic Python file into modules. Output ONLY a JSON object "
            "(no markdown fence) with keys: 'phases' (array of exactly 4 objects, each having 'name', "
            "'description', 'risks' as array of at least 2 strings, 'success_criteria' as array of at least 2 "
            "strings), 'rollback_plan' (string of 2-3 sentences), 'estimated_duration_days' (integer)."
        ),
        grader=_grade_refactor_json,
        grader_explainer="Valid JSON with 4 phases (each w/ name+description+risks[≥2]+criteria[≥2]) + rollback + days",
        max_output_tokens=1024,
    ),
    TestCase(
        name="plan_p99_investigation",
        category="planning",
        prompt=(
            "Production microservice p99 latency just doubled from 200ms to 400ms with no recent deploys. "
            "Outline a 5-step investigation plan, ordered from cheapest/fastest to most invasive. "
            "For each step include: what to check, expected signal of a hit, time estimate, required access. "
            "Numbered Markdown list."
        ),
        grader=_grade_p99_plan,
        grader_explainer="5 ordered steps, cheap-before-invasive, all 4 fields per step + diagnostic vocabulary",
        max_output_tokens=900,
    ),
    # --- Classification (multi-class, some ambiguity) ---
    TestCase(
        name="cls_tone",
        category="classification",
        prompt=(
            "Classify the tone of this sentence as exactly one of: sarcastic, sincere, anxious, defiant. "
            "'Oh sure, because what could possibly go wrong with another all-hands meeting.' "
            "Reply with one word, lowercase, no punctuation."
        ),
        grader=_grade_first_word("sarcastic"),
        grader_explainer="First word equals 'sarcastic'",
        max_output_tokens=16,
    ),
    TestCase(
        name="cls_intensity",
        category="classification",
        prompt=(
            "Pick one: strong_positive, mild_positive, neutral, mild_negative, strong_negative. "
            "'The interface is fine I guess, gets the job done eventually.' "
            "Reply with the label only."
        ),
        grader=_grade_first_word("mild_negative"),
        grader_explainer="First word equals 'mild_negative'",
        max_output_tokens=16,
    ),
    TestCase(
        name="cls_bug_severity",
        category="classification",
        prompt=(
            "Bug severity (one of: critical, major, minor, trivial): 'Form labels are missing for 3 fields "
            "in our internal admin panel; screen readers cannot announce which field is focused.' "
            "Reply with the label only."
        ),
        grader=_grade_first_word("major"),
        grader_explainer="First word equals 'major'",
        max_output_tokens=16,
    ),
    TestCase(
        name="cls_code_review",
        category="classification",
        prompt=(
            "Code review comment type (one of: nitpick, suggestion, blocker, question): "
            "'This violates our auth policy - every endpoint needs the @require_auth decorator before merge.' "
            "Reply with the label only."
        ),
        grader=_grade_first_word("blocker"),
        grader_explainer="First word equals 'blocker'",
        max_output_tokens=16,
    ),
    TestCase(
        name="cls_emotion",
        category="classification",
        prompt=(
            "Dominant emotion (one of: anger, fear, joy, sadness, surprise, disgust): "
            "'Wait. They scheduled the deploy for Friday afternoon?' "
            "Reply with the label only."
        ),
        grader=_grade_first_word("surprise"),
        grader_explainer="First word equals 'surprise'",
        max_output_tokens=16,
    ),
    # --- Code (harder) ---
    TestCase(
        name="code_levenshtein",
        category="code",
        prompt=(
            "Write a Python function `levenshtein(a: str, b: str) -> int` returning the Levenshtein edit "
            "distance using dynamic programming. Handle empty strings. Reply with only the function in a "
            "single ```python``` block."
        ),
        grader=_grade_levenshtein,
        grader_explainer="Parses; defines levenshtein; passes 5 unit cases incl. empty + kitten/sitting=3",
        max_output_tokens=900,
    ),
    TestCase(
        name="code_merge_intervals",
        category="code",
        prompt=(
            "Write a Python function `merge_intervals(intervals: list[tuple[int,int]]) -> list[tuple[int,int]]` "
            "that merges overlapping intervals. Example: [(1,3),(2,6),(8,10)] -> [(1,6),(8,10)]. "
            "Empty input returns []. Reply with only the function in a ```python``` block."
        ),
        grader=_grade_merge_intervals,
        grader_explainer="Parses; merges 4 unit cases correctly including empty + chained overlaps",
        max_output_tokens=700,
    ),
    TestCase(
        name="code_lru_cache",
        category="code",
        prompt=(
            "Write a Python class `LRUCache` with __init__(capacity), get(key) -> value or -1, put(key, value). "
            "Both O(1). Use dict + doubly-linked list (or OrderedDict). Reply with only the class in a "
            "```python``` block."
        ),
        grader=_grade_lru_cache,
        grader_explainer="Parses; standard LeetCode LRU sequence passes (5 assertions)",
        max_output_tokens=900,
    ),
    TestCase(
        name="code_sql_second_highest",
        category="code",
        prompt=(
            "Write SQL to find the second-highest salary from an Employee table with columns id, name, salary. "
            "Return NULL when there is no second-highest. Use standard SQL. Reply with only the SQL statement, "
            "no explanation."
        ),
        grader=_grade_sql_second_highest,
        grader_explainer="SELECT + (LIMIT 1 OFFSET 1 | DENSE_RANK | MAX <) + NULL handling + Employee table",
        max_output_tokens=300,
    ),
    # --- Math (harder) ---
    TestCase(
        name="math_power",
        category="math",
        prompt="What is 23 to the 4th power (23^4)? Reply with only the integer answer.",
        grader=_grade_number(23 ** 4),
        grader_explainer=f"Answer is {23 ** 4}",
        max_output_tokens=32,
    ),
    TestCase(
        name="math_trains",
        category="math",
        prompt=(
            "Train A leaves station A at 8:00 AM going 60 mph east. Train B leaves station B (180 miles east "
            "of A) at 9:00 AM going 75 mph west, toward A. At what time do they meet? "
            "Reply with only the time in HH:MM AM/PM format."
        ),
        grader=_grade_time_953,
        grader_explainer="9:53 AM ± 1 minute",
        max_output_tokens=64,
    ),
    TestCase(
        name="math_algebra",
        category="math",
        prompt="Solve for x: 2x + 5 = 3(x - 4) + 7. Reply with only the integer value of x.",
        grader=_grade_number(10),
        grader_explainer="x = 10",
        max_output_tokens=32,
    ),
    TestCase(
        name="math_determinant",
        category="math",
        prompt=(
            "Compute the determinant of the 3x3 matrix [[1,2,3],[4,5,6],[7,8,10]]. "
            "Reply with only the integer."
        ),
        grader=_grade_number(-3),
        grader_explainer="det = -3",
        max_output_tokens=64,
    ),
    TestCase(
        name="math_probability",
        category="math",
        prompt=(
            "A fair coin is flipped 5 times. What's the probability of getting exactly 3 heads? "
            "Reply with only a reduced fraction like 5/16."
        ),
        grader=_grade_fraction(5, 16),
        grader_explainer="5/16",
        max_output_tokens=32,
    ),
    # --- Structured output (JSON) ---
    TestCase(
        name="json_request_parse",
        category="structured",
        prompt=(
            "Parse this user request into a JSON object. Schema: "
            "{action: one of [create, read, update, delete], entity: string, filters: object of key-value pairs, "
            "output_format: one of [json, csv, markdown]}. "
            "User said: 'Show me all overdue invoices for customer ACME-42 as a CSV'. "
            "Reply with valid JSON only, no markdown fence, no prose."
        ),
        grader=_grade_request_parse,
        grader_explainer="JSON with action=read, entity=invoices, filters contain overdue + ACME-42, output=csv",
        max_output_tokens=300,
    ),
    TestCase(
        name="json_flight_extract",
        category="structured",
        prompt=(
            "Extract structured data from this booking request: "
            "'Book the 8:30am ORD to LAX on United for Mr. James Smith, frequent flyer 4429-XK.' "
            "Output a JSON object matching: {airline: string, flight: {origin: string, destination: string, "
            "departure_time: string in HH:MM format}, passenger: {first_name: string, last_name: string, "
            "title: string, ff_number: string}}. Reply with JSON only, no markdown fence."
        ),
        grader=_grade_flight_extract,
        grader_explainer="JSON with airline=United, ORD->LAX, 8:30, Smith, 4429-XK",
        max_output_tokens=400,
    ),
    TestCase(
        name="json_function_schema",
        category="structured",
        prompt=(
            "Generate the JSON Schema for an OpenAI function-calling parameters object for a function named "
            "schedule_meeting. Parameters: title (string, required), participants (array of strings, required), "
            "duration_minutes (integer, min 15), location (string, optional). Output a JSON object with a top-"
            "level 'parameters' key whose value is the JSON Schema (type=object, properties, required). "
            "Reply with JSON only, no markdown fence, no prose."
        ),
        grader=_grade_function_schema,
        grader_explainer="JSON has parameters.type=object, properties has 4 fields, required contains title+participants",
        max_output_tokens=500,
    ),
    TestCase(
        name="json_review_extract",
        category="structured",
        prompt=(
            "From this review extract structured data: 'The Eternal Drift movie was visually stunning with great "
            "cinematography but the pacing dragged in act two, ultimately 7 out of 10.' "
            "Output JSON: {title: string, score: number 0-10, pros: array of strings, cons: array of strings}. "
            "JSON only, no markdown fence."
        ),
        grader=_grade_review_extract,
        grader_explainer="JSON: title=The Eternal Drift, score=7, ≥1 pro, ≥1 con",
        max_output_tokens=400,
    ),
    # --- Hard structured output (modeled on YBM pydantic schemas) ---
    TestCase(
        name="json_intent_classify",
        category="structured_hard",
        prompt=(
            "You are a router for a local agent-control bot. Classify this user message into JSON matching:\n"
            "{\n"
            "  \"is_task\": boolean,\n"
            "  \"task_type\": one of [development, configuration, admin_control, desktop_observation, question, status_request, other],\n"
            "  \"normalized_objective\": string or null,\n"
            "  \"confidence\": number 0.0-1.0,\n"
            "  \"reason\": string (one sentence),\n"
            "  \"intent\": {\n"
            "    \"route\": one of [conversation, status, desktop.observe, computer.use, browser.open, browser.control, filesystem.manage, document.manage, artifact.deliver, code.interpreter, coding.agent, schedule.manage, configuration, unknown],\n"
            "    \"objective\": string or null,\n"
            "    \"url\": string or null,\n"
            "    \"folder_path\": string or null,\n"
            "    \"file_path\": string or null,\n"
            "    \"page_limit\": integer or null,\n"
            "    \"delivery\": one of [none, latest, file, screenshot],\n"
            "    \"needs_plan_first\": boolean\n"
            "  } or null\n"
            "}\n\n"
            "Rule: anything browser/file/automation/code → task_type=other (the specific route lives in intent.route).\n\n"
            "User message: \"Open chrome go to news.ycombinator.com take a screenshot of the front page and send it to me.\"\n\n"
            "Reply with JSON only, no markdown fence, no prose."
        ),
        grader=_grade_intent_classify,
        grader_explainer="is_task=true, task_type=other, route=browser.open, url has ycombinator, delivery=screenshot, nested intent shape",
        max_output_tokens=600,
    ),
    TestCase(
        name="json_plan_orchestration",
        category="structured_hard",
        prompt=(
            "Generate a JSON execution plan for: \"Take a screenshot of the current desktop and send it to me.\"\n\n"
            "Schema:\n"
            "{\n"
            "  \"objective\": string,\n"
            "  \"assumptions\": array of strings,\n"
            "  \"required_capabilities\": array of capability strings (from: telegram.send, llm.generate, vscode.read_state, terminal.run, filesystem.read, filesystem.write, desktop.screenshot, desktop.control, browser.open, browser.control, schedule.manage, github.read, github.push, dependencies.install),\n"
            "  \"steps\": array of {\n"
            "    \"title\": string,\n"
            "    \"description\": string,\n"
            "    \"required_capabilities\": array of capability strings (subset of the plan's top-level required_capabilities),\n"
            "    \"risk_level\": one of [low, medium, high, critical],\n"
            "    \"requires_approval\": boolean,\n"
            "    \"tool_name\": string,\n"
            "    \"tool_input\": object,\n"
            "    \"expected_output\": string\n"
            "  },\n"
            "  \"success_criteria\": array of strings,\n"
            "  \"approval_gates\": array of {capability: string, risk_level: enum, summary: string}\n"
            "}\n\n"
            "Hard constraints:\n"
            "- The plan MUST include at least one step using desktop.screenshot AND at least one using telegram.send.\n"
            "- Each step's required_capabilities MUST be a subset of the plan's top-level required_capabilities.\n"
            "- desktop.screenshot and telegram.send are both risk_level=low and do NOT require approval.\n\n"
            "Reply with JSON only, no markdown fence, no prose."
        ),
        grader=_grade_plan_orchestration,
        grader_explainer="Valid PlanModel with screenshot+send caps, ≥2 steps, capability-subset constraint holds",
        max_output_tokens=1200,
    ),
    TestCase(
        name="json_approval_decision",
        category="structured_hard",
        prompt=(
            "A user wants to delete the file C:\\Users\\me\\Documents\\proposal_draft.docx. "
            "This is a destructive action on a user document. Generate a cautious approval decision as JSON:\n\n"
            "{\n"
            "  \"decision\": one of [approve, reject, needs_clarification],\n"
            "  \"risk_level\": one of [low, medium, high, critical],\n"
            "  \"reasons\": array of at least 2 strings,\n"
            "  \"required_capabilities\": array of strings (from: filesystem.read, filesystem.write, filesystem.delete, telegram.send),\n"
            "  \"mitigations\": array of strings (can be empty),\n"
            "  \"alternative_action\": object with {action: string, rationale: string} OR null,\n"
            "  \"needs_user_confirmation\": boolean,\n"
            "  \"confirmation_prompt\": string OR null\n"
            "}\n\n"
            "Hard constraint: confirmation_prompt MUST be a non-empty string when needs_user_confirmation=true, "
            "and MUST be null when needs_user_confirmation=false.\n\n"
            "Reply with JSON only, no markdown fence, no prose."
        ),
        grader=_grade_approval_decision,
        grader_explainer="Cautious decision (reject/needs_clarification), high risk, ≥2 reasons, conditional confirmation_prompt holds",
        max_output_tokens=600,
    ),
    TestCase(
        name="json_multi_task_extract",
        category="structured_hard",
        prompt=(
            "Extract a task list from this user message:\n"
            "\"Hey can you do these: Watch my Downloads folder for new PDFs and send a summary of each one to me. "
            "Also schedule a daily 9am report on yesterday's commits in my github.com/oney-erge/LocalDeploy repo. "
            "Don't bother me about Codex or anything advanced.\"\n\n"
            "Schema:\n"
            "{\n"
            "  \"tasks\": array of {\n"
            "    \"id\": integer (sequential starting at 1),\n"
            "    \"objective\": string (concise),\n"
            "    \"route\": one of [filesystem.manage, document.manage, schedule.manage, browser.open, coding.agent],\n"
            "    \"trigger\": {kind: one of [immediate, watch, schedule], details: string},\n"
            "    \"delivery\": one of [none, latest, file, screenshot],\n"
            "    \"constraints\": array of strings (user's negative constraints)\n"
            "  },\n"
            "  \"ignored_instructions\": array of strings (instructions you cannot fulfill, can be empty)\n"
            "}\n\n"
            "Reply with JSON only, no markdown fence, no prose."
        ),
        grader=_grade_multi_task_extract,
        grader_explainer="2 tasks: one fs/doc watch + one schedule.manage with 9am/daily, Codex constraint preserved",
        max_output_tokens=900,
    ),
]


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


@dataclass
class TestResult:
    name: str
    category: str
    success: bool
    elapsed_seconds: float
    response_length: int
    response_preview: str
    accuracy: float
    error: Optional[str] = None
    warning: Optional[str] = None
    approx_tokens_per_second: Optional[float] = None
    repetition: int = 1
    metrics: Dict[str, Any] = field(default_factory=dict)
    tokens_per_second_source: str = "estimated"
    warm_state: Optional[str] = None
    context_limit_used: Optional[int] = None


@dataclass
class ProfileResult:
    profile: str
    model_id: str
    backend: str
    enabled: bool
    recommended_for_8gb_vram: Any
    vram_before_mb: Optional[int]
    vram_after_mb: Optional[int]
    vram_peak_mb: Optional[int]
    tests: List[TestResult] = field(default_factory=list)
    fits_in_vram: bool = True
    notes: List[str] = field(default_factory=list)


def _percentile(values: List[float], pct: float) -> Optional[float]:
    """Linear-interpolation percentile (the common default, e.g. numpy's).
    None for an empty list; the single value for a singleton list."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(ordered) - 1)
    if f == c:
        return ordered[f]
    return ordered[f] * (c - k) + ordered[c] * (k - f)


def _stats_block(values: List[float], digits: int = 3) -> Dict[str, Optional[float]]:
    """Mean/median/stdev/min/max/P90/P95 in one call - the repeated-run
    statistics the benchmark expansion plan asks for, computed once and
    reused for both latency and tokens/sec rather than duplicated per metric."""
    if not values:
        return {
            "mean": None, "median": None, "stdev": 0.0, "min": None, "max": None,
            "p90": None, "p95": None,
        }
    return {
        "mean": round(statistics.mean(values), digits),
        "median": round(statistics.median(values), digits),
        "stdev": round(statistics.stdev(values), digits) if len(values) > 1 else 0.0,
        "min": round(min(values), digits),
        "max": round(max(values), digits),
        "p90": round(_percentile(values, 90), digits),
        "p95": round(_percentile(values, 95), digits),
    }


def nvidia_smi_used_mb() -> Optional[int]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    try:
        return int(output.decode("ascii", errors="ignore").strip().splitlines()[0].strip())
    except (ValueError, IndexError):
        return None


_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


def strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models (Qwen3, DeepSeek-R1).

    Leaves the final answer that comes after the last </think>. If the response is just
    thinking with no </think> close (truncated), returns the original so the grader sees
    something rather than empty.
    """
    if "</think>" in text.lower():
        cleaned = _THINK_BLOCK.sub("", text).strip()
        return cleaned if cleaned else text
    return text


def call_chat(
    base_url: str,
    profile_name: str,
    profile_cfg: Dict[str, Any],
    test: TestCase,
    timeout: int,
    num_gpu: Optional[int] = None,
    context_override: Optional[int] = None,
) -> Dict[str, Any]:
    suffix = str(profile_cfg.get("prompt_suffix") or "")
    prompt = test.prompt + (("\n\n" + suffix) if suffix else "")
    payload = {
        "profile": profile_name,
        "prompt": prompt,
        "safe_mode": True,
        "max_output_tokens": test.max_output_tokens,
    }
    # Pin the device on the inference call itself so a CPU/GPU benchmark stays on
    # the device it warmed on instead of letting Ollama re-place the model.
    if num_gpu is not None:
        payload["num_gpu"] = num_gpu
    # Context-scaling sweep: /chat already accepts a per-request context_limit
    # override (prepare_request), so no server-side plumbing is needed to test
    # the same model at several context lengths without editing its profile.
    if context_override is not None:
        payload["context_limit"] = context_override
        payload["allow_clamp"] = True
    try:
        response = requests.post(
            f"{base_url}/chat",
            json=payload,
            headers=api_auth_headers(),
            timeout=timeout + 10,
        )
    except requests.Timeout:
        return {"success": False, "error": f"client-side timeout after {timeout}s", "elapsed_seconds": float(timeout)}
    except requests.ConnectionError:
        return {"success": False, "error": f"could not reach {base_url}", "elapsed_seconds": 0.0}
    try:
        data = response.json()
    except ValueError:
        return {"success": False, "error": f"HTTP {response.status_code}: {response.text[:200]}", "elapsed_seconds": 0.0}

    # Strip <think> blocks if the profile is a reasoning model - the answer lives
    # after </think> for these. Leave the raw response intact when not requested.
    if profile_cfg.get("strip_thinking_tags") and isinstance(data.get("response"), str):
        data["response"] = strip_thinking_tags(data["response"])
    return data


def _is_oom(error: Optional[str]) -> bool:
    err = str(error or "").lower()
    return "memory" in err or "cuda" in err or "out of memory" in err


def _is_not_pulled(error: Optional[str]) -> bool:
    err = str(error or "").lower()
    return "not available" in err or "not found" in err or "pull" in err


def execute_test(
    base_url: str,
    profile_name: str,
    profile: Dict[str, Any],
    test: TestCase,
    timeout: int,
    num_gpu: Optional[int] = None,
    repetition: int = 1,
    warm_state: Optional[str] = None,
    context_override: Optional[int] = None,
) -> TestResult:
    """Run one test against one profile and grade it.

    This is the shared per-test unit used by both the CLI run loop (`run_profile`)
    and the streaming `/benchmark/run` endpoint, so call + grading never diverge.
    It does not print PASS/FAIL or sample VRAM - those stay with the caller.
    ``num_gpu`` pins the device on the inference call (0 = CPU) for forced runs.
    ``context_override`` pins a per-call context length for context-scaling sweeps.
    """
    started = time.perf_counter()
    data = call_chat(base_url, profile_name, profile, test, timeout, num_gpu=num_gpu, context_override=context_override)
    elapsed = float(data.get("elapsed_seconds") or (time.perf_counter() - started))
    raw_resp = data.get("response")
    response_text = raw_resp if isinstance(raw_resp, str) else json.dumps(raw_resp or "", ensure_ascii=False)
    response_text = str(response_text or "")
    success = bool(data.get("success"))
    try:
        accuracy = test.grader(response_text) if success else 0.0
    except Exception as exc:  # grader bug should not crash the run
        accuracy = 0.0
        print(f"  WARN grader for {test.name} raised: {exc}", flush=True)
    metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
    native_tps = metrics.get("tokens_per_second")
    try:
        native_tps = float(native_tps) if native_tps is not None else None
    except (TypeError, ValueError):
        native_tps = None
    approx_tps = native_tps or (((len(response_text) / 4) / elapsed) if success and elapsed > 0 else None)

    return TestResult(
        name=test.name,
        category=test.category,
        success=success,
        elapsed_seconds=round(elapsed, 3),
        response_length=len(response_text),
        response_preview=response_text[:240].replace("\n", " "),
        accuracy=round(accuracy, 3),
        error=data.get("error"),
        warning=data.get("warning"),
        approx_tokens_per_second=round(approx_tps, 2) if approx_tps is not None else None,
        repetition=repetition,
        metrics=metrics,
        tokens_per_second_source="backend" if native_tps is not None else "estimated",
        warm_state=warm_state,
        context_limit_used=data.get("context_limit_used"),
    )


def run_profile(base_url: str, profile_name: str, profile: Dict[str, Any], tests: List[TestCase], timeout: int) -> ProfileResult:
    print(f"\n=== {profile_name} ({profile.get('model_id')}) ===", flush=True)
    vram_before = nvidia_smi_used_mb()
    print(f"  VRAM before: {vram_before} MB" if vram_before is not None else "  VRAM before: unavailable", flush=True)

    result = ProfileResult(
        profile=profile_name,
        model_id=str(profile.get("model_id") or ""),
        backend=str(profile.get("backend") or ""),
        enabled=bool(profile.get("enabled", False)),
        recommended_for_8gb_vram=profile.get("recommended_for_8gb_vram"),
        vram_before_mb=vram_before,
        vram_after_mb=None,
        vram_peak_mb=vram_before,
    )

    consecutive_fail = 0
    for test in tests:
        item = execute_test(base_url, profile_name, profile, test, timeout)

        vram_now = nvidia_smi_used_mb()
        if vram_now is not None and (result.vram_peak_mb is None or vram_now > result.vram_peak_mb):
            result.vram_peak_mb = vram_now

        result.tests.append(item)

        status = "PASS" if item.success else "FAIL"
        acc_label = f"acc={item.accuracy:.2f}" if item.success else "acc=n/a"
        tps_label = f"~{item.approx_tokens_per_second:.1f} tok/s" if item.approx_tokens_per_second else ""
        err_label = f" ({item.error})" if not item.success else ""
        print(
            f"  [{status}] {test.category:14} {test.name:26} {item.elapsed_seconds:6.2f}s  {acc_label}  {tps_label}{err_label}",
            flush=True,
        )

        if not item.success:
            consecutive_fail += 1
            if _is_oom(item.error):
                result.fits_in_vram = False
                result.notes.append(f"OOM-like failure on {test.name}")
            if _is_not_pulled(item.error):
                result.notes.append(f"Model not pulled: {item.error}")
                break
            if consecutive_fail >= 4:
                result.notes.append(f"Aborted after 4 consecutive failures starting at {test.name}.")
                break
        else:
            consecutive_fail = 0

    result.vram_after_mb = nvidia_smi_used_mb()
    print(f"  VRAM after:  {result.vram_after_mb} MB  peak: {result.vram_peak_mb} MB", flush=True)
    return result


def iter_run(
    base_url: str,
    profiles_map: Dict[str, Any],
    selected: List[str],
    tests: List[TestCase],
    timeout: int,
    num_gpu: Optional[int] = None,
    repetitions: int = 1,
    provenance: Optional[Dict[str, Any]] = None,
) -> Iterator[Dict[str, Any]]:
    """Stream a benchmark run as a sequence of event dicts.

    Shares `execute_test` with the CLI so grading is identical. Emits:
    run_start, profile_start, test_start/test_result(*), profile_aborted?,
    profile_end, run_end. Used by the streaming /benchmark/run endpoint. No printing, no
    VRAM tracking - the caller decides how to present events. ``num_gpu`` pins
    the device on each inference call (0 = CPU) for forced CPU/GPU runs.
    """
    yield {
        "event": "run_start",
        "profiles": list(selected),
        "test_count": len(tests) * repetitions,
        "unique_test_count": len(tests),
        "repetitions": repetitions,
        "tests": [t.name for t in tests],
        "categories": sorted({t.category for t in tests}),
        "provenance": provenance or {},
    }
    started = time.perf_counter()
    overall: List[Dict[str, Any]] = []
    for name in selected:
        profile = profiles_map[name]
        profile_provenance = ((provenance or {}).get("profiles") or {}).get(name, {})
        warm_state = profile_provenance.get("warm_state")
        yield {
            "event": "profile_start",
            "profile": name,
            "model_id": profile.get("model_id"),
            "provenance": profile_provenance,
        }
        consecutive_fail = 0
        prof: List[TestResult] = []
        abort = False
        for test in tests:
            repeated: List[TestResult] = []
            for repetition in range(1, repetitions + 1):
                yield {
                    "event": "test_start",
                    "profile": name,
                    "name": test.name,
                    "category": test.category,
                    "repetition": repetition,
                    "repetitions": repetitions,
                }
                item = execute_test(
                    base_url,
                    name,
                    profile,
                    test,
                    timeout,
                    num_gpu=num_gpu,
                )
                item.repetition = repetition
                item.warm_state = warm_state if repetition == 1 else "warm"
                prof.append(item)
                repeated.append(item)
                event = asdict(item)
                event["event"] = "test_result"
                event["profile"] = name
                event["repetitions"] = repetitions
                yield event

                if not item.success:
                    consecutive_fail += 1
                    if _is_not_pulled(item.error):
                        yield {
                            "event": "profile_aborted",
                            "profile": name,
                            "reason": f"model not pulled: {item.error}",
                        }
                        abort = True
                        break
                    if consecutive_fail >= 4:
                        yield {"event": "profile_aborted", "profile": name, "reason": "4 consecutive failures"}
                        abort = True
                        break
                else:
                    consecutive_fail = 0
            successful = [item for item in repeated if item.success]
            rates = [item.approx_tokens_per_second for item in successful if item.approx_tokens_per_second is not None]
            latency_stats = _stats_block([item.elapsed_seconds for item in successful], digits=3)
            tps_stats = _stats_block(rates, digits=2)
            yield {
                "event": "test_aggregate",
                "profile": name,
                "name": test.name,
                "runs": len(repeated),
                "successful_runs": len(successful),
                "mean_latency_seconds": latency_stats["mean"],
                "latency_stdev_seconds": latency_stats["stdev"],
                "latency_p90_seconds": latency_stats["p90"],
                "latency_p95_seconds": latency_stats["p95"],
                "mean_tokens_per_second": tps_stats["mean"],
                "tokens_per_second_stdev": tps_stats["stdev"],
                "tokens_per_second_p90": tps_stats["p90"],
                "accuracy_stdev": round(statistics.stdev([item.accuracy for item in repeated]), 3)
                if len(repeated) > 1
                else 0.0,
            }
            if abort:
                break

        passed = sum(1 for t in prof if t.success)
        avg_acc = round(statistics.mean([t.accuracy for t in prof]), 3) if prof else 0.0
        successful_rates = [t.approx_tokens_per_second for t in prof if t.success and t.approx_tokens_per_second is not None]
        successful_latencies = [t.elapsed_seconds for t in prof if t.success]
        summary = {"tests": len(prof), "passed": passed, "avg_accuracy": avg_acc}
        if repetitions > 1 or provenance is not None:
            latency_stats = _stats_block(successful_latencies, digits=3)
            tps_stats = _stats_block(successful_rates, digits=2)
            summary.update(
                {
                    "unique_tests": len({t.name for t in prof}),
                    "repetitions": repetitions,
                    "latency_stdev_seconds": latency_stats["stdev"],
                    "latency_p90_seconds": latency_stats["p90"],
                    "latency_p95_seconds": latency_stats["p95"],
                    "tokens_per_second_stdev": tps_stats["stdev"],
                    "tokens_per_second_p90": tps_stats["p90"],
                }
            )
        overall.append({"profile": name, **summary})
        yield {"event": "profile_end", "profile": name, "summary": summary}

    yield {
        "event": "run_end",
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "profiles": overall,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def category_summary(tests: List[TestResult]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for cat in {t.category for t in tests}:
        subset = [t for t in tests if t.category == cat]
        successes = [t for t in subset if t.success]
        elapsed = [t.elapsed_seconds for t in successes]
        accs = [t.accuracy for t in subset]
        out[cat] = {
            "n": len(subset),
            "passed": len(successes),
            "avg_elapsed_seconds": round(statistics.mean(elapsed), 3) if elapsed else 0.0,
            "avg_accuracy": round(statistics.mean(accs), 3) if accs else 0.0,
        }
    return out


def write_reports(results: List[ProfileResult], run_meta: Dict[str, Any]) -> Tuple[Path, Path]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    json_path = REPORTS_DIR / f"benchmark_{stamp}.json"
    md_path = REPORTS_DIR / f"benchmark_{stamp}.md"

    payload = {
        "meta": run_meta,
        "results": [
            {**asdict(p), "tests": [asdict(t) for t in p.tests], "category_summary": category_summary(p.tests)}
            for p in results
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines: List[str] = [
        f"# LocalDeploy benchmark v2 - {stamp}",
        "",
        f"- GPU: `{run_meta.get('gpu_name')}` ({run_meta.get('gpu_total_mb')} MB total)",
        f"- API: `{run_meta.get('base_url')}`",
        f"- Test cases per profile: {run_meta.get('test_count')}",
        f"- Total elapsed: {run_meta.get('elapsed_seconds')}s",
        "",
        "## Categories",
        "",
        "- **planning** - software-project decomposition (3 tests)",
        "- **classification** - multi-class label with ambiguity (5 tests)",
        "- **code** - non-trivial Python + SQL, graded by AST + unit tests (4 tests)",
        "- **math** - multi-step arithmetic, algebra, probability, matrices (5 tests)",
        "- **structured** - JSON-schema-shaped output, field-level grading (4 tests)",
        "- **structured_hard** - pydantic-schema output: classification, plan orchestration, "
        "approval gate, multi-task extraction (4 tests)",
        "",
        "## Overall scoreboard",
        "",
        "| Profile | Model | Pass | Avg accuracy | Avg latency | Peak VRAM | Verdict |",
        "|---|---|---:|---:|---:|---:|---|",
    ]

    def overall_acc(p: ProfileResult) -> float:
        accs = [t.accuracy for t in p.tests if t.success]
        return statistics.mean(accs) if accs else 0.0

    for p in sorted(results, key=lambda x: (-overall_acc(x), -sum(1 for t in x.tests if t.success))):
        passed = sum(1 for t in p.tests if t.success)
        total = len(p.tests)
        accs = [t.accuracy for t in p.tests if t.success]
        avg_acc = statistics.mean(accs) if accs else 0.0
        lat = [t.elapsed_seconds for t in p.tests if t.success]
        avg_lat = statistics.mean(lat) if lat else 0.0
        peak = p.vram_peak_mb or 0
        if not p.fits_in_vram:
            verdict = "OOM"
        elif passed == 0:
            verdict = "not pulled / unreachable"
        elif passed == total:
            verdict = "fits"
        else:
            verdict = f"partial ({passed}/{total})"
        lines.append(
            f"| `{p.profile}` | `{p.model_id}` | {passed}/{total} | "
            f"{avg_acc:.2f} | {avg_lat:.2f}s | {peak} MB | {verdict} |"
        )

    # Category-by-model breakdown. Categories are derived from TEST_CASES (not
    # hardcoded) so a newly added category always shows up here instead of
    # silently being dropped from the scoreboard while still scored elsewhere.
    category_order = list(dict.fromkeys(t.category for t in TEST_CASES))
    lines.extend(
        [
            "",
            "## Per-category accuracy (avg, per profile)",
            "",
            "| Profile | " + " | ".join(category_order) + " |",
            "|---|" + "---:|" * len(category_order),
        ]
    )
    for p in results:
        cs = category_summary(p.tests)
        cells = [f"`{p.profile}`"]
        for cat in category_order:
            s = cs.get(cat, {"avg_accuracy": 0.0, "n": 0})
            cells.append(f"{s['avg_accuracy']:.2f}" if s.get("n", 0) else "-")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    for p in results:
        lines.append(f"## `{p.profile}`")
        lines.append("")
        lines.append(f"- Model: `{p.model_id}` ({p.backend}) - recommended_for_8gb_vram: `{p.recommended_for_8gb_vram}`")
        lines.append(f"- VRAM: before {p.vram_before_mb} MB, peak {p.vram_peak_mb} MB, after {p.vram_after_mb} MB")
        if p.notes:
            for note in p.notes:
                lines.append(f"- Note: {note}")
        lines.append("")
        cs = category_summary(p.tests)
        lines.append("| Category | n | passed | avg latency | avg accuracy |")
        lines.append("|---|---:|---:|---:|---:|")
        for cat, s in sorted(cs.items()):
            lines.append(f"| {cat} | {int(s['n'])} | {int(s['passed'])} | {s['avg_elapsed_seconds']}s | {s['avg_accuracy']} |")
        lines.append("")
        lines.append("| Test | Result | Latency | tok/s | Accuracy | Response preview |")
        lines.append("|---|---|---:|---:|---:|---|")
        for t in p.tests:
            status = "PASS" if t.success else "FAIL"
            tps = f"{t.approx_tokens_per_second}" if t.approx_tokens_per_second else "-"
            preview = (t.response_preview or t.error or "").replace("|", "\\|")[:160]
            lines.append(f"| {t.name} | {status} | {t.elapsed_seconds}s | {tps} | {t.accuracy} | {preview} |")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def gpu_info() -> Dict[str, Any]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT,
            timeout=5,
        ).decode("ascii", errors="ignore").strip().splitlines()[0]
        name, total = [x.strip() for x in output.split(",", 1)]
        return {"gpu_name": name, "gpu_total_mb": int(total)}
    except Exception:
        return {"gpu_name": None, "gpu_total_mb": None}


# ---------------------------------------------------------------------------
# Question-set grader registry (safe, JSON-driven graders for uploaded sets)
# ---------------------------------------------------------------------------

GRADER_TYPES = [
    "contains_all",
    "json_array_min_len",
    "number_within",
    "exact_match",
    "classification_set",
    "json_keys_present",
]


def build_grader(spec: Any) -> Callable[[str], float]:
    """Build a grader callable from a JSON spec. Raises ValueError on a bad spec.

    Keeps uploaded question sets safe JSON (no arbitrary code): each grader is
    selected by `type` from a fixed registry and reuses the helpers already in
    this module (`_extract_json`, `_grade_number`).
    """
    if not isinstance(spec, dict):
        raise ValueError("grader must be an object with a 'type' field")
    gtype = spec.get("type")

    if gtype == "contains_all":
        keywords = spec.get("keywords")
        if not isinstance(keywords, list) or not keywords:
            raise ValueError("contains_all requires a non-empty 'keywords' list")
        case_sensitive = bool(spec.get("case_sensitive", False))
        needles = [str(k) if case_sensitive else str(k).lower() for k in keywords]

        def grade_contains(text: str) -> float:
            hay = text if case_sensitive else text.lower()
            hits = sum(1 for needle in needles if needle in hay)
            return hits / len(needles)

        return grade_contains

    if gtype == "json_array_min_len":
        minimum = spec.get("min", 1)
        if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 0:
            raise ValueError("json_array_min_len requires a non-negative integer 'min'")

        def grade_array(text: str) -> float:
            data = _extract_json(text)
            return 1.0 if isinstance(data, list) and len(data) >= minimum else 0.0

        return grade_array

    if gtype == "number_within":
        if "expected" not in spec:
            raise ValueError("number_within requires 'expected'")
        try:
            expected = float(spec["expected"])
            tolerance = float(spec.get("tolerance", 0.5))
        except (TypeError, ValueError):
            raise ValueError("number_within 'expected'/'tolerance' must be numbers")
        return _grade_number(expected, tolerance)

    if gtype == "exact_match":
        if "expected" not in spec:
            raise ValueError("exact_match requires 'expected'")
        expected_str = str(spec["expected"])
        case_sensitive = bool(spec.get("case_sensitive", False))

        def grade_exact(text: str) -> float:
            got, want = text.strip(), expected_str.strip()
            if not case_sensitive:
                got, want = got.lower(), want.lower()
            return 1.0 if got == want else 0.0

        return grade_exact

    if gtype == "classification_set":
        expected = spec.get("expected")
        if not isinstance(expected, list):
            raise ValueError("classification_set requires an 'expected' list")
        expected_set = {str(x).strip().lower() for x in expected}

        def grade_set(text: str) -> float:
            data = _extract_json(text)
            if isinstance(data, list):
                got = {str(x).strip().lower() for x in data}
            else:
                got = {p.strip().lower() for p in re.split(r"[,\n]", text) if p.strip()}
            return 1.0 if got == expected_set else 0.0

        return grade_set

    if gtype == "json_keys_present":
        required = spec.get("required")
        if not isinstance(required, list) or not required:
            raise ValueError("json_keys_present requires a non-empty 'required' list")
        if len(required) > 100:
            raise ValueError("json_keys_present accepts at most 100 required keys")
        if any(not isinstance(key, str) or not key.strip() or len(key) > 256 for key in required):
            raise ValueError("json_keys_present keys must be non-empty strings up to 256 characters")
        if len(set(required)) != len(required):
            raise ValueError("json_keys_present 'required' keys must be unique")
        allow_extra = spec.get("allow_extra", True)
        if not isinstance(allow_extra, bool):
            raise ValueError("json_keys_present 'allow_extra' must be a boolean")
        required_keys = set(required)

        def grade_keys(text: str) -> float:
            data = _extract_json(text)
            if not isinstance(data, dict):
                return 0.0
            actual_keys = set(data)
            if not required_keys.issubset(actual_keys):
                return 0.0
            if not allow_extra and actual_keys != required_keys:
                return 0.0
            return 1.0

        return grade_keys

    if gtype == "builtin":
        name = spec.get("name")
        if not isinstance(name, str) or name not in BUILTIN_GRADER_BY_NAME:
            raise ValueError("builtin grader requires a valid built-in test name")
        return BUILTIN_GRADER_BY_NAME[name]

    raise ValueError(f"unknown grader type '{gtype}' (allowed: {', '.join(GRADER_TYPES)})")


def validate_question_set(payload: Any) -> Dict[str, Any]:
    """Validate an uploaded question set against the schema and grader registry.

    Returns a structured report with per-row errors; never raises.
    """
    errors: List[Dict[str, Any]] = []
    if not isinstance(payload, dict):
        return {
            "success": True,
            "valid": False,
            "question_count": 0,
            "errors": [{"index": -1, "error": "top-level must be a JSON object"}],
            "grader_types": GRADER_TYPES,
        }

    questions = payload.get("questions")
    if not isinstance(questions, list) or not questions:
        errors.append({"index": -1, "error": "'questions' must be a non-empty list"})
        questions = []

    seen_names: set = set()
    for index, question in enumerate(questions):
        if not isinstance(question, dict):
            errors.append({"index": index, "error": "question must be an object"})
            continue
        name = question.get("name")
        if not name or not isinstance(name, str):
            errors.append({"index": index, "error": "missing or invalid 'name'"})
        elif name in seen_names:
            errors.append({"index": index, "name": name, "error": "duplicate 'name'"})
        else:
            seen_names.add(name)
        if not question.get("category") or not isinstance(question.get("category"), str):
            errors.append({"index": index, "name": question.get("name"), "error": "missing or invalid 'category'"})
        if not question.get("prompt") or not isinstance(question.get("prompt"), str):
            errors.append({"index": index, "name": question.get("name"), "error": "missing or invalid 'prompt'"})
        mot = question.get("max_output_tokens", 256)
        if not isinstance(mot, int) or isinstance(mot, bool) or mot <= 0:
            errors.append({"index": index, "name": question.get("name"), "error": "'max_output_tokens' must be a positive integer"})
        try:
            build_grader(question.get("grader"))
        except ValueError as exc:
            errors.append({"index": index, "name": question.get("name"), "error": f"grader: {exc}"})

    return {
        "success": True,
        "valid": len(errors) == 0,
        "question_count": len(questions),
        "errors": errors,
        "grader_types": GRADER_TYPES,
    }


def build_test_cases(payload: Dict[str, Any]) -> List[TestCase]:
    """Turn a validated question set into TestCase objects for a run."""
    cases: List[TestCase] = []
    for question in payload.get("questions", []):
        cases.append(
            TestCase(
                name=str(question["name"]),
                category=str(question.get("category", "custom")),
                prompt=str(question["prompt"]),
                grader=build_grader(question.get("grader")),
                grader_explainer=str(question.get("grader_explainer", "")),
                max_output_tokens=int(question.get("max_output_tokens", 256)),
            )
        )
    return cases


GRADER_TYPES = tuple(dict.fromkeys([*GRADER_TYPES, "builtin"]))


BUILTIN_DATA_QUESTIONS: List[Dict[str, Any]] = [
    {
        "name": "plan_incident_triage_array",
        "category": "planning",
        "prompt": (
            "A checkout API has a sudden error-rate spike. Return a JSON array containing at least four "
            "ordered triage steps, starting with the cheapest observation. JSON only."
        ),
        "max_output_tokens": 300,
        "grader": {"type": "json_array_min_len", "min": 4},
        "grader_explainer": "Valid JSON array with at least four ordered triage steps.",
    },
    {
        "name": "cls_incident_signals",
        "category": "classification",
        "prompt": (
            "A service returns HTTP 503 for 30% of requests and its p95 response time doubled. "
            "Choose every matching label from [availability, latency, correctness, security]. "
            "Return a JSON array of labels only."
        ),
        "max_output_tokens": 40,
        "grader": {"type": "classification_set", "expected": ["availability", "latency"]},
        "grader_explainer": "Label set is exactly availability and latency.",
    },
    {
        "name": "code_merge_sort_complexity",
        "category": "code",
        "prompt": (
            "State the worst-case time complexity and auxiliary space complexity of merge sort on an array. "
            "Use Big O notation and include no implementation."
        ),
        "max_output_tokens": 80,
        "grader": {"type": "contains_all", "keywords": ["O(n log n)", "O(n)"]},
        "grader_explainer": "Contains O(n log n) time and O(n) auxiliary space.",
    },
    {
        "name": "math_percentage_increase",
        "category": "math",
        "prompt": "A latency value rises from 80 ms to 100 ms. What is the percentage increase? Reply with the number only.",
        "max_output_tokens": 32,
        "grader": {"type": "number_within", "expected": 25, "tolerance": 0.1},
        "grader_explainer": "Answer is within 0.1 of 25.",
    },
    {
        "name": "json_release_metadata_keys",
        "category": "structured",
        "prompt": (
            "Return a JSON object describing a software release with keys version, date, changes, and breaking. "
            "Use an array for changes and a boolean for breaking. JSON only."
        ),
        "max_output_tokens": 180,
        "grader": {
            "type": "json_keys_present",
            "required": ["version", "date", "changes", "breaking"],
        },
        "grader_explainer": "JSON object contains version, date, changes, and breaking keys.",
    },
]


BUILTIN_GRADER_BY_NAME = {test.name: test.grader for test in TEST_CASES}
BUILTIN_GRADER_BY_NAME.update(
    {question["name"]: build_grader(question["grader"]) for question in BUILTIN_DATA_QUESTIONS}
)


BUILTIN_QUESTION_SET: Dict[str, Any] = {
    "version": 1,
    "questions": [
        {
            "name": test.name,
            "category": test.category,
            "prompt": test.prompt,
            "max_output_tokens": test.max_output_tokens,
            "grader": {"type": "builtin", "name": test.name},
            "grader_explainer": test.grader_explainer,
        }
        for test in TEST_CASES
    ] + [dict(question) for question in BUILTIN_DATA_QUESTIONS],
}


def builtin_test_cases() -> List[TestCase]:
    """Return a fresh set of every built-in benchmark case.

    Building new objects for each caller keeps request-specific token limits
    and filtering from mutating the shared grader definitions.
    """
    return build_test_cases(BUILTIN_QUESTION_SET)


EXAMPLE_QUESTION_SET: Dict[str, Any] = {
    "version": 1,
    "questions": [
        {
            "name": "planning_triage_basic",
            "category": "planning",
            "prompt": "List 3 first steps to triage a service outage. Return a JSON array of strings.",
            "max_output_tokens": 512,
            "grader": {"type": "json_array_min_len", "min": 3},
            "grader_explainer": "Passes if the model returns a JSON array with at least 3 steps.",
        },
        {
            "name": "math_tolerance",
            "category": "reasoning",
            "prompt": "What is 12.5% of 240? Answer with the number only.",
            "max_output_tokens": 32,
            "grader": {"type": "number_within", "expected": 30, "tolerance": 0.5},
            "grader_explainer": "Passes if the parsed number is within 0.5 of 30.",
        },
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser(description="LocalDeploy capability benchmark (v2).")
    parser.add_argument("--profile", action="append", help="Limit to one or more named profiles (repeatable).")
    parser.add_argument("--max-output-tokens", type=int, help="Bump all non-classification tests to this cap (if higher).")
    parser.add_argument("--timeout", type=int, default=240, help="Per-request timeout seconds.")
    parser.add_argument("--skip-categories", help="Comma-separated categories to skip.")
    parser.add_argument("--include-categories", help="Comma-separated categories to ONLY include (whitelist).")
    args = parser.parse_args()

    config = load_config()
    profiles = config.get("profiles", {})
    selected = args.profile or [name for name, p in profiles.items() if p.get("enabled", False)]
    selected = [name for name in selected if name in profiles]
    if not selected:
        print("No profiles selected. Enable some in config.json or pass --profile NAME.")
        return 2

    skip = {c.strip() for c in (args.skip_categories or "").split(",") if c.strip()}
    include = {c.strip() for c in (args.include_categories or "").split(",") if c.strip()}
    tests = [
        t
        for t in builtin_test_cases()
        if t.category not in skip and (not include or t.category in include)
    ]
    if args.max_output_tokens:
        for t in tests:
            if t.category != "classification":
                t.max_output_tokens = max(t.max_output_tokens, args.max_output_tokens)

    base_url = api_base_url()
    try:
        health = requests.get(f"{base_url}/health", headers=api_auth_headers(), timeout=10).json()
    except Exception as exc:
        print(f"API server is not reachable at {base_url}: {exc}")
        print("Start it with: .\\scripts\\start.ps1  (or set $env:API_PORT='8011' first)")
        return 2
    if not health.get("server"):
        print(f"API server at {base_url} returned unexpected health: {health}")
        return 2

    print(f"Profiles selected ({len(selected)}): {', '.join(selected)}")
    print(f"Tests per profile: {len(tests)}")
    print(f"Timeout: {args.timeout}s per request")

    info = gpu_info()
    run_meta = {
        "base_url": base_url,
        "default_profile": health.get("default_profile"),
        "test_count": len(tests),
        "tests": [t.name for t in tests],
        "categories": sorted({t.category for t in tests}),
        **info,
    }

    started = time.perf_counter()
    results: List[ProfileResult] = []
    for name in selected:
        result = run_profile(base_url, name, profiles[name], tests, args.timeout)
        results.append(result)

    elapsed = time.perf_counter() - started
    run_meta["elapsed_seconds"] = round(elapsed, 2)

    json_path, md_path = write_reports(results, run_meta)
    print(f"\nReports written:\n  {json_path}\n  {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
