"""Compare decision models on a labeled routing set, through a running LocalDeploy.

    python scripts/decision_bench.py --profile d1_omni_600m_decision --profile laya_decision
    python scripts/decision_bench.py --url http://127.0.0.1:8000 --profile d1_3b_decision --out reports/decision.json

For each message it asks one choice question ("which route?") at POST /v1/systemone and
reports accuracy, the two mistakes that matter to a bot (a real task answered as chat, and
chat turned into a task), whether the confidences can be trusted, how many messages a
confidence cut-off would settle without a language model, and latency.
Standard library plus ``requests`` only.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import requests

DEFAULT_DATASET = Path(__file__).resolve().parent.parent / "examples" / "decision" / "routing_eval.json"
CHAT_ROUTE = "conversation"


def ask(url: str, profile: str, state: str, question: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    started = time.perf_counter()
    response = requests.post(
        f"{url}/v1/systemone",
        json={"model": profile, "state": state, "questions": {"q": question}},
        timeout=timeout,
    )
    wall_ms = (time.perf_counter() - started) * 1000
    if response.status_code != 200:
        raise RuntimeError(f"{profile}: HTTP {response.status_code}: {response.text[:300]}")
    body = response.json()
    answer = body["answers"]["q"]
    probabilities = answer.get("probabilities")
    if isinstance(probabilities, dict):
        top = max(probabilities.values()) if probabilities else None
    else:
        top = None
    return {
        "choice": answer["choice"],
        "confidence": float(answer.get("confidence", top if top is not None else 0.0)),
        "probabilities": probabilities,
        "server_ms": float(body.get("latency_ms", wall_ms)),
        "wall_ms": wall_ms,
        "load_ms": body.get("load_ms"),
    }


def ece(pairs: List[tuple], bins: int = 10) -> float:
    """Expected calibration error: how far stated confidence is from actual accuracy."""
    if not pairs:
        return 0.0
    total = len(pairs)
    error = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [(c, ok) for c, ok in pairs if low <= c < high or (index == bins - 1 and c == 1.0)]
        if members:
            confidence = sum(c for c, _ in members) / len(members)
            accuracy = sum(1 for _, ok in members if ok) / len(members)
            error += len(members) / total * abs(confidence - accuracy)
    return error


def coverage_at(pairs: List[tuple], target: float) -> Dict[str, Any]:
    """Largest share of messages a confidence cut-off can settle while staying at or above ``target`` accuracy."""
    best = {"coverage": 0.0, "threshold": None, "accuracy": None}
    for threshold in [t / 100 for t in range(30, 100)]:
        answered = [ok for c, ok in pairs if c >= threshold]
        if not answered:
            continue
        accuracy = sum(answered) / len(answered)
        coverage = len(answered) / len(pairs)
        if accuracy >= target and coverage > best["coverage"]:
            best = {"coverage": coverage, "threshold": threshold, "accuracy": accuracy}
    return best


def percentile(values: List[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def evaluate(url: str, profile: str, dataset: Dict[str, Any], warmup: int, timeout: float, log, limit: int = 0) -> Dict[str, Any]:
    question = dataset["route_question"]
    items = dataset["items"][:limit] if limit else dataset["items"]
    first_load = None
    for _ in range(max(1, warmup)):
        warm = ask(url, profile, "hello", question, timeout)
        first_load = first_load or warm.get("load_ms")
    rows = []
    for index, item in enumerate(items, 1):
        result = ask(url, profile, item["text"], question, timeout)
        result.update(text=item["text"], expected=item["route"], correct=result["choice"] == item["route"])
        rows.append(result)
        if index % 25 == 0:
            log(f"  {profile}: {index}/{len(items)}")
    pairs = [(r["confidence"], r["correct"]) for r in rows]
    right = [r["confidence"] for r in rows if r["correct"]]
    wrong = [r["confidence"] for r in rows if not r["correct"]]
    dropped = [r for r in rows if r["expected"] != CHAT_ROUTE and r["choice"] == CHAT_ROUTE]
    spurious = [r for r in rows if r["expected"] == CHAT_ROUTE and r["choice"] != CHAT_ROUTE]
    chat_vs_task_ok = sum(1 for r in rows if (r["expected"] == CHAT_ROUTE) == (r["choice"] == CHAT_ROUTE))
    server = [r["server_ms"] for r in rows]
    per_route: Dict[str, Dict[str, int]] = {}
    for r in rows:
        stats = per_route.setdefault(r["expected"], {"n": 0, "ok": 0})
        stats["n"] += 1
        stats["ok"] += int(r["correct"])
    confusions = Counter((r["expected"], r["choice"]) for r in rows if not r["correct"])
    return {
        "profile": profile,
        "n": len(rows),
        "accuracy": sum(r["correct"] for r in rows) / len(rows),
        "chat_vs_task_accuracy": chat_vs_task_ok / len(rows),
        "dropped_tasks": len(dropped),
        "spurious_tasks": len(spurious),
        "mean_confidence_right": statistics.mean(right) if right else None,
        "mean_confidence_wrong": statistics.mean(wrong) if wrong else None,
        "ece": ece(pairs),
        "coverage_at_95": coverage_at(pairs, 0.95),
        "coverage_at_90": coverage_at(pairs, 0.90),
        "latency_ms": {"median": statistics.median(server), "p95": percentile(server, 0.95), "max": max(server)},
        "first_load_ms": first_load,
        "per_route": {k: v["ok"] / v["n"] for k, v in sorted(per_route.items())},
        "top_confusions": [{"expected": e, "predicted": p, "count": c} for (e, p), c in confusions.most_common(8)],
        "dropped_examples": [{"text": r["text"], "expected": r["expected"], "confidence": round(r["confidence"], 3)} for r in dropped[:8]],
        "rows": rows,
    }


def evaluate_steering(url: str, profile: str, dataset: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    question = dataset["steering_question"]
    ok = 0
    misses = []
    items = dataset["steering_items"]
    for item in items:
        state = f"Running now: {item['running']}\nWaiting on the user's answer to: {item['pending']}\nUser's new message: {item['text']}"
        result = ask(url, profile, state, question, timeout)
        if result["choice"] == item["label"]:
            ok += 1
        else:
            misses.append({"text": item["text"], "expected": item["label"], "predicted": result["choice"]})
    return {"n": len(items), "accuracy": ok / len(items), "misses": misses}


def markdown(results: List[Dict[str, Any]]) -> str:
    lines = [
        "| Profile | Route acc. | Chat vs task | Dropped tasks | Spurious tasks | ECE | Settled at 95% | Median ms | p95 ms | Steering acc. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        cov = r["coverage_at_95"]
        steering = r.get("steering", {}).get("accuracy")
        lines.append(
            "| {profile} | {acc:.0%} | {cvt:.0%} | {dropped} | {spurious} | {ece:.3f} | {cov:.0%} (t>={thr}) | {med:.0f} | {p95:.0f} | {steer} |".format(
                profile=r["profile"], acc=r["accuracy"], cvt=r["chat_vs_task_accuracy"], dropped=r["dropped_tasks"],
                spurious=r["spurious_tasks"], ece=r["ece"], cov=cov["coverage"], thr=cov["threshold"] if cov["threshold"] else "-",
                med=r["latency_ms"]["median"], p95=r["latency_ms"]["p95"], steer=f"{steering:.0%}" if steering is not None else "-",
            )
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--profile", action="append", required=True, help="decision profile (repeatable)")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--warmup", type=int, default=2, help="untimed requests before measuring")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--limit", type=int, default=0, help="only the first N messages, and no steering set (a quick timing sample)")
    parser.add_argument("--out", help="write the full JSON report here")
    args = parser.parse_args()

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    log = lambda message: print(message, file=sys.stderr, flush=True)  # noqa: E731
    results = []
    for profile in args.profile:
        log(f"{profile}: warming up")
        try:
            result = evaluate(args.url.rstrip("/"), profile, dataset, args.warmup, args.timeout, log, args.limit)
            result["steering"] = (
                {"n": 0, "accuracy": 0.0, "misses": []} if args.limit else evaluate_steering(args.url.rstrip("/"), profile, dataset, args.timeout)
            )
        except (RuntimeError, requests.RequestException) as exc:
            log(f"{profile}: FAILED - {exc}")
            results.append({"profile": profile, "error": str(exc)})
            continue
        results.append(result)
    scored = [r for r in results if "error" not in r]
    if scored:
        print(markdown(scored))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"dataset": dataset["name"], "results": results}, indent=2), encoding="utf-8")
        log(f"report written to {args.out}")
    return 0 if len(scored) == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
