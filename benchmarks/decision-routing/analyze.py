"""Recompute summary.csv and analysis.json from the published runs.

    python benchmarks/decision-routing/analyze.py

Reads runs/2026-10-29-validation: the repeated and held-out decision-model runs, start-up, idle and under-load tests,
and the LLM on both sets. Prints the numbers the README uses: 95% Wilson intervals, an exact McNemar test of d3-lite
against the LLM on the same messages, and a replay of YBM's routing modes using each model's recorded answer and time.
Standard library only.
"""
import csv
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
VAL = HERE / "runs" / "2026-10-29-validation"
OUT = HERE
LLM_MAIN = VAL / "llm-main.json"
LLM_HELDOUT = VAL / "llm-heldout.json"
CASCADE_ROUTES = {"filesystem.manage", "desktop.observe", "document.manage", "code.interpreter", "workspace.manage",
                  "adapter.factory", "browser.open"}
CONCIERGE_ONLY = {"coding.agent"}
CHAT = "conversation"


def wilson(k, n, z=1.96):
    if not n:
        return [0.0, 0.0]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value: b = only A right, c = only B right."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def bench_runs():
    runs = []
    for path in sorted(VAL.glob("*-cpu-*.json")) + sorted(VAL.glob("*-cuda-*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "run" not in data:
            continue
        runs.append((path.stem, data["run"], data["results"][0]))
    return runs


def accepted(row):
    return row.get("accepted") or [row["expected"]]


def simulate(decision_rows, llm_rows, mode, threshold=0.9):
    """Route each message as YBM would under `mode`, with the LLM's real answer and time where it decides."""
    llm_by_text = {r["text"]: r for r in llm_rows}
    right, alone, alone_right, times = 0, 0, 0, []
    for d in decision_rows:
        llm = llm_by_text[d["text"]]
        ok_routes = accepted(d)
        if mode == "llm":
            use_decision = False
        else:
            sure = d["confidence"] >= threshold
            use_decision = sure and (d["choice"] in CASCADE_ROUTES if mode == "cascade" else d["choice"] not in CONCIERGE_ONLY)
        if use_decision:
            alone += 1
            alone_right += d["choice"] in ok_routes
            right += d["choice"] in ok_routes
            times.append(d["wall_ms"])
        else:
            right += llm["choice"] in ok_routes
            times.append(llm["ms"] + (0 if mode == "llm" else d["wall_ms"]))
    n = len(decision_rows)
    return {
        "mode": mode, "threshold": threshold, "n": n, "accuracy": right / n, "accuracy_ci95": wilson(right, n),
        "decided_alone": alone, "decided_alone_share": alone / n,
        "alone_accuracy": alone_right / alone if alone else None,
        "median_s": statistics.median(times) / 1000, "mean_s": statistics.mean(times) / 1000,
    }


def paired(decision_rows, llm_rows):
    llm_by_text = {r["text"]: r for r in llm_rows}
    b = c = both = neither = 0
    for d in decision_rows:
        l_ok = llm_by_text[d["text"]]["choice"] in accepted(d)
        d_ok = d["choice"] in accepted(d)
        both += d_ok and l_ok
        neither += (not d_ok) and (not l_ok)
        b += d_ok and not l_ok
        c += l_ok and not d_ok
    return {"both_right": both, "only_decision_right": b, "only_llm_right": c, "both_wrong": neither,
            "mcnemar_p": mcnemar_exact(b, c)}


def main():
    runs = bench_runs()
    analysis = {"runs": []}
    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run", "model_file", "device", "threads", "set", "tag", "n", "route_accuracy", "ci95_low", "ci95_high",
                    "chat_vs_task", "dropped_tasks", "spurious_tasks", "ece", "settled_at_95", "median_ms", "p95_ms",
                    "steering_accuracy", "server_ready_ms", "first_request_ms", "working_set_mb"])
        for name, run, r in runs:
            lo, hi = r.get("accuracy_ci95") or wilson(sum(x["correct"] for x in r["rows"]), r["n"])
            steering = (r.get("steering") or {}).get("accuracy")
            row = [name, run["model_file"], run["device"], run["threads"], run["set"], run["tag"], r["n"],
                   round(r["accuracy"], 4), round(lo, 4), round(hi, 4), round(r["chat_vs_task_accuracy"], 4),
                   r["dropped_tasks"], r["spurious_tasks"], round(r["ece"], 4), round(r["coverage_at_95"]["coverage"], 4),
                   round(r["latency_ms"]["median"]), round(r["latency_ms"]["p95"]),
                   round(steering, 4) if steering is not None else "", round(run["server_healthy_ms"]),
                   round(run["first_request_ms"]), run["memory_after_load"]["working_set_mb"]]
            w.writerow(row)
            analysis["runs"].append(dict(zip(["run", "model_file", "device", "threads", "set", "tag", "n", "accuracy", "lo", "hi"], row)))
            print(f"{name:<45} {r['accuracy']:.1%} ({lo:.0%}-{hi:.0%})  cvt {r['chat_vs_task_accuracy']:.0%}  "
                  f"drop {r['dropped_tasks']}  spur {r['spurious_tasks']}  med {r['latency_ms']['median']:.0f} ms  "
                  f"p95 {r['latency_ms']['p95']:.0f}  ready {run['server_healthy_ms']:.0f} ms  first {run['first_request_ms']:.0f} ms  "
                  f"ws {run['memory_after_load']['working_set_mb']} MB")

    # Repeats: identical answers?
    by_config = {}
    for name, run, r in runs:
        by_config.setdefault((run["model_file"], run["device"], run["set"]), []).append((run["tag"], r))
    analysis["repeats"] = []
    for (model, device, dset), group in by_config.items():
        if len(group) < 2:
            continue
        rows = [r["rows"] for _, r in group]
        same = all(len({rr[i]["choice"] for rr in rows}) == 1 for i in range(len(rows[0])))
        medians = [r["latency_ms"]["median"] for _, r in group]
        analysis["repeats"].append({"model": model, "device": device, "set": dset, "runs": len(group),
                                    "identical_choices": same, "medians_ms": medians})
        print(f"repeat {model} {device} {dset}: {len(group)} runs, identical choices {same}, medians {[round(m) for m in medians]}")

    # Paired comparison and routing modes, main set (d3-lite Q4 short wording, run r1) against the LLM's own run.
    llm_main = json.loads(LLM_MAIN.read_text(encoding="utf-8"))["rows"]
    d_main = next(r for name, run, r in runs if run["model_file"] == "vllm-d3-lite-Q4_K_M.gguf" and run["set"] == "main_short"
                  and run["tag"] == "r1" and run["device"] == "cpu")["rows"]
    assert {r["text"] for r in d_main} == {r["text"] for r in llm_main}
    llm_right = sum(r["choice"] in accepted(r) for r in llm_main)
    analysis["llm_main"] = {"n": len(llm_main), "accuracy": llm_right / len(llm_main), "ci95": wilson(llm_right, len(llm_main)),
                            "median_s": statistics.median(r["ms"] for r in llm_main) / 1000}
    analysis["paired_main"] = paired(d_main, llm_main)
    analysis["modes_main"] = [simulate(d_main, llm_main, m) for m in ("llm", "cascade", "decision")]
    print("LLM main:", json.dumps(analysis["llm_main"]))
    print("paired main:", json.dumps(analysis["paired_main"]))
    for m in analysis["modes_main"]:
        print("mode main:", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()}))

    if LLM_HELDOUT.exists():
        llm_held = json.loads(LLM_HELDOUT.read_text(encoding="utf-8"))["rows"]
        d_held = next(r for name, run, r in runs if run["model_file"] == "vllm-d3-lite-Q4_K_M.gguf"
                      and run["set"] == "heldout_short" and run["device"] == "cpu")["rows"]
        held_right = sum(r["correct"] for r in llm_held)
        analysis["llm_heldout"] = {"n": len(llm_held), "accuracy": held_right / len(llm_held),
                                   "ci95": wilson(held_right, len(llm_held)),
                                   "median_s": statistics.median(r["ms"] for r in llm_held) / 1000,
                                   "dropped_tasks": sum(1 for r in llm_held if CHAT not in r["accepted"] and r["choice"] == CHAT)}
        analysis["paired_heldout"] = paired(d_held, llm_held)
        analysis["modes_heldout"] = [simulate(d_held, llm_held, m) for m in ("llm", "cascade", "decision")]
        print("LLM heldout:", json.dumps(analysis["llm_heldout"]))
        print("paired heldout:", json.dumps(analysis["paired_heldout"]))
        for m in analysis["modes_heldout"]:
            print("mode heldout:", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()}))
        for r in llm_held:
            if not r["correct"]:
                print(f"  LLM miss: {r['choice']:<18} want {'/'.join(r['accepted']):<30} {r['text'][:70]}")

    for path in sorted(VAL.glob("startup-*.json")) + sorted(VAL.glob("idle-*.json")) + sorted(VAL.glob("load-*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        analysis[path.stem] = data.get("summary") or data
        print(path.stem, json.dumps(data.get("summary") or {k: v for k, v in data.items() if not isinstance(v, list)})[:600])
    (OUT / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
