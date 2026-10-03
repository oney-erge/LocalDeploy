"use strict";

import { esc } from "./shared.js?v=20260719-ui32";

function runLabel(run) { return [run.profile || run.modelId || "run", run.actualDevice || run.requestedDevice].filter(Boolean).join(" · "); }
function categorySummary(tests = []) {
  const groups = new Map();
  tests.forEach((test) => { const key = test.category || "?"; const row = groups.get(key) || []; row.push(test); groups.set(key, row); });
  return [...groups].map(([category, rows]) => ({ category, avg_accuracy: rows.reduce((sum, row) => sum + Number(row.accuracy || 0), 0) / Math.max(rows.length, 1) }));
}
function heatColor(accuracy) { const value = Math.max(0, Math.min(1, Number(accuracy || 0))); return value >= 0.8 ? "heat-good" : value >= 0.5 ? "heat-warn" : "heat-bad"; }

export function renderWinners(runs) {
  if (!runs.length) return "";
  const ranked = [...runs].sort((a, b) => (b.summary.passed || 0) - (a.summary.passed || 0) || (b.summary.avg_accuracy || 0) - (a.summary.avg_accuracy || 0) || (a.summary.avg_latency_s || 999999) - (b.summary.avg_latency_s || 999999));
  const top = ranked[0];
  const fastest = [...runs].filter((run) => run.summary.avg_latency_s != null).sort((a, b) => a.summary.avg_latency_s - b.summary.avg_latency_s)[0];
  const bestTps = [...runs].filter((run) => run.summary.avg_tokens_per_second != null).sort((a, b) => b.summary.avg_tokens_per_second - a.summary.avg_tokens_per_second)[0];
  const categories = new Set(runs.flatMap((run) => (run.category_summary || categorySummary(run.tests)).map((row) => row.category)));
  return [
    `<div class="metric-tile"><span>Top run</span><strong>${esc(runLabel(top))}</strong><small>${esc(top.summary.passed)}/${esc(top.summary.tests)} completed</small></div>`,
    `<div class="metric-tile"><span>Accuracy</span><strong>${esc(top.summary.avg_accuracy)}</strong><small>${esc(top.questionSetName || "benchmark")}</small></div>`,
    fastest ? `<div class="metric-tile"><span>Fastest</span><strong>${esc(fastest.summary.avg_latency_s)}s</strong><small>${esc(runLabel(fastest))}</small></div>` : "",
    bestTps ? `<div class="metric-tile"><span>Best tok/s</span><strong>${esc(bestTps.summary.avg_tokens_per_second)}</strong><small>${esc(runLabel(bestTps))}</small></div>` : "",
    `<div class="metric-tile"><span>Categories</span><strong>${esc(categories.size)}</strong><small>${esc(runs.length)} run${runs.length === 1 ? "" : "s"}</small></div>`,
  ].join("");
}

export function renderLeaderboard(runs) {
  if (!runs.length) return "Benchmark results appear here after the first streamed test result.";
  const rows = [...runs].sort((a, b) => (b.summary.passed || 0) - (a.summary.passed || 0) || (b.summary.avg_accuracy || 0) - (a.summary.avg_accuracy || 0) || (a.summary.avg_latency_s || 999999) - (b.summary.avg_latency_s || 999999)).map((run, index) => `<div class="leaderboard-row"><span class="rank">${index + 1}</span><div class="leaderboard-name"><b>${esc(runLabel(run))}</b><span>${esc(run.questionSetName || "")}</span></div><div class="leaderboard-metrics"><span><b>${esc(run.summary.passed)}/${esc(run.summary.tests)}</b> completed</span><span><b>${esc(run.summary.avg_accuracy)}</b> acc</span><span><b>${esc(run.summary.avg_latency_s)}s</b> latency</span><span><b>${esc(run.summary.avg_tokens_per_second ?? "-")}</b> tok/s</span></div></div>`).join("");
  return `<div class="leaderboard-list">${rows}</div>`;
}

export function renderHeatmap(runs) {
  if (!runs.length) return "Run benchmarks to fill the category heatmap.";
  const categories = Array.from(new Set(runs.flatMap((run) => (run.category_summary || categorySummary(run.tests)).map((row) => row.category)))).sort();
  const columns = `minmax(140px, 180px) repeat(${runs.length}, minmax(120px, 160px))`;
  const cells = categories.map((category) => `<div class="heat-category">${esc(category)}</div>${runs.map((run) => { const row = (run.category_summary || categorySummary(run.tests)).find((item) => item.category === category); const accuracy = row?.avg_accuracy ?? null; return `<div class="heat-cell ${accuracy == null ? "" : heatColor(accuracy)}" title="${esc(runLabel(run))} · ${esc(category)}">${accuracy == null ? "-" : Number(accuracy).toFixed(2)}</div>`; }).join("")}`).join("");
  return `<div class="heatmap-scroll"><div class="heatmap-grid" style="grid-template-columns:${columns}"><div class="heat-head">Category</div>${runs.map((run) => `<div class="heat-head">${esc(runLabel(run))}</div>`).join("")}${cells}</div></div>`;
}

export function renderScatter(runs) {
  const points = runs.filter((run) => run.summary.avg_latency_s != null && run.summary.avg_accuracy != null);
  if (!points.length) return "Run at least one benchmark to plot speed and quality.";
  if (points.length < 2) { const run = points[0]; return `<div class="scatter-single"><span class="eyebrow">One run captured</span><strong>${esc(run.summary.avg_accuracy)} accuracy</strong><span>${esc(run.summary.avg_latency_s)}s latency · ${esc(run.summary.avg_tokens_per_second ?? "-")} tok/s</span><small>Run another model or device to compare speed vs quality.</small></div>`; }
  const maxLatency = Math.max(...points.map((run) => Number(run.summary.avg_latency_s || 0)), 1);
  const labels = points.map((run, index) => { const x = 46 + (Number(run.summary.avg_latency_s || 0) / maxLatency) * 260; const y = 170 - Number(run.summary.avg_accuracy || 0) * 125; return `<g><circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="6" /><text x="${(x + 10).toFixed(1)}" y="${(y + 4).toFixed(1)}">${esc(String(index + 1))}</text><title>${esc(runLabel(run))}: ${esc(run.summary.avg_latency_s)}s, acc ${esc(run.summary.avg_accuracy)}</title></g>`; }).join("");
  return `<svg class="scatter" viewBox="0 0 340 205" role="img" aria-label="Speed quality scatter"><line x1="40" y1="180" x2="316" y2="180"></line><line x1="40" y1="34" x2="40" y2="180"></line><line class="gridline" x1="40" y1="96" x2="316" y2="96"></line><text x="118" y="199">avg latency, lower is better</text><text x="8" y="28">accuracy</text>${labels}</svg>`;
}

export function renderMatrix(runs) {
  if (!runs.length) return "Run benchmarks to fill the pass/fail matrix.";
  const tests = Array.from(new Set(runs.flatMap((run) => (run.tests || []).map((test) => test.name)))).sort(); const maxRows = 80;
  const rows = tests.slice(0, maxRows).map((name) => `<tr><th>${esc(name)}</th>${runs.map((run) => { const test = (run.tests || []).find((item) => item.name === name); const className = !test ? "" : test.success ? "matrix-pass" : "matrix-fail"; return `<td><span class="matrix-pill ${className}">${test ? `${test.success ? "PASS" : "FAIL"} · ${Number(test.accuracy || 0).toFixed(2)}` : "-"}</span></td>`; }).join("")}</tr>`).join("");
  const note = tests.length > maxRows ? `<div class="muted small matrix-note">Showing first ${maxRows} of ${tests.length} tests. Use Detailed results for filters.</div>` : "";
  return `<div class="table-wrap matrix-wrap"><table class="results matrix"><thead><tr><th>Test</th>${runs.map((run) => `<th>${esc(runLabel(run))}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table></div>${note}`;
}

export function renderResponseComparison({ testName, runs }) {
  return `<div class="card-head"><h3 class="sub">Responses: ${esc(testName)}</h3><button class="btn compact" id="close-response-drawer">Close</button></div>
    <div class="response-compare-grid">${runs.map((run) => {
      const test = (run.tests || []).find((item) => item.name === testName);
      const detail = test
        ? `${test.success ? "PASS" : "FAIL"} · acc ${test.accuracy} · ${test.elapsed_seconds}s`
        : "No result for this run";
      return `<div class="response-compare-card">
        <b>${esc(runLabel(run))}</b>
        <div class="muted small">${esc(detail)}</div>
        <pre>${esc(test?.response_preview || test?.error || "(no response preview)")}</pre>
      </div>`;
    }).join("")}</div>`;
}
