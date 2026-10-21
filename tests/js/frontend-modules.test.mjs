import assert from "node:assert/strict";
import test from "node:test";

import {
  categorySummary,
  normalizeRunRecord,
  summaryFromTests,
} from "../../localdeploy/web/js/benchmark.js";
import {
  renderHeatmap,
  renderLeaderboard,
  renderMatrix,
  renderResponseComparison,
  renderScatter,
  renderWinners,
} from "../../localdeploy/web/js/benchmark-views.js";
import {
  fitCacheKey,
  fitResultIsCurrent,
  paramsFromModelName,
  remoteSizeMatches,
} from "../../localdeploy/web/js/models.js";

const tests = [
  { name: "reasoning", category: "quality", success: true, accuracy: 1, elapsed_seconds: 2, approx_tokens_per_second: 20 },
  { name: "format", category: "quality", success: false, accuracy: 0.25, elapsed_seconds: 4 },
];

test("model discovery helpers normalize sizes and fit cache keys", () => {
  assert.equal(paramsFromModelName("qwen3.5:4b"), 4);
  assert.equal(paramsFromModelName("model-800m-q4"), 0.8);
  assert.equal(remoteSizeMatches(4, "4to8"), true);
  assert.equal(remoteSizeMatches(3.9, "4to8"), false);
  assert.equal(fitCacheKey("qwen3.5:4b", 8192), "qwen3.5:4b|8192");
  assert.equal(fitCacheKey("qwen3.5:4b", null), "qwen3.5:4b|auto");
});

test("a fit answer for an earlier budget is not shown once the budget changes", () => {
  const requestedWithLiveFree = fitCacheKey("qwen3:8b", 5600);
  assert.equal(fitResultIsCurrent(requestedWithLiveFree, "qwen3:8b", 5600), true);
  assert.equal(fitResultIsCurrent(requestedWithLiveFree, "qwen3:8b", 8192), false);
  assert.equal(fitResultIsCurrent(requestedWithLiveFree, "gemma3:4b", 5600), false);
});

test("benchmark normalization preserves compatibility fields", () => {
  assert.deepEqual(summaryFromTests(tests), {
    tests: 2,
    passed: 1,
    avg_accuracy: 0.625,
    avg_latency_s: 3,
    avg_tokens_per_second: 20,
    avg_ttft_ms: null,
  });
  assert.equal(categorySummary(tests)[0].category, "quality");
  const normalized = normalizeRunRecord({
    id: "run-1",
    profile: "qwen",
    model_id: "qwen3.5:4b",
    device: "gpu",
    peak_vram_mb: 4096,
    tests,
  }, "import");
  assert.equal(normalized.modelId, "qwen3.5:4b");
  assert.equal(normalized.actualDevice, "gpu");
  assert.equal(normalized.peakVramMb, 4096);
  assert.equal(normalized.source, "import");
});

test("benchmark visual builders render prepared runs without DOM state", () => {
  const run = normalizeRunRecord({
    id: "run-visual",
    profile: "qwen",
    model_id: "qwen3.5:4b",
    device: "gpu",
    questionSetName: "smoke",
    tests,
  });
  assert.match(renderWinners([run]), /Top run/);
  assert.match(renderLeaderboard([run]), /leaderboard-row/);
  assert.match(renderHeatmap([run]), /heatmap-grid/);
  assert.match(renderScatter([run]), /One run captured/);
  assert.match(renderMatrix([run]), /matrix-pass/);
  assert.match(renderResponseComparison({ testName: "reasoning", runs: [run] }), /response-compare-card/);
});
