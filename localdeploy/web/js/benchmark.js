"use strict";

import { $, $$, busy, downloadFile, esc, fmtMb, getJSON, postJSON, postMaybeStream, simpleModal, toast } from "./shared.js?v=20260719-ui32";
import { renderHeatmap, renderLeaderboard, renderMatrix, renderResponseComparison, renderScatter, renderWinners } from "./benchmark-views.js?v=20260719-ui32";

const state = {
  profiles: [], profileData: {}, profileModels: {}, defaultProfile: null,
  installedLoaded: false, benchmarkSelectedProfiles: [], benchShowUnpulled: false,
  testBenchInfo: null, questionSetValidation: null,
  lastRun: null, benchmarkRuns: [], liveBenchmarkRuns: [], selectedRunIds: [],
  compareBaselineId: null, activeRunId: null, currentQueue: [], activeController: null,
  queueCancelled: false, benchHistoryServer: false,
  systemContext: { vramBudgetMb: null, vramTotalMb: null, vramFreeMb: null },
};
const BENCHMARK_RUNS_KEY = "localdeploy.benchmarkRuns.v1";
const BENCH_HISTORY_SERVER_KEY = "localdeploy.benchHistoryServer.v1";
let initialized = false;
let onModelStateInvalidated = async () => {};
let setDefaultProfileAction = async () => {};

function targetVram() { return state.systemContext.vramBudgetMb ?? state.systemContext.vramTotalMb ?? state.systemContext.vramFreeMb; }
function installedStatusForProfile(profileName) {
  const availability = state.profileData[profileName]?.availability;
  return availability ? { label: availability.label, cls: availability.className } : { label: "unknown", cls: "off" };
}
function profileIsUnpulled(name) {
  if (!state.installedLoaded) return false;
  return ["not pulled", "file missing"].includes(installedStatusForProfile(name).label);
}
function detectDevice(profileName) {
  const placement = state.profileData[profileName]?.placement;
  return placement ? String(placement).toLowerCase() : null;
}
async function setDefaultProfile(profile, button) {
  busy(button, true);
  try { await setDefaultProfileAction(profile); toast(`Default profile set to ${profile}.`, "success"); }
  catch (error) { toast(`Set default failed: ${error.message}`, "error"); }
  finally { busy(button, false); }
}

export function updateModelContext(snapshot = {}) {
  state.profiles = (snapshot.profiles || []).map((profile) => profile.name);
  state.profileData = Object.fromEntries((snapshot.profiles || []).map((profile) => [profile.name, structuredClone(profile)]));
  state.profileModels = Object.fromEntries((snapshot.profiles || []).map((profile) => [profile.name, profile.model_id || profile.name]));
  state.defaultProfile = snapshot.defaultProfile || null;
  state.installedLoaded = !!snapshot.installedLoaded;
  renderBenchmarkProfileChips(); updateBenchmarkSummary(); renderBenchmarkWorkspace();
}
export function updateSystemContext(snapshot = {}) {
  state.systemContext = { vramBudgetMb: snapshot.vramBudgetMb ?? null, vramTotalMb: snapshot.vramTotalMb ?? null, vramFreeMb: snapshot.vramFreeMb ?? null };
  updateBenchmarkSummary();
}

function extractCard(text) {
  try {
    const j = JSON.parse(text);
    if (j && (j.kind === "localdeploy.report_card" || Array.isArray(j.tests))) return j;
  } catch {
    /* not raw JSON; try embedded */
  }
  const m = text.match(/<script[^>]*id=["']localdeploy-card["'][^>]*>([\s\S]*?)<\/script>/);
  if (m) {
    // Reverse the server's html.escape(..., quote=False): &lt; and &gt; first,
    // then &amp; last so escaped entities round-trip correctly.
    const unescaped = m[1].replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&");
    try {
      return JSON.parse(unescaped);
    } catch {
      /* fall through */
    }
  }
  return null;
}

function runId() {
  return `run-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

function hashText(text) {
  let h = 2166136261;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0).toString(16).padStart(8, "0");
}

function loadBenchmarkRuns() {
  try {
    const raw = localStorage.getItem(BENCHMARK_RUNS_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    state.benchmarkRuns = Array.isArray(parsed) ? parsed.slice(0, 80) : [];
    state.benchmarkRuns.forEach((r) => {
      r.source = r.source || "restored-history";
      r.id = r.id || runId();
    });
  } catch {
    state.benchmarkRuns = [];
  }
}

function saveBenchmarkRuns() {
  try {
    localStorage.setItem(BENCHMARK_RUNS_KEY, JSON.stringify(state.benchmarkRuns.slice(0, 80)));
  } catch {
    /* localStorage can be disabled; the current session still works */
  }
}

// --- opt-in server-side history (reports/benchmark-history) -----------------
// localStorage dies with the browser profile; when the toggle is on, completed
// runs are mirrored to the server as JSON files and merged back in on load.

async function syncRunToServer(run) {
  try {
    const res = await postJSON("/benchmark/history/save", { run });
    if (!res.success) toast(res.error || "Could not store run on server.", "error");
  } catch (err) {
    toast(`Server history save failed: ${err.message}`, "error");
  }
}

async function loadServerHistory(notify = false) {
  try {
    const res = await getJSON("/benchmark/history");
    if (!res.success) return;
    const existing = new Set(state.benchmarkRuns.map((r) => r.id));
    const fresh = (res.runs || []).filter((r) => r && r.id && !existing.has(r.id));
    if (fresh.length) {
      addBenchmarkRuns(fresh.map((r) => ({ ...r, source: "server-history" })), false);
    }
    if (notify) toast(`Loaded ${fresh.length} run(s) from server history.`, "success");
  } catch {
    /* best-effort: a missing server history never breaks the workspace */
  }
}

function initServerHistoryToggle() {
  const cb = $("#bench-history-server");
  if (!cb) return;
  try {
    state.benchHistoryServer = localStorage.getItem(BENCH_HISTORY_SERVER_KEY) === "1";
  } catch {}
  cb.checked = state.benchHistoryServer;
  cb.addEventListener("change", async () => {
    state.benchHistoryServer = cb.checked;
    try {
      localStorage.setItem(BENCH_HISTORY_SERVER_KEY, cb.checked ? "1" : "0");
    } catch {}
    if (cb.checked) {
      // Push what this browser already has, then pull anything it's missing.
      for (const run of state.benchmarkRuns) await syncRunToServer(run);
      await loadServerHistory(true);
      toast("Completed runs will also be saved under reports/benchmark-history.", "success");
    } else {
      toast("Server-side storing is off. Files already under reports/ are kept.", "info");
    }
  });
  if (state.benchHistoryServer) void loadServerHistory();
}

function allBenchmarkRuns() {
  const liveIds = new Set(state.liveBenchmarkRuns.map((r) => r.id));
  return [...state.liveBenchmarkRuns, ...state.benchmarkRuns.filter((r) => !liveIds.has(r.id))];
}

function runLabel(run) {
  const model = run?.modelId || run?.model_id || run?.profile || "run";
  const dev = run?.actualDevice || run?.requestedDevice || run?.device;
  return `${model}${dev ? `/${String(dev).toUpperCase()}` : ""}`;
}

function summaryFromTests(tests) {
  const rows = tests || [];
  const successes = rows.filter((t) => t.success);
  const mean = (arr) => (arr.length ? arr.reduce((s, v) => s + Number(v || 0), 0) / arr.length : 0);
  const tps = successes.map((t) => t.approx_tokens_per_second).filter((v) => v != null);
  const ttft = successes.map((t) => t.metrics?.ttft_ms).filter((v) => v != null);
  return {
    tests: rows.length,
    passed: successes.length,
    avg_accuracy: Number(mean(rows.map((t) => t.accuracy || 0)).toFixed(3)),
    avg_latency_s: Number(mean(rows.map((t) => t.elapsed_seconds || 0)).toFixed(3)),
    avg_tokens_per_second: tps.length ? Number(mean(tps).toFixed(2)) : null,
    avg_ttft_ms: ttft.length ? Number(mean(ttft).toFixed(1)) : null,
  };
}

function categorySummary(tests) {
  const groups = {};
  for (const t of tests || []) {
    (groups[t.category || "?"] ||= []).push(t);
  }
  return Object.keys(groups)
    .sort()
    .map((category) => {
      const rows = groups[category];
      return { category, ...summaryFromTests(rows) };
    });
}

function normalizeRunRecord(input, source = "current-run") {
  const tests = input.tests || [];
  const summary = input.summary || summaryFromTests(tests);
  const profile = input.profile || input.profileName || null;
  const modelId = input.modelId || input.model_id || profile || null;
  const requestedDevice = input.requestedDevice || input.device || null;
  const actualDevice = input.actualDevice || input.device || requestedDevice || null;
  const profileProvenance = (input.provenance?.profiles || {})[profile] || {};
  return {
    id: input.id || runId(),
    createdAt: input.createdAt || input.generated_at || new Date().toISOString(),
    profile,
    modelId,
    backend: input.backend || profileProvenance.backend || null,
    requestedDevice,
    actualDevice,
    questionSetName: input.questionSetName || "Imported report card",
    questionSetHash: input.questionSetHash || null,
    hardware: input.hardware || {},
    provenance: input.provenance || {},
    variance: input.variance || summary.variance || {},
    aggregates: input.aggregates || [],
    repetitions: input.repetitions || summary.repetitions || 1,
    peakVramMb: input.peakVramMb ?? input.peak_vram_mb ?? null,
    tests,
    summary,
    category_summary: input.category_summary || input.categorySummary || categorySummary(tests),
    elapsedSeconds: input.elapsedSeconds ?? summary.elapsed_seconds ?? null,
    source,
  };
}

function addBenchmarkRuns(runs, selectNew = true) {
  const normalized = runs.map((r) => normalizeRunRecord(r, r.source || "current-run"));
  const existing = new Set(state.benchmarkRuns.map((r) => r.id));
  const fresh = normalized.filter((r) => !existing.has(r.id));
  const previousSelected = state.selectedRunIds.filter((id) => existing.has(id));
  const previousLatest = state.benchmarkRuns[0]?.id ? [state.benchmarkRuns[0].id] : [];
  state.benchmarkRuns = [...fresh, ...state.benchmarkRuns].slice(0, 80);
  if (selectNew && fresh.length) {
    const freshIds = fresh.map((r) => r.id);
    let nextSelected = previousSelected.length ? previousSelected : previousLatest;
    freshIds.forEach((id) => {
      if (!nextSelected.includes(id)) nextSelected.push(id);
    });
    // Keep every newly completed run selected (no cap): a 5th+ model must stay
    // in the dashboard/queue instead of silently dropping an earlier one.
    state.selectedRunIds = nextSelected;
    state.compareBaselineId = state.compareBaselineId && nextSelected.includes(state.compareBaselineId)
      ? state.compareBaselineId
      : nextSelected[0];
    state.activeRunId = fresh[0].id;
    state.lastRun = fresh[0];
  }
  saveBenchmarkRuns();
  if (state.benchHistoryServer) {
    // Mirror only freshly completed local runs - imports and server-restored
    // records are already durable somewhere.
    fresh.filter((r) => r.source === "current-run").forEach((r) => void syncRunToServer(r));
  }
  renderBenchmarkWorkspace();
}

function liveRecordFromQueueItem(item) {
  const tests = item.tests || [];
  return {
    ...normalizeRunRecord(
      {
        ...item,
        hardware: state.lastHardware || {},
        summary: summaryFromTests(tests),
        category_summary: categorySummary(tests),
        source: "live-run",
      },
      "live-run"
    ),
    status: item.status,
    progress: item.progress || 0,
  };
}

function syncLiveBenchmarkRun(item, select = false) {
  if (!item || !(item.tests || []).length) return;
  const record = liveRecordFromQueueItem(item);
  const idx = state.liveBenchmarkRuns.findIndex((r) => r.id === record.id);
  if (idx >= 0) state.liveBenchmarkRuns[idx] = record;
  else state.liveBenchmarkRuns.unshift(record);
  state.activeRunId = record.id;
  if (select && !state.selectedRunIds.includes(record.id)) {
    state.selectedRunIds = [...state.selectedRunIds, record.id];
    if (!state.compareBaselineId) state.compareBaselineId = record.id;
  }
  renderBenchmarkWorkspace();
}

function removeLiveBenchmarkRun(id, rerender = true) {
  const before = state.liveBenchmarkRuns.length;
  state.liveBenchmarkRuns = state.liveBenchmarkRuns.filter((r) => r.id !== id);
  if (rerender && before !== state.liveBenchmarkRuns.length) renderBenchmarkWorkspace();
}

function selectedBenchProfiles() {
  const selected = state.benchmarkSelectedProfiles.filter((name) => state.profiles.includes(name));
  if (selected.length) return selected;
  const fallback = $("#bench-profile-select")?.value;
  return fallback ? [fallback] : [];
}

function renderBenchmarkProfileChips() {
  const body = $("#bench-profile-chips");
  if (!body) return;
  if (!state.profiles.length) {
    body.innerHTML = `<div class="empty-state">No profiles yet. Pull a model in <b>Setup &amp; Deploy → Get a model</b> and it appears here automatically.</div>`;
    return;
  }
  const currentlySelected = new Set(selectedBenchProfiles());
  if (!state.benchmarkSelectedProfiles.length) {
    if (currentlySelected.size) {
      state.benchmarkSelectedProfiles = Array.from(currentlySelected);
    } else if (state.defaultProfile) {
      state.benchmarkSelectedProfiles = [state.defaultProfile];
      currentlySelected.add(state.defaultProfile);
    }
  }
  const filter = ($("#bench-profile-filter")?.value || "").trim().toLowerCase();
  const matchesFilter = (name) => {
    const model = state.profileModels[name] || name;
    return !filter || name.toLowerCase().includes(filter) || model.toLowerCase().includes(filter);
  };
  // Not-pulled profiles stay hidden unless revealed (or already selected -
  // never hide something the user has checked).
  const candidates = state.profiles.filter(matchesFilter);
  const hidden = state.benchShowUnpulled
    ? []
    : candidates.filter((name) => profileIsUnpulled(name) && !currentlySelected.has(name));
  const visibleProfiles = candidates.filter((name) => !hidden.includes(name));
  const toggleHtml = hidden.length || state.benchShowUnpulled
    ? `<button class="link-btn" id="bench-toggle-unpulled" type="button">${
        state.benchShowUnpulled
          ? "Hide profiles without a pulled model"
          : `Show ${hidden.length} hidden (model not pulled)`
      }</button>`
    : "";
  if (!visibleProfiles.length) {
    body.innerHTML = filter
      ? `<div class="empty-state">No profiles match this filter.${toggleHtml ? ` ${toggleHtml}` : ""}</div>`
      : `<div class="empty-state">No pulled models to benchmark yet. Pull one in <b>Setup &amp; Deploy</b> first.${toggleHtml ? ` ${toggleHtml}` : ""}</div>`;
    return;
  }
  body.innerHTML = visibleProfiles
    .map((name) => {
      const p = state.profileData[name] || {};
      const model = state.profileModels[name] || name;
      const status = installedStatusForProfile(name);
      const checked = currentlySelected.has(name) ? " checked" : "";
      const backend = p.backend || "ollama";
      return `<label class="profile-chip-card${checked ? " selected" : ""}">
        <input type="checkbox" value="${esc(name)}"${checked} />
        <span class="profile-chip-main">
          <b title="${esc(name)}">${esc(name)}</b>
          <span class="profile-chip-model" title="${esc(model)}">${esc(model)}</span>
        </span>
        <span class="profile-chip-badges">
          <span class="badge">${esc(backend)}</span>
          <span class="badge ${esc(status.cls)}">${esc(status.label)}</span>
        </span>
      </label>`;
    })
    .join("") + (toggleHtml ? `<div class="chip-grid-footer">${toggleHtml}</div>` : "");
}

function handleBenchmarkProfilesClick(event) {
  if (event.target.closest("#bench-toggle-unpulled")) {
    state.benchShowUnpulled = !state.benchShowUnpulled;
    renderBenchmarkProfileChips();
  }
}

function handleBenchmarkProfilesChange(event) {
  const input = event.target.closest('input[type="checkbox"]');
  if (!input) return;
  const selected = new Set(state.benchmarkSelectedProfiles);
  if (input.checked) selected.add(input.value);
  else selected.delete(input.value);
  state.benchmarkSelectedProfiles = Array.from(selected);
  input.closest(".profile-chip-card")?.classList.toggle("selected", input.checked);
  const first = selectedBenchProfiles()[0];
  if (first) $("#bench-profile-select").value = first;
  updateBenchmarkSummary();
}

function currentQuestionSetInfo() {
  const raw = $("#qs-editor")?.value?.trim() || "";
  if (!raw) {
    const count = state.testBenchInfo?.test_count;
    const cats = Object.keys(state.testBenchInfo?.categories || {}).length;
    return {
      questions: null,
      name: "Built-in LocalDeploy bench",
      hash: "builtin",
      meta: count ? `${count} tests${cats ? ` · ${cats} categories` : ""}` : "Built-in suite",
    };
  }
  const parsed = JSON.parse(raw);
  const count = Array.isArray(parsed.questions) ? parsed.questions.length : 0;
  const cats = new Set((parsed.questions || []).map((q) => q.category || "?")).size;
  const validation = state.questionSetValidation?.valid ? "validated" : "not validated";
  return {
    questions: parsed,
    name: "Custom JSON question set",
    hash: hashText(raw),
    meta: `${count} question${count === 1 ? "" : "s"}${cats ? ` · ${cats} categories` : ""} · ${validation}`,
  };
}

function updateBenchmarkSummary() {
  const profiles = selectedBenchProfiles();
  const setName = $("#bench-set-name");
  const setMeta = $("#bench-set-meta");
  try {
    const q = currentQuestionSetInfo();
    if (setName) setName.textContent = q.name;
    if (setMeta) setMeta.textContent = q.meta;
  } catch {
    if (setName) setName.textContent = "Custom JSON question set";
    if (setMeta) setMeta.textContent = "Invalid JSON until validated.";
  }
  const selected = $("#bench-selected-count");
  if (selected) selected.textContent = `${profiles.length} profile${profiles.length === 1 ? "" : "s"}`;
  const history = $("#bench-history-count");
  if (history) history.textContent = `${state.benchmarkRuns.length} run${state.benchmarkRuns.length === 1 ? "" : "s"}`;
  const exportSelected = $("#btn-export-selected");
  if (exportSelected) exportSelected.disabled = state.selectedRunIds.length === 0;
}

// ---------------------------------------------------------------------------
// Tab 1 - Hardware
// ---------------------------------------------------------------------------

function updateBenchHelp(info) {
  if (!info?.test_count) return;
  const graderTypes = $("#grader-types")?.innerHTML || "…";
  state.testBenchInfo = info;
  const categories = Object.entries(info.categories || {})
    .map(([name, count]) => `${name} (${count})`)
    .join(", ");
  $("#bench-help").innerHTML =
    `Leave the editor empty to run the built-in LocalDeploy test bench: <b>${esc(info.test_count)} tests</b>` +
    (categories ? ` across ${esc(categories)}` : "") +
    `. Use JSON here only when you want a custom set: each question needs <code>name</code>, <code>category</code>, <code>prompt</code>, <code>max_output_tokens</code>, and a <code>grader</code> (one of <span id="grader-types">${graderTypes}</span>).`;
}

async function useBuiltInBench() {
  try {
    const info = await getJSON("/benchmark/test-bench");
    updateBenchHelp(info);
    $("#qs-editor").value = JSON.stringify(info.question_set || { version: 1, questions: [] }, null, 2);
    state.questionSetValidation = null;
    $("#validate-result").className = "result ok";
    $("#validate-result").textContent =
      `Loaded LocalDeploy test bench JSON: ${info.test_count} tests.`;
    $("#question-set-details").open = true;
    updateBenchmarkSummary();
  } catch (err) {
    toast(`Could not load LocalDeploy test bench metadata: ${err.message}`, "error");
  }
}

async function loadExample() {
  try {
    const example = await getJSON("/benchmark/example");
    $("#qs-editor").value = JSON.stringify(example, null, 2);
    state.questionSetValidation = null;
    $("#validate-result").className = "result";
    $("#validate-result").textContent = "Loaded a small JSON sample. Run benchmark will use this custom set until you clear the editor.";
    $("#question-set-details").open = true;
    updateBenchmarkSummary();
  } catch (err) {
    toast(`Could not load JSON sample: ${err.message}`, "error");
  }
}

function uploadFile(input) {
  const file = input.files?.[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    $("#qs-editor").value = String(reader.result || "");
    state.questionSetValidation = null;
    $("#validate-result").textContent = "";
    $("#question-set-details").open = true;
    updateBenchmarkSummary();
  };
  reader.onerror = () => toast("Could not read file.", "error");
  reader.readAsText(file);
  input.value = "";
}

function parseEditor() {
  const raw = $("#qs-editor").value.trim();
  if (!raw) throw new Error("The editor is empty.");
  try {
    return JSON.parse(raw);
  } catch (err) {
    throw new Error(`Invalid JSON: ${err.message}`);
  }
}

async function validateSet() {
  const node = $("#validate-result");
  let payload;
  try {
    payload = parseEditor();
  } catch (err) {
    state.questionSetValidation = { valid: false };
    node.className = "result err";
    node.textContent = err.message;
    $("#question-set-details").open = true;
    return null;
  }
  try {
    const report = await postJSON("/benchmark/validate", payload);
    if (report.valid) {
      state.questionSetValidation = { valid: true, questionCount: report.question_count };
      node.className = "result ok";
      node.textContent = `Valid - ${report.question_count} question(s).`;
      $("#question-set-details").open = true;
    } else {
      state.questionSetValidation = { valid: false };
      node.className = "result err";
      $("#question-set-details").open = true;
      node.innerHTML =
        `Invalid set:` +
        `<ul class="err-list">` +
        (report.errors || [])
          .map((e) => `<li>${esc(e.name ? `[${e.name}] ` : e.index >= 0 ? `row ${e.index}: ` : "")}${esc(e.error)}</li>`)
          .join("") +
        `</ul>`;
    }
    return report;
  } catch (err) {
    node.className = "result err";
    node.textContent = `Validation failed: ${err.message}`;
    return null;
  }
}

// ---------------------------------------------------------------------------
// Tab 2 - Run (streamed)
// ---------------------------------------------------------------------------

// Portable per-test fields for a report card (shared by single + dual runs).

function collectTest(evt) {
  return {
    name: evt.name,
    category: evt.category,
    success: evt.success,
    accuracy: evt.accuracy,
    elapsed_seconds: evt.elapsed_seconds,
    approx_tokens_per_second: evt.approx_tokens_per_second ?? null,
    error: evt.error || null,
    warning: evt.warning || null,
    response_preview: evt.response_preview || "",
    repetition: evt.repetition || 1,
    repetitions: evt.repetitions || 1,
    metrics: evt.metrics || {},
    tokens_per_second_source: evt.tokens_per_second_source || "estimated",
    warm_state: evt.warm_state || null,
  };
}

// Append one test_result row (+ collapsible preview) to a tbody. Shared by the
// live queue view and the persisted detailed-results table.

function appendResultRow(tbody, evt, runName = "") {
  const tps = evt.approx_tokens_per_second ?? null;
  const resultBadge = evt.success ? `<span class="pass">PASS</span>` : `<span class="fail">FAIL</span>`;
  const errSnippet = !evt.success && evt.error
    ? ` <span class="muted small" title="${esc(evt.error)}">${esc(evt.error.slice(0, 50))}${evt.error.length > 50 ? "…" : ""}</span>`
    : "";
  const warnBadge = evt.warning ? ` <span class="warn-badge" title="${esc(evt.warning)}">⚠</span>` : "";
  const rateTitle = evt.tokens_per_second_source === "backend" ? "Measured by the inference backend" : "Estimated from response length";
  const tpsCell = tps != null ? `<span title="${rateTitle}">${tps.toFixed(1)}</span>` : `<span class="muted">-</span>`;
  const testName = evt.repetitions > 1 ? `${evt.name} · r${evt.repetition}` : evt.name;
  const hasPreview = !!evt.response_preview;

  const tr = document.createElement("tr");
  tr.innerHTML = `<td>${esc(runName)}</td>
    <td>${esc(testName)}</td>
    <td>${esc(evt.category)}</td>
    <td>${resultBadge}${errSnippet}${warnBadge}</td>
    <td class="num">${esc(evt.elapsed_seconds)}s</td>
    <td class="num">${tpsCell}</td>
    <td class="num">${esc(evt.accuracy)}</td>
    <td><button class="btn-preview${hasPreview ? "" : " btn-preview-none"}" aria-label="Toggle response preview" title="${hasPreview ? "Show/hide response" : "No response captured"}">▸</button></td>`;
  tbody.appendChild(tr);

  const previewTr = document.createElement("tr");
  previewTr.className = "preview-row hidden";
  previewTr.innerHTML = `<td colspan="8"><div class="response-preview">${esc(evt.response_preview || "(no preview)")}</div></td>`;
  tbody.appendChild(previewTr);
  const toggleBtn = tr.querySelector(".btn-preview");
  if (hasPreview) {
    toggleBtn.addEventListener("click", () => {
      const hidden = previewTr.classList.toggle("hidden");
      toggleBtn.textContent = hidden ? "▸" : "▾";
    });
  } else {
    toggleBtn.disabled = true;
  }
}

// Update the run progress bar's width + ARIA value together.

function setProgress(pct) {
  const bar = $("#run-progress-bar");
  bar.style.width = `${pct}%`;
  bar.setAttribute("aria-valuenow", String(pct));
  const label = $("#run-active-percent");
  if (label) label.textContent = `${Math.max(0, Math.min(100, Math.round(pct)))}%`;
}

function setActiveRun(item, done = 0, total = 0) {
  const panel = $("#run-active");
  if (!panel) return;
  if (!item) {
    panel.classList.add("hidden");
    return;
  }
  panel.classList.remove("hidden");
  $("#run-active-title").textContent = runLabel(item);
  $("#run-active-detail").textContent = item.current || item.status || "Running";
  const elapsed = item.startedAt ? `${Math.round((Date.now() - item.startedAt) / 1000)}s elapsed` : "";
  const tests = total ? `${done} of ${total} tests` : "";
  $("#run-active-count").textContent = [tests, elapsed].filter(Boolean).join(" · ") || item.status || "";
}

function buildRunQueue(profiles, deviceChoice, questionInfo) {
  const devices = deviceChoice === "both" ? ["cpu", "gpu"] : [deviceChoice || "auto"];
  return profiles.flatMap((profile) =>
    devices.map((device) => ({
      id: runId(),
      profile,
      modelId: state.profileModels[profile] || profile,
      requestedDevice: device,
      actualDevice: null,
      questionSetName: questionInfo.name,
      questionSetHash: questionInfo.hash,
      createdAt: new Date().toISOString(),
      status: "waiting",
      progress: 0,
      current: "Waiting",
      tests: [],
      elapsedSeconds: null,
      error: null,
    }))
  );
}

function renderRunQueue(queue) {
  const slot = $("#run-queue");
  if (!queue.length) {
    slot.innerHTML = `<div class="muted small">No benchmark runs are queued.</div>`;
    return;
  }
  const finished = queue.filter((item) => ["complete", "failed", "stopped"].includes(item.status)).length;
  slot.innerHTML =
    (finished > 1
      ? `<div class="queue-toolbar"><button class="btn compact" id="btn-clear-finished" title="Remove completed, failed, and stopped rows from the queue">Clear finished (${finished})</button></div>`
      : "") +
    queue
      .map((item) => {
        const cls =
          item.status === "complete"
            ? "on"
            : item.status === "failed"
              ? "wont"
              : item.status === "stopped"
                ? "tight"
                : item.status === "deploying" || item.status === "running"
                  ? "cpu"
                  : "off";
        const label =
          {
            waiting: "Queued",
            deploying: "Deploying",
            running: "Running",
            complete: "Finished",
            failed: "Failed",
            stopped: "Stopped",
          }[item.status] || item.status;
        const elapsed = item.elapsedSeconds != null
          ? `${item.elapsedSeconds}s`
          : item.startedAt
            ? `${Math.round((Date.now() - item.startedAt) / 1000)}s`
            : "0s";
        const canMove = item.status === "waiting";
        const isFinished = ["complete", "failed", "stopped"].includes(item.status);
        const showProgress = ["deploying", "running"].includes(item.status) && (item.status === "deploying" || item.total || item.progress);
        const indeterminate = item.status === "deploying" && !item.progress;
        const progress = showProgress
          ? `<div class="queue-row-progress"><div class="run-progress-track"><div class="run-progress-fill${indeterminate ? " indeterminate" : ""}"${indeterminate ? "" : ` style="width:${esc(item.progress || 0)}%"`}></div></div></div>`
          : "";
        // Waiting rows can be reordered + removed; finished rows can be dismissed.
        const controls =
          canMove || isFinished
            ? `<div class="queue-actions" aria-label="Queue controls">
                ${canMove ? `<button class="btn compact queue-move" data-id="${esc(item.id)}" data-delta="-1" title="Move up">Up</button>
                <button class="btn compact queue-move" data-id="${esc(item.id)}" data-delta="1" title="Move down">Down</button>` : ""}
                <button class="btn compact danger queue-remove" data-id="${esc(item.id)}" title="${canMove ? "Remove from queue" : "Dismiss"}">${canMove ? "Remove" : "Dismiss"}</button>
              </div>`
            : "";
        const reason = item.status === "failed" && item.error
          ? `<span class="muted small queue-reason" title="${esc(item.error)}">${esc(item.error)}</span>`
          : "";
        return `<div class="queue-row queue-status-${esc(item.status)}" data-id="${esc(item.id)}">
          <span class="queue-status-dot" aria-hidden="true"></span>
          <div class="queue-row-main">
            <b>${esc(runLabel(item))}</b>
            <span class="muted small">${esc(item.profile)} · ${esc(item.current || item.status)} · ${esc(elapsed)}</span>
            ${progress}
            ${reason}
          </div>
          <div class="queue-row-side">
            <span class="badge ${cls}">${esc(label)}</span>
            ${controls}
          </div>
        </div>`;
      })
      .join("");
}

// Remove all finished (complete/failed/stopped) rows from the queue, leaving
// waiting and active runs intact.

function clearFinishedQueue() {
  const queue = state.currentQueue || [];
  state.currentQueue = queue.filter((item) => !["complete", "failed", "stopped"].includes(item.status));
  renderRunQueue(state.currentQueue);
}

function removeQueuedRun(id) {
  const queue = state.currentQueue || [];
  const idx = queue.findIndex((item) => item.id === id);
  if (idx < 0) return;
  // Never yank the row that's actively deploying/running out from under the
  // run loop; use the active-run Stop button for that.
  if (["deploying", "running"].includes(queue[idx].status)) return;
  const wasWaiting = queue[idx].status === "waiting";
  queue.splice(idx, 1);
  renderRunQueue(queue);
  const summary = $("#run-summary");
  if (summary && wasWaiting) {
    summary.className = "result";
    summary.textContent = `Removed a waiting benchmark. ${queue.filter((q) => q.status === "waiting").length} waiting.`;
  }
}

function moveQueuedRun(id, delta) {
  const queue = state.currentQueue || [];
  const idx = queue.findIndex((item) => item.id === id);
  const next = idx + delta;
  if (idx < 0 || next < 0 || next >= queue.length) return;
  if (queue[idx].status !== "waiting" || queue[next].status !== "waiting") return;
  [queue[idx], queue[next]] = [queue[next], queue[idx]];
  renderRunQueue(queue);
}

async function runQueueItem(item, questionInfo, timeout, repetitions, controller) {
  const summary = $("#run-summary");
  const body = { profiles: [item.profile], timeout, repetitions };
  if (item.requestedDevice && item.requestedDevice !== "auto") body.device = item.requestedDevice;
  if (questionInfo.questions) body.questions = questionInfo.questions;
  const pack = $("#bench-pack")?.value;
  if (pack && !questionInfo.questions) body.pack = pack;

  let total = 0;
  let done = 0;
  const started = Date.now();
  item.startedAt = started;
  item.status = "running";
  item.current = "Starting";
  setActiveRun(item);
  renderRunQueue(state.currentQueue || []);
  let out;
  try {
    out = await postMaybeStream("/benchmark/run", body, (evt) => {
    if (evt.event === "deploy_start") {
      item.status = "deploying";
      item.current = `Deploying on ${(evt.device || item.requestedDevice).toUpperCase()}`;
      setActiveRun(item, done, total);
    } else if (evt.event === "deploy_end") {
      item.current = "Deployment complete";
      if (evt.warning) item.warning = evt.warning;
      setActiveRun(item, done, total);
    } else if (evt.event === "run_start") {
      item.status = "running";
      item.done = 0;
      total = evt.test_count || 0;
      item.total = total;
      item.current = `0 / ${total} tests`;
      item.provenance = evt.provenance || {};
      item.repetitions = evt.repetitions || repetitions;
      setProgress(0);
      setActiveRun(item, 0, total);
    } else if (evt.event === "profile_start") {
      item.status = "running";
      item.current = "Model warm-up";
      setActiveRun(item, done, total);
    } else if (evt.event === "test_aggregate") {
      (item.aggregates ||= []).push(evt);
    } else if (evt.event === "test_start") {
      item.status = "running";
      item.current = `Running ${evt.name || "test"}`;
      setActiveRun(item, done, total);
    } else if (evt.event === "test_result") {
      item.status = "running";
      done++;
      item.tests.push(collectTest(evt));
      item.progress = total ? Math.round((done / total) * 100) : item.progress;
      item.done = done;
      item.total = total;
      item.current = `${done} / ${total || "?"} tests`;
      setProgress(item.progress);
      setActiveRun(item, done, total);
      syncLiveBenchmarkRun(item, true);
    } else if (evt.event === "profile_aborted") {
      item.status = "failed";
      item.error = evt.reason || "profile aborted";
      item.current = item.error;
      setActiveRun(item, done, total);
      removeLiveBenchmarkRun(item.id);
    } else if (evt.event === "profile_end") {
      if (evt.actual_device) item.actualDevice = evt.actual_device;
      item.variance = {
        latency_stdev_seconds: evt.summary?.latency_stdev_seconds ?? 0,
        tokens_per_second_stdev: evt.summary?.tokens_per_second_stdev ?? 0,
        by_test: item.aggregates || [],
      };
      item.peakVramMb = evt.summary?.peak_vram_mb ?? null;
      item.current = "Benchmark complete";
      setActiveRun(item, done || total, total);
      syncLiveBenchmarkRun(item, true);
    } else if (evt.event === "benchmark_unload_end") {
      item.current = "Temporary benchmark model unloaded";
      setActiveRun(item, done || total, total);
    } else if (evt.event === "benchmark_unload_error") {
      item.current = "Benchmark complete; unload needs manual check";
      item.warning = evt.error || "temporary benchmark unload failed";
      setActiveRun(item, done || total, total);
    } else if (evt.event === "run_end") {
      item.elapsedSeconds = evt.elapsed_seconds;
      item.progress = 100;
      setProgress(100);
      setActiveRun(item, done || total, total);
      syncLiveBenchmarkRun(item, true);
    } else if (evt.event === "error") {
      item.status = "failed";
      item.error = evt.error || "run failed";
      item.current = item.error;
      setActiveRun(item, done, total);
      removeLiveBenchmarkRun(item.id);
    }
    renderRunQueue(state.currentQueue || []);
    }, controller.signal);
  } catch (err) {
    // A whole-queue cancel sets queueCancelled and bubbles up so the loop stops.
    // A per-item Stop aborts only this run: mark it stopped and let the caller
    // continue with the rest of the queue.
    if (err.name === "AbortError") {
      if (state.queueCancelled) throw err;
      item.status = "stopped";
      item.current = "Stopped";
      item.error = "Stopped by user";
      setActiveRun(null);
      removeLiveBenchmarkRun(item.id);
      renderRunQueue(state.currentQueue || []);
      const note = $("#run-summary");
      if (note) {
        note.className = "result";
        note.textContent = `Stopped ${runLabel(item)}. Continuing with the rest of the queue.`;
      }
      return null;
    }
    throw err;
  }

  if (!out.streamed) {
    const j = out.json || {};
    item.status = "failed";
    item.error = j.error || "Run could not start.";
    item.current = item.error;
    removeLiveBenchmarkRun(item.id);
    renderRunQueue(state.currentQueue || []);
    return null;
  }
  if (item.status === "failed") {
    removeLiveBenchmarkRun(item.id);
    return null;
  }
  if (!item.tests.length) {
    item.status = "failed";
    item.error = "No test results streamed.";
    item.current = item.error;
    removeLiveBenchmarkRun(item.id);
    renderRunQueue(state.currentQueue || []);
    return null;
  }
  item.actualDevice = item.actualDevice || (item.requestedDevice === "auto" ? null : item.requestedDevice);
  item.status = "complete";
  item.progress = 100;
  item.current = "Complete";
  item.elapsedSeconds = item.elapsedSeconds ?? Number(((Date.now() - started) / 1000).toFixed(2));
  const run = normalizeRunRecord(
    {
      ...item,
      hardware: item.provenance?.hardware || state.lastHardware || {},
      source: "current-run",
      summary: summaryFromTests(item.tests),
      category_summary: categorySummary(item.tests),
    },
    "current-run"
  );
  summary.className = "result";
  summary.textContent = `Completed ${runLabel(run)}. Continuing queue.`;
  removeLiveBenchmarkRun(item.id, false);
  renderRunQueue(state.currentQueue || []);
  return run;
}

async function runBenchmark() {
  const btn = $("#btn-run");
  const cancelBtn = $("#btn-run-cancel");
  const summary = $("#run-summary");
  const profiles = selectedBenchProfiles();
  if (!profiles.length) {
    summary.className = "result err";
    summary.textContent = "Select at least one profile.";
    return;
  }
  let questionInfo;
  try {
    questionInfo = currentQuestionSetInfo();
  } catch (err) {
    summary.className = "result err";
    summary.textContent = `Question set error: ${err.message}`;
    return;
  }

  const deviceChoice = ($("#bench-device")?.value || "auto").toLowerCase();
  const queue = buildRunQueue(profiles, deviceChoice, questionInfo);
  state.currentQueue = queue;
  renderRunQueue(queue);
  $("#run-table").classList.remove("hidden");
  const categoryRollup = $("#run-category-rollup");
  if (categoryRollup) categoryRollup.innerHTML = "";
  summary.className = "result";
  summary.textContent = `Running ${queue.length} queued benchmark${queue.length === 1 ? "" : "s"}.`;
  setProgress(0);
  setActiveRun(queue[0]);
  btn.disabled = true;
  cancelBtn.hidden = false;
  $("#btn-export").disabled = true;
  $("#btn-contribute").disabled = true;

  state.queueCancelled = false;
  // The global Cancel stops the whole queue: flag it, then abort whatever run
  // is currently active so its stream unwinds immediately.
  const onCancel = () => {
    state.queueCancelled = true;
    state.activeController?.abort();
  };
  cancelBtn.addEventListener("click", onCancel);
  const completed = [];
  const timeout = Number($("#bench-timeout").value) || 240;
  const repetitions = Math.max(1, Math.min(10, Number($("#bench-repetitions")?.value) || 1));
  const activeTimer = setInterval(() => {
    const active = queue.find((q) => ["deploying", "running"].includes(q.status));
    if (active) setActiveRun(active, active.done || 0, active.total || 0);
  }, 1000);
  try {
    let index = 0;
    while (index < queue.length) {
      if (state.queueCancelled) throw new DOMException("Aborted", "AbortError");
      const item = queue[index];
      if (!item || item.status !== "waiting") {
        index++;
        continue;
      }
      // One controller per item so the active-run Stop button can abort just
      // this run while the queue keeps going.
      const itemController = new AbortController();
      state.activeController = itemController;
      const run = await runQueueItem(item, questionInfo, timeout, repetitions, itemController);
      state.activeController = null;
      if (run) {
        completed.push(run);
        addBenchmarkRuns([run], true);
      }
      index++;
    }
    if (completed.length) {
      summary.className = "result ok";
      summary.textContent = `Completed ${completed.length}/${queue.length} benchmark run${queue.length === 1 ? "" : "s"}.`;
    } else {
      renderBenchmarkWorkspace();
      summary.className = "result err";
      summary.textContent = "No benchmark runs completed.";
    }
  } catch (err) {
    if (err.name === "AbortError") {
      queue.filter((q) => ["waiting", "deploying", "running"].includes(q.status)).forEach((q) => {
        q.status = "failed";
        q.current = "Cancelled";
      });
      summary.className = "result";
      summary.textContent = "Benchmark queue cancelled.";
      toast("Benchmark queue cancelled.", "info");
    } else {
      // A mid-stream failure (dropped connection, server crash) must not leave
      // the active row stuck "running"/"deploying" forever - removeQueuedRun
      // refuses to remove rows in those states, and the dashboard would keep
      // showing it as an active run indefinitely otherwise.
      queue.filter((q) => ["waiting", "deploying", "running"].includes(q.status)).forEach((q) => {
        q.status = "failed";
        q.error = q.error || err.message || "Unexpected error";
        q.current = q.error;
        removeLiveBenchmarkRun(q.id, false);
      });
      summary.className = "result err";
      summary.textContent = `Benchmark queue failed: ${err.message}`;
    }
    renderRunQueue(queue);
  } finally {
    clearInterval(activeTimer);
    setActiveRun(null);
    state.activeController = null;
    state.queueCancelled = false;
    cancelBtn.hidden = true;
    cancelBtn.removeEventListener("click", onCancel);
    btn.disabled = false;
    await onModelStateInvalidated();
    updateBenchmarkSummary();
  }
}

// ---------------------------------------------------------------------------
// Tab 2 - Report cards: export + compare
// ---------------------------------------------------------------------------

function runsForDashboard() {
  const runs = allBenchmarkRuns().filter((r) => (r.tests || []).length);
  if (!state.selectedRunIds.length) return runs;
  const selected = new Set(state.selectedRunIds);
  return runs.filter((r) => selected.has(r.id));
}

function renderDetailedResults(runs = runsForDashboard()) {
  const table = $("#run-table");
  const tbody = $("tbody", table);
  const runFilter = $("#detail-filter-run");
  const catFilter = $("#detail-filter-category");
  const resultFilter = $("#detail-filter-result")?.value || "";
  const selectedRun = runFilter?.value || "";
  const selectedCat = catFilter?.value || "";
  const allRuns = allBenchmarkRuns();
  const categories = Array.from(new Set(allRuns.flatMap((r) => (r.tests || []).map((t) => t.category || "?")))).sort();
  if (runFilter) {
    const current = runFilter.value;
    runFilter.innerHTML = `<option value="">All runs</option>${allRuns.map((r) => `<option value="${esc(r.id)}">${esc(runLabel(r))}</option>`).join("")}`;
    runFilter.value = current;
  }
  if (catFilter) {
    const current = catFilter.value;
    catFilter.innerHTML = `<option value="">All categories</option>${categories.map((c) => `<option value="${esc(c)}">${esc(c)}</option>`).join("")}`;
    catFilter.value = current;
  }
  const rows = [];
  for (const r of allRuns) {
    if (selectedRun && r.id !== selectedRun) continue;
    for (const t of r.tests || []) {
      if (selectedCat && t.category !== selectedCat) continue;
      if (resultFilter === "pass" && !t.success) continue;
      if (resultFilter === "fail" && t.success) continue;
      if (resultFilter === "slow" && t.success) continue;
      rows.push({ run: r, test: t });
    }
  }
  if (resultFilter === "slow") rows.sort((a, b) => Number(b.test.elapsed_seconds || 0) - Number(a.test.elapsed_seconds || 0));
  tbody.innerHTML = "";
  rows.slice(0, 250).forEach(({ run, test }) => appendResultRow(tbody, test, runLabel(run)));
  table.classList.toggle("hidden", rows.length === 0);
}

function renderRunLibrary() {
  const body = $("#run-library");
  if (!body) return;
  const runs = allBenchmarkRuns();
  if (!runs.length) {
    body.innerHTML = `<div class="muted small">No active, completed, or imported runs yet.</div>`;
    return;
  }
  const selected = new Set(state.selectedRunIds);
  body.innerHTML = runs
    .map((r) => {
      const live = r.source === "live-run";
      const badgeClass = live ? "cpu" : r.source === "imported-card" ? "cpu" : "on";
      const source = live ? `${r.status || "running"} · ${r.progress || 0}%` : `${r.source} · ${new Date(r.createdAt).toLocaleString()}`;
      const action = live
        ? `<span class="muted small">Active</span>`
        : `<button class="btn compact danger run-delete" data-id="${esc(r.id)}" title="Remove this run from history" aria-label="Remove run">×</button>`;
      return `<div class="run-library-row${selected.has(r.id) ? " selected" : ""}${live ? " live" : ""}">
      <label class="run-library-pick">
        <input type="checkbox" value="${esc(r.id)}"${selected.has(r.id) ? " checked" : ""} />
        <span><b>${esc(runLabel(r))}</b><span class="muted small">${esc(source)}</span></span>
        <span class="badge ${badgeClass}">${esc(r.summary.passed)}/${esc(r.summary.tests)}</span>
      </label>
      ${action}
    </div>`;
    })
    .join("");
}

// Remove a single run from the local history (and any selection/baseline that
// pointed at it), so the library can be pruned without clearing everything.

function deleteBenchmarkRun(id) {
  if (state.benchHistoryServer) {
    // Best-effort: the run may or may not have a server copy.
    void postJSON("/benchmark/history/delete", { id }).catch(() => {});
  }
  state.benchmarkRuns = state.benchmarkRuns.filter((r) => r.id !== id);
  state.selectedRunIds = state.selectedRunIds.filter((x) => x !== id);
  if (state.compareBaselineId === id) state.compareBaselineId = state.selectedRunIds[0] || null;
  if (state.activeRunId === id) state.activeRunId = null;
  if (state.lastRun?.id === id) state.lastRun = null;
  saveBenchmarkRuns();
  renderBenchmarkWorkspace();
}

function renderCompareControls() {
  const baseline = $("#compare-baseline");
  const runs = allBenchmarkRuns();
  const selectedCompleted = state.selectedRunIds.filter((id) => state.benchmarkRuns.some((r) => r.id === id)).length;
  if (baseline) {
    baseline.innerHTML = state.selectedRunIds
      .map((id) => {
        const r = runs.find((x) => x.id === id);
        return r ? `<option value="${esc(id)}">${esc(runLabel(r))}</option>` : "";
      })
      .join("");
    baseline.value = state.compareBaselineId || state.selectedRunIds[0] || "";
  }
  const status = $("#compare-status");
  if (status) status.textContent = `${state.selectedRunIds.length} selected. Select 2 or more runs to compare.`;
  $("#btn-export-selected").disabled = selectedCompleted === 0;
  const hasActiveRun = !(state.benchmarkRuns.find((r) => r.id === state.activeRunId) || state.lastRun);
  $("#btn-export").disabled = hasActiveRun;
  $("#btn-contribute").disabled = hasActiveRun;
}

function selectedComparisonRuns() {
  const runs = allBenchmarkRuns();
  return state.selectedRunIds.map((id) => runs.find((r) => r.id === id)).filter(Boolean);
}

function regressionDiffsHtml(resp) {
  const diffRows = (resp.dimension_diffs || [])
    .map(
      (d) =>
        `<tr class="${d.changed ? "regression-changed" : ""}"><td>${esc(d.dimension)}</td><td>${esc(d.a ?? "-")}</td><td>${esc(d.b ?? "-")}</td><td>${d.changed ? "⚠ changed" : "same"}</td></tr>`
    )
    .join("");
  const sd = resp.summary_delta || {};
  const deltaCell = (v, unit = "") => (v == null ? "-" : `${v > 0 ? "+" : ""}${esc(v)}${unit}`);
  return `<h3 class="sub">What changed (regression check)</h3>
    <div class="row gap wrap regression-deltas">
      <span class="badge">tok/s Δ ${deltaCell(sd.avg_tokens_per_second)}</span>
      <span class="badge">TTFT Δ ${deltaCell(sd.avg_ttft_ms, " ms")}</span>
      <span class="badge">peak VRAM Δ ${deltaCell(sd.peak_vram_mb, " MB")}</span>
      <span class="badge">accuracy Δ ${deltaCell(sd.avg_accuracy)}</span>
    </div>
    ${
      diffRows
        ? `<div class="table-wrap"><table class="results"><thead><tr><th>Dimension</th><th>${esc(resp.label_a)}</th><th>${esc(resp.label_b)}</th><th></th></tr></thead><tbody>${diffRows}</tbody></table></div>`
        : `<div class="muted small">No provenance recorded on these runs to compare dimensions.</div>`
    }`;
}

async function renderComparison(runs, baseline) {
  if (!runs.length || !baseline) {
    $("#compare-body").innerHTML = "";
    $("#response-drawer").classList.add("hidden");
    return;
  }
  state.compareBaselineId = baseline.id;
  const baseSummary = baseline.summary || summaryFromTests(baseline.tests);
  const delta = (a, b) => (a == null || b == null ? "-" : (Number(b) - Number(a)).toFixed(3));
  const rows = runs
    .map((r) => {
      const s = r.summary || summaryFromTests(r.tests);
      return `<tr><td><b>${esc(runLabel(r))}</b>${r.id === baseline.id ? ` <span class="badge on">baseline</span>` : ""}</td>
        <td class="num">${esc(s.passed)}/${esc(s.tests)}</td>
        <td class="num">${esc(s.avg_accuracy)} <span class="muted">(${esc(delta(baseSummary.avg_accuracy, s.avg_accuracy))})</span></td>
        <td class="num">${esc(s.avg_latency_s)}s <span class="muted">(${esc(delta(baseSummary.avg_latency_s, s.avg_latency_s))})</span></td>
        <td class="num">${esc(s.avg_tokens_per_second ?? "-")} <span class="muted">(${esc(delta(baseSummary.avg_tokens_per_second, s.avg_tokens_per_second))})</span></td></tr>`;
    })
    .join("");
  const tests = Array.from(new Set(runs.flatMap((r) => (r.tests || []).map((t) => t.name)))).sort();
  const testButtons = tests
    .map((name) => `<button class="btn compact response-compare-btn" data-test="${esc(name)}">${esc(name)}</button>`)
    .join("");
  $("#compare-body").innerHTML = `<div class="table-wrap"><table class="results">
    <thead><tr><th>Run</th><th class="num">Completed</th><th class="num">Accuracy Δ</th><th class="num">Latency Δ</th><th class="num">tok/s Δ</th></tr></thead>
    <tbody>${rows}</tbody></table></div>
    <h3 class="sub">Response detail</h3><div class="response-test-list">${testButtons}</div>
    <div id="regression-diffs"></div>`;
  // The richer provenance-aware diff (dimension changes, TTFT/VRAM deltas) is
  // only meaningful pairwise - a 3+ run comparison keeps the table above only.
  if (runs.length === 2) {
    const other = runs.find((r) => r.id !== baseline.id) || runs[1];
    try {
      const resp = await postJSON("/benchmark/compare", {
        card_a: runToCardPayload(baseline),
        card_b: runToCardPayload(other),
      });
      if (resp.success) {
        const slot = $("#regression-diffs");
        if (slot) slot.innerHTML = regressionDiffsHtml(resp);
      }
    } catch {
      // Regression detail is a bonus panel; the primary comparison table above still rendered.
    }
  }
}

function renderAutoComparison() {
  const runs = selectedComparisonRuns();
  const baseline = allBenchmarkRuns().find((r) => r.id === state.compareBaselineId) || runs[0];
  if (runs.length >= 2 && baseline) {
    renderComparison(runs, baseline);
  } else {
    $("#compare-body").innerHTML = "";
    $("#response-drawer").classList.add("hidden");
  }
}

function renderBenchmarkWorkspace() {
  const runs = runsForDashboard();
  const hasRuns = runs.length > 0;
  $("#results-dashboard-card")?.classList.toggle("hidden", !hasRuns);
  $("#detailed-results-card")?.classList.toggle("hidden", !hasRuns);
  $("#winner-badges").innerHTML = renderWinners(runs);
  $("#leaderboard-body").innerHTML = renderLeaderboard(runs);
  $("#heatmap-body").innerHTML = renderHeatmap(runs);
  $("#scatter-body").innerHTML = renderScatter(runs);
  $("#matrix-body").innerHTML = renderMatrix(runs);
  renderRunLibrary();
  renderCompareControls();
  renderAutoComparison();
  renderDetailedResults(runs);
  updateBenchmarkSummary();
}

// Map a normalized run object (camelCase, client-side) to the snake_case
// card shape report.py's build_card/benchmark_compare expect. One adapter
// reused by export (so provenance/peak VRAM survive a round-trip through the
// downloaded HTML) and by the regression-detection compare call below.

function runToCardPayload(run) {
  return {
    profile: run.profile,
    model_id: run.modelId,
    device: run.actualDevice || run.requestedDevice,
    hardware: run.hardware,
    tests: run.tests,
    summary: run.summary,
    category_summary: run.category_summary,
    provenance: run.provenance || {},
    repetitions: run.repetitions || 1,
    variance: run.variance || {},
    peak_vram_mb: run.peakVramMb ?? null,
  };
}

async function exportOneRun(run, btn) {
  busy(btn, true);
  try {
    const out = await postJSON("/benchmark/export", runToCardPayload(run));
    if (!out.success) throw new Error(out.error || "export failed");
    const name = (run.profile || run.modelId || "model").replace(/[^\w.-]+/g, "_");
    const devSuffix = run.actualDevice || run.requestedDevice ? `-${run.actualDevice || run.requestedDevice}` : "";
    downloadFile(`localdeploy-card-${name}${devSuffix}.html`, out.html, "text/html");
    toast("Report card downloaded.", "success");
  } finally {
    busy(btn, false);
  }
}

async function exportCard() {
  const run = state.benchmarkRuns.find((r) => r.id === state.activeRunId) || state.lastRun;
  if (!run) {
    toast("Select or run a benchmark first.", "error");
    return;
  }
  const btn = $("#btn-export");
  try {
    await exportOneRun(run, btn);
  } catch (err) {
    toast(`Export failed: ${err.message}`, "error");
  }
}

// ---------------------------------------------------------------------------
// Community benchmark sharing - local-only preview/save (Release R8)
// ---------------------------------------------------------------------------

async function contributeRun() {
  const run = state.benchmarkRuns.find((r) => r.id === state.activeRunId) || state.lastRun;
  if (!run) {
    toast("Select or run a benchmark first.", "error");
    return;
  }
  const btn = $("#btn-contribute");
  busy(btn, true);
  try {
    const out = await postJSON("/system/community/preview", { card: runToCardPayload(run) });
    if (!out.success) throw new Error(out.error || "Preview failed.");
    const overlay = simpleModal(
      "Save an anonymized benchmark",
      `This anonymized snapshot is saved to a file on this computer. Nothing is sent anywhere. ${esc(out.note)}`,
      `<pre class="log manifest-yaml">${esc(JSON.stringify(out.would_share, null, 2))}</pre>
       <details style="margin-top:0.5rem"><summary class="muted small">Never included</summary>
         <div class="muted small">${out.excluded_fields.map(esc).join(", ")}</div>
       </details>
       <div class="row gap wrap" style="margin-top:0.75rem">
         <button class="btn primary" id="contribute-save-local">Save locally</button>
       </div>
       <div id="contribute-save-status" class="muted small"></div>`
    );
    $("#contribute-save-local", overlay)?.addEventListener("click", async (e) => {
      busy(e.currentTarget, true);
      try {
        const saved = await postJSON("/system/community/export", { card: runToCardPayload(run) });
        if (!saved.success) throw new Error(saved.error || "Save failed.");
        $("#contribute-save-status", overlay).textContent = `Saved to ${saved.path}. ${saved.note}`;
        toast("Saved anonymized benchmark locally.", "success");
      } catch (err) {
        $("#contribute-save-status", overlay).textContent = `Save failed: ${err.message}`;
      } finally {
        busy(e.currentTarget, false);
      }
    });
  } catch (err) {
    toast(`Could not build preview: ${err.message}`, "error");
  } finally {
    busy(btn, false);
  }
}

async function exportSelectedRuns() {
  const runs = state.selectedRunIds.map((id) => state.benchmarkRuns.find((r) => r.id === id)).filter(Boolean);
  if (!runs.length) {
    toast("Select at least one completed run to export.", "error");
    return;
  }
  const btn = $("#btn-export-selected");
  if (runs.length === 1) {
    try {
      await exportOneRun(runs[0], btn);
    } catch (err) {
      toast(`Export failed: ${err.message}`, "error");
    }
    return;
  }
  downloadFile(
    `localdeploy-benchmark-runs-${new Date().toISOString().slice(0, 10)}.json`,
    JSON.stringify({ kind: "localdeploy.run_bundle", version: 1, runs }, null, 2),
    "application/json"
  );
  toast("Selected runs exported as JSON bundle.", "success");
}

function readCardFiles(input) {
  const files = Array.from(input.files || []);
  if (!files.length) return;
  Promise.all(
    files.map(
      (file) =>
        new Promise((resolve) => {
          const reader = new FileReader();
          reader.onload = () => {
            const text = String(reader.result || "");
            let runs = [];
            try {
              const parsed = JSON.parse(text);
              if (parsed.kind === "localdeploy.run_bundle" && Array.isArray(parsed.runs)) runs = parsed.runs;
            } catch {
              /* maybe a report card HTML */
            }
            const card = runs.length ? null : extractCard(text);
            if (card) runs = [card];
            resolve(runs.map((r) => ({ ...normalizeRunRecord(r, "imported-card"), source: "imported-card" })));
          };
          reader.onerror = () => resolve([]);
          reader.readAsText(file);
        })
    )
  ).then((groups) => {
    const runs = groups.flat();
    if (!runs.length) toast("No LocalDeploy report cards found.", "error");
    else {
      addBenchmarkRuns(runs, true);
      toast(`Imported ${runs.length} run${runs.length === 1 ? "" : "s"}.`, "success");
    }
  });
  input.value = "";
}

function compareSelectedRuns() {
  const runs = selectedComparisonRuns();
  const baseline = state.benchmarkRuns.find((r) => r.id === ($("#compare-baseline")?.value || state.compareBaselineId)) || runs[0];
  if (runs.length < 2 || !baseline) {
    toast("Select 2 or more runs to compare.", "error");
    return;
  }
  renderComparison(runs, baseline);
}

function renderResponseDrawer(testName, runs) {
  const drawer = $("#response-drawer");
  drawer.classList.remove("hidden");
  drawer.innerHTML = renderResponseComparison({ testName, runs });
}

function handleComparisonClick(event) {
  const compareButton = event.target.closest(".response-compare-btn");
  if (compareButton) renderResponseDrawer(compareButton.dataset.test, selectedComparisonRuns());
}

function handleResponseDrawerClick(event) {
  if (event.target.closest("#close-response-drawer")) $("#response-drawer")?.classList.add("hidden");
}

// ---------------------------------------------------------------------------
// Tab 1 - Auto-pick a profile (Step 14)
// ---------------------------------------------------------------------------

function renderRecommendProgress(body, candidates, current) {
  const rows = candidates
    .map((c) => {
      const cls = c.status === "done" ? "ok" : c.status === "running" ? "running" : "";
      const detail =
        c.status === "done"
          ? `acc ${c.avg_accuracy} · ${c.avg_latency_s}s`
          : c.status === "running"
            ? `${esc(c.currentTest || "starting…")}`
            : "queued";
      return `<li class="tune-candidate-row ${cls}">
        <span class="spin-inline ${c.status === "running" ? "" : "hidden"}"></span>
        <b>${esc(c.profile)}</b><span class="muted small">${esc(detail)}</span>
      </li>`;
    })
    .join("");
  body.innerHTML = `
    <div class="tune-progress">
      <div class="running-top">
        <div>
          <div class="model-title">${esc(current)}</div>
          <div class="muted small">Evaluating candidates against the ${esc(fmtMb(targetVram()))} GPU budget</div>
        </div>
        <span class="spin-inline"></span>
      </div>
      <div class="run-progress-track"><div class="run-progress-fill indeterminate" role="progressbar" aria-label="Tune progress"></div></div>
      <ul class="tune-candidate-list">${rows || '<li class="muted small">Fit-checking saved profiles…</li>'}</ul>
    </div>`;
}

// Named scoring tilts for the three preset buttons. Each reuses the same
// /system/recommend/stream ranking - only the accuracy/speed/headroom weights
// change - so presets stay valid for any user's config.json profiles instead
// of hardcoding specific model names.

const RECOMMEND_PRESETS = {
  safe: { quality_weight: 0.4, speed_weight: 0.3, headroom_weight: 0.3 },
  quality: { quality_weight: 0.8, speed_weight: 0.1, headroom_weight: 0.1 },
  fast: { quality_weight: 0.15, speed_weight: 0.65, headroom_weight: 0.2 },
};

async function recommendTune(weights = null) {
  const btn = $("#btn-recommend");
  const body = $("#recommend-body");
  busy(btn, true);
  $$(".preset-btn").forEach((b) => (b.disabled = true));
  const candidates = [];
  const byProfile = {};
  const addCandidate = (profile) => {
    if (byProfile[profile]) return byProfile[profile];
    const row = { profile, status: "queued" };
    byProfile[profile] = row;
    candidates.push(row);
    return row;
  };
  renderRecommendProgress(body, candidates, "Fit-checking saved profiles…");
  let finalResult = null;
  // Quick (default) samples a handful of the fastest tests per candidate to
  // keep tuning snappy; Full runs the entire test bench for a slower, more
  // thorough score. sample_size is clamped server-side to the test count.
  const fullBench = $("#recommend-full-bench")?.checked;
  const sampleSize = fullBench ? 9999 : 3;
  try {
    const result = await postMaybeStream(
      "/system/recommend/stream",
      { free_vram_mb: targetVram(), sample_size: sampleSize, ...(weights || {}) },
      (evt) => {
        if (evt.event === "recommend_start") {
          (evt.candidates || []).forEach(addCandidate);
          renderRecommendProgress(body, candidates, "Benchmarking candidates…");
        } else if (evt.event === "candidate_start") {
          const row = addCandidate(evt.profile);
          row.status = "running";
          renderRecommendProgress(body, candidates, `Benchmarking ${evt.profile}…`);
        } else if (evt.event === "test_start") {
          const row = addCandidate(evt.profile);
          row.currentTest = evt.name;
          renderRecommendProgress(body, candidates, `Benchmarking ${evt.profile}…`);
        } else if (evt.event === "candidate_end") {
          const row = addCandidate(evt.profile);
          row.status = "done";
          row.avg_accuracy = evt.avg_accuracy;
          row.avg_latency_s = evt.avg_latency_s;
          renderRecommendProgress(body, candidates, `Benchmarking candidates…`);
        } else if (evt.event === "error") {
          finalResult = { success: false, error: evt.error };
        } else if (evt.event === "recommend_end") {
          finalResult = { success: true, ...evt };
        }
      }
    );
    const res = result.streamed ? finalResult : result.json;
    if (!res) throw new Error("Stream ended without a result.");
    if (!res.success && res.success !== undefined) {
      body.innerHTML = `<div class="muted">${esc(res.error || "Could not run.")}</div>`;
      return;
    }
    if (!res.recommended) {
      body.innerHTML = `<div class="muted">${esc(res.message || "No profile fits the available VRAM.")}</div>`;
      return;
    }
    const rec = res.recommended;
    const rows = (res.candidates || [])
      .map((c) => {
        const star = c.profile === rec.profile ? " ★" : "";
        const p = state.profileData[c.profile] || {};
        const model = p.model_id || state.profileModels[c.profile] || c.profile;
        const installed = installedStatusForProfile(c.profile);
        return `<tr><td><b>${esc(c.profile)}${star}</b><div class="muted small">${esc(model)}</div></td>
          <td>${esc(p.backend || "ollama")}</td>
          <td><span class="badge ${installed.cls}">${esc(installed.label)}</span></td>
          <td class="num">${esc(c.avg_accuracy)}</td>
          <td class="num">${esc(c.avg_latency_s)}s</td>
          <td class="num">${esc(c.margin_gb ?? "-")}</td>
          <td class="num">${esc(c.score)}</td></tr>`;
      })
      .join("");
    const skipped = (res.skipped || [])
      .map((s) => `<li>${esc(s.profile)} - ${esc(s.reason)}${s.required_gb ? ` (~${esc(s.required_gb)} GB)` : ""}</li>`)
      .join("");
    const sampleTests = (res.sample_tests || []).map((t) => `<code>${esc(t)}</code>`).join(", ");
    body.innerHTML = `
      <div class="result ok tune-result">
        <div><b>Recommended:</b> ${esc(rec.profile)}</div>
        <div class="muted small">Evaluated enabled saved profiles that fit the selected GPU budget. Models must already be pulled or reachable; failed benchmark responses score 0.</div>
        <div class="muted small">Why: ${esc(rec.reasoning)}. Score favors accuracy first, then speed, then VRAM headroom.</div>
        ${sampleTests ? `<div class="muted small">Benchmarked sample tests: ${sampleTests}</div>` : ""}
        <button class="btn set-default-btn" data-profile="${esc(rec.profile)}">Set as default</button>
      </div>
      <div class="table-wrap" style="margin-top:.5rem"><table class="results">
        <thead><tr><th>Saved profile / model</th><th>Backend</th><th>Status</th><th class="num">Accuracy</th><th class="num">Latency</th><th class="num">Headroom</th><th class="num">Score</th></tr></thead>
        <tbody>${rows}</tbody></table></div>
      ${skipped ? `<h3 class="sub">Skipped (won’t fit)</h3><ul class="err-list">${skipped}</ul>` : ""}`;
    const sd = body.querySelector(".set-default-btn");
    if (sd) sd.addEventListener("click", () => setDefaultProfile(sd.dataset.profile, sd));
  } catch (err) {
    body.innerHTML = `<div class="muted">Tuning failed - ${esc(err.message)}</div>`;
    toast(`Tune failed: ${err.message}`, "error");
  } finally {
    busy(btn, false);
    $$(".preset-btn").forEach((b) => (b.disabled = false));
  }
}

async function loadGraderTypes() {
  try {
    const benchInfo = await getJSON("/benchmark/test-bench");
    updateBenchHelp(benchInfo);
    const example = await getJSON("/benchmark/example");
    // grader_types are returned by /benchmark/validate; fetch them cheaply.
    const report = await postJSON("/benchmark/validate", example);
    if (report.grader_types) {
      $("#grader-types").innerHTML = report.grader_types.map((t) => `<code>${esc(t)}</code>`).join(", ");
    }
  } catch {
    $("#grader-types").textContent = "contains_all, json_array_min_len, number_within, exact_match, classification_set";
  }
}

async function loadBenchmarkPacks() {
  const sel = $("#bench-pack");
  if (!sel) return;
  try {
    const data = await getJSON("/benchmark/packs");
    if (!data.success) return;
    sel.innerHTML =
      `<option value="">Full suite</option>` +
      data.packs
        .map((p) => `<option value="${esc(p.id)}" title="${esc(p.description)}">${esc(p.label)} (${esc(p.test_count)})</option>`)
        .join("");
  } catch {
    // Packs are a convenience filter; the full suite remains the default on failure.
  }
}

export { categorySummary, normalizeRunRecord, renderBenchmarkWorkspace, summaryFromTests };

export async function refreshBenchmarkMetadata() {
  await Promise.allSettled([loadGraderTypes(), loadBenchmarkPacks()]);
}

function handleQueueClick(event) {
  const button = event.target.closest("button");
  if (!button) return;
  if (button.id === "btn-clear-finished") clearFinishedQueue();
  else if (button.classList.contains("queue-remove")) removeQueuedRun(button.dataset.id);
  else if (button.classList.contains("queue-move")) moveQueuedRun(button.dataset.id, Number(button.dataset.delta));
}

function handleRunLibraryClick(event) {
  const button = event.target.closest(".run-delete");
  if (button) deleteBenchmarkRun(button.dataset.id);
}

function handleRunLibraryChange(event) {
  const input = event.target.closest('input[type="checkbox"]');
  if (!input) return;
  const ids = new Set(state.selectedRunIds);
  if (input.checked) ids.add(input.value);
  else ids.delete(input.value);
  state.selectedRunIds = Array.from(ids);
  if (!state.selectedRunIds.includes(state.compareBaselineId)) state.compareBaselineId = state.selectedRunIds[0] || null;
  state.activeRunId = input.checked ? input.value : state.activeRunId;
  renderBenchmarkWorkspace();
}

export function initBenchmark(options = {}) {
  if (initialized) return;
  initialized = true;
  onModelStateInvalidated = options.onModelStateInvalidated || onModelStateInvalidated;
  setDefaultProfileAction = options.setDefaultProfile || setDefaultProfileAction;
  loadBenchmarkRuns(); initServerHistoryToggle(); renderBenchmarkWorkspace();
  $("#bench-profile-chips")?.addEventListener("click", handleBenchmarkProfilesClick);
  $("#bench-profile-chips")?.addEventListener("change", handleBenchmarkProfilesChange);
  $("#run-queue")?.addEventListener("click", handleQueueClick);
  $("#run-library")?.addEventListener("click", handleRunLibraryClick);
  $("#run-library")?.addEventListener("change", handleRunLibraryChange);
  $("#compare-body")?.addEventListener("click", handleComparisonClick);
  $("#response-drawer")?.addEventListener("click", handleResponseDrawerClick);
  $("#btn-builtin-bench")?.addEventListener("click", useBuiltInBench);
  $("#btn-example")?.addEventListener("click", loadExample);
  $("#btn-validate")?.addEventListener("click", validateSet);
  $("#btn-run")?.addEventListener("click", runBenchmark);
  $("#btn-stop-active")?.addEventListener("click", () => { if (state.activeController && !state.activeController.signal.aborted) { state.activeController.abort(); toast("Stopping the current run…", "info"); } });
  $("#btn-select-all-runs")?.addEventListener("click", () => { state.selectedRunIds = allBenchmarkRuns().map((run) => run.id); if (!state.selectedRunIds.includes(state.compareBaselineId)) state.compareBaselineId = state.selectedRunIds[0] || null; renderBenchmarkWorkspace(); });
  $("#btn-deselect-all-runs")?.addEventListener("click", () => { state.selectedRunIds = []; state.compareBaselineId = null; renderBenchmarkWorkspace(); });
  $("#upload-json")?.addEventListener("change", (event) => uploadFile(event.target));
  $("#btn-recommend")?.addEventListener("click", () => recommendTune());
  $$(".preset-btn").forEach((button) => button.addEventListener("click", () => recommendTune(RECOMMEND_PRESETS[button.dataset.preset])));
  $("#btn-export")?.addEventListener("click", exportCard);
  $("#btn-export-selected")?.addEventListener("click", exportSelectedRuns);
  $("#btn-contribute")?.addEventListener("click", contributeRun);
  $("#btn-compare")?.addEventListener("click", compareSelectedRuns);
  $("#card-import")?.addEventListener("change", (event) => readCardFiles(event.target));
  $("#btn-clear-runs")?.addEventListener("click", () => { const count = state.benchmarkRuns.length; if (!count) { toast("No saved runs to clear.", "info"); return; } if (!window.confirm(`Clear all ${count} saved run${count === 1 ? "" : "s"} (including imported cards)? This cannot be undone.`)) return; state.benchmarkRuns = []; state.selectedRunIds = []; state.compareBaselineId = null; state.activeRunId = null; state.lastRun = null; saveBenchmarkRuns(); renderBenchmarkWorkspace(); toast(`Cleared ${count} run${count === 1 ? "" : "s"}.`, "success"); });
  $("#compare-baseline")?.addEventListener("change", (event) => { state.compareBaselineId = event.target.value || null; renderAutoComparison(); });
  $("#bench-profile-filter")?.addEventListener("input", renderBenchmarkProfileChips);
  ["#detail-filter-run", "#detail-filter-category", "#detail-filter-result"].forEach((selector) => $(selector)?.addEventListener("change", () => renderDetailedResults()));
  $("#qs-editor")?.addEventListener("keydown", (event) => { if ((event.metaKey || event.ctrlKey) && event.key === "Enter") { event.preventDefault(); void runBenchmark(); } });
  $("#qs-editor")?.addEventListener("input", () => { state.questionSetValidation = null; updateBenchmarkSummary(); });
}
