"use strict";

import { $, $$, busy, esc, fmtDuration, fmtMb, getJSON, postJSON, skeletonHtml, toast } from "./shared.js?v=20261020-ui33";

const state = {
  hardware: null,
  vramTotalMb: null,
  vramFreeMb: null,
  vramBudgetMb: null,
  ollamaStatus: { installed: null, reachable: null },
  modelContext: { installedCount: 0, running: [] },
  monitor: { timer: null, tpsHistory: [] },
};
const listeners = new Set();
let initialized = false;
let onNavigate = () => {};
let onModelStateInvalidated = async () => {};

function clone(value) {
  return value == null ? value : structuredClone(value);
}

function readonly(value) {
  if (value && typeof value === "object" && !Object.isFrozen(value)) {
    Object.values(value).forEach(readonly);
    Object.freeze(value);
  }
  return value;
}

export function getSystemSnapshot() {
  return readonly({
    hardware: clone(state.hardware),
    vramTotalMb: state.vramTotalMb,
    vramFreeMb: state.vramFreeMb,
    vramBudgetMb: targetVram(),
    ollamaStatus: { ...state.ollamaStatus },
  });
}

function notifySystemChanges() {
  const snapshot = getSystemSnapshot();
  listeners.forEach((listener) => listener(snapshot));
}

export function subscribeToSystemChanges(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function updateModelContext(summary = {}) {
  state.modelContext = {
    installedCount: Number(summary.installedCount || 0),
    running: clone(summary.running || []),
  };
  updateHardwareVramUI();
  updateProgressRail();
}

function updateHwChip(hw) {
  const chip = $("#hw-chip");
  const ram = hw?.system?.ram_total_mb ? ` · ${fmtMb(hw.system.ram_total_mb)} RAM` : "";
  if (hw && hw.gpu_available && hw.gpus?.[0]) {
    const g = hw.gpus[0];
    const pool = hw.gpu_summary?.best_pool_total_mb ?? g.vram_total_mb;
    const count = hw.gpu_summary?.gpu_count || hw.gpus.length;
    const mem = g.unified_memory ? "unified memory" : `${fmtMb(pool)} compatible VRAM`;
    chip.innerHTML = `<span class="dot"></span>${count > 1 ? `${count} GPUs` : esc(g.name)} · ${mem}${ram}`;
  } else {
    chip.innerHTML = `<span class="dot none"></span>CPU only${ram}`;
  }
  chip.classList.remove("hidden");
}

function syncHardwareState(hw) {
  const g = hw?.gpus?.[0];
  state.vramTotalMb = hw?.gpu_summary?.best_pool_total_mb ?? g?.vram_total_mb ?? null;
  state.vramFreeMb = hw?.gpu_summary?.best_pool_free_mb ?? g?.vram_free_mb ?? null;
  if (state.vramBudgetMb == null && state.vramTotalMb != null) {
    state.vramBudgetMb = state.vramTotalMb;
    const input = $("#vram-budget-gb");
    if (input && !input.value.trim()) input.value = (state.vramBudgetMb / 1024).toFixed(1);
  }
  updateHardwareVramUI();
  updateVramBudgetUI();
}

function setVramBudgetMb(mb) {
  if (mb == null || !Number.isFinite(Number(mb)) || Number(mb) <= 0) return;
  state.vramBudgetMb = Math.round(Number(mb));
  const input = $("#vram-budget-gb");
  if (input) input.value = (state.vramBudgetMb / 1024).toFixed(1);
  updateVramBudgetUI();
  notifySystemChanges();
}

function updateVramBudgetUI() {
  const label = $("#vram-budget-label");
  const live = $("#vram-live-label");
  const help = $("#vram-budget-help");
  const meter = $("#vram-budget-meter");
  if (!label || !live || !help || !meter) return;
  const budget = targetVram();
  label.textContent = budget ? `${fmtMb(budget)} for model fit checks` : "No VRAM budget set";
  const total = state.vramTotalMb;
  const free = state.vramFreeMb;
  live.textContent = total ? `${fmtMb(free)} live free / ${fmtMb(total)} total` : "";
  if (total && free != null) {
    const usedPct = Math.max(0, Math.min(100, ((total - free) / total) * 100));
    const budgetPct = budget ? Math.max(0, Math.min(100, (budget / total) * 100)) : 0;
    meter.innerHTML = `
      <div class="capacity-fill used" style="width:${usedPct.toFixed(1)}%"></div>
      <div class="capacity-marker" style="left:${budgetPct.toFixed(1)}%" title="Fit budget"></div>`;
  } else {
    meter.innerHTML = "";
  }
  if (budget && total && free != null && free < budget * 0.5) {
    help.textContent =
      `Fit checks use a ${fmtMb(budget)} budget. Live free VRAM is lower right now; choose "Current free" to filter against the memory available without unloading models.`;
  } else if (budget) {
    help.textContent = `Installed model badges, saved profile scan, and remote model search all use this budget.`;
  } else {
    help.textContent = "Set a GPU budget to filter models by fit.";
  }
  updateProgressRail();
}

// ---------------------------------------------------------------------------
// Setup progress rail + recommended next action
// ---------------------------------------------------------------------------

function progressRailSteps() {
  return {
    hardware: !!state.hardware,
    budget: !!state.vramBudgetMb,
    model: state.modelContext.installedCount > 0,
    deploy: state.modelContext.running.length > 0,
  };
}

function nextActionHtml(title, detail, targetSel) {
  return `<div><strong>${esc(title)}</strong><div class="muted small">${esc(detail)}</div></div>
    <button class="btn primary compact next-action-go" data-target="${esc(targetSel)}">Take me there</button>`;
}

function updateNextActionCard() {
  const card = $("#next-action-card");
  if (!card) return;
  const steps = progressRailSteps();
  let html = "";
  if (!steps.hardware) {
    html = nextActionHtml(
      "Check hardware first",
      "See your GPU, VRAM, CPU, and RAM before picking a model to pull.",
      "#btn-hardware"
    );
  } else if (!steps.budget) {
    html = nextActionHtml(
      "Set a fit budget",
      "Confirm the GPU budget used for fit checks, pulls, and Hugging Face search.",
      "#vram-budget-gb"
    );
  } else if (!steps.model) {
    html = nextActionHtml(
      "Get your first model",
      "No models installed yet - get a recommended one that fits your hardware.",
      "#btn-recommend-models"
    );
  } else if (!steps.deploy) {
    html = nextActionHtml(
      "Deploy a model",
      "A model is ready - deploy it from Your models to start serving it.",
      "#your-models-card"
    );
  }
  card.innerHTML = html;
  card.classList.toggle("hidden", !html);
  const goBtn = card.querySelector(".next-action-go");
  if (goBtn) {
    goBtn.addEventListener("click", () => {
      const target = document.querySelector(goBtn.dataset.target);
      if (!target) return;
      target.scrollIntoView({ behavior: "smooth", block: "center" });
      target.focus?.();
    });
  }
}

function updateProgressRail() {
  const rail = $("#progress-rail");
  if (!rail) return;
  const steps = progressRailSteps();
  let activeAssigned = false;
  $$(".rail-step", rail).forEach((el) => {
    const done = !!steps[el.dataset.step];
    const active = !done && !activeAssigned;
    el.classList.toggle("done", done);
    el.classList.toggle("active", active);
    if (active) activeAssigned = true;
  });
  updateNextActionCard();
}

function updateHardwareVramUI() {
  const label = $("#hardware-vram-label");
  const live = $("#hardware-vram-live-label");
  const help = $("#hardware-vram-help");
  const meter = $("#hardware-vram-meter");
  if (!label || !live || !help || !meter) return;
  const total = state.vramTotalMb;
  const free = state.vramFreeMb;
  if (!total || free == null) {
    label.textContent = "No separate VRAM reading";
    live.textContent = "";
    meter.innerHTML = "";
    help.textContent = "This machine may be CPU-only or use unified memory.";
    return;
  }
  const used = Math.max(0, total - free);
  const usedPct = Math.max(0, Math.min(100, (used / total) * 100));
  label.textContent = `${fmtMb(used)} used / ${fmtMb(total)} total`;
  live.textContent = `${fmtMb(free)} free now`;
  meter.innerHTML = `<div class="capacity-fill used" style="width:${usedPct.toFixed(1)}%"></div>`;
  const running = (state.modelContext.running || [])
    .map((m) => ({
      name: m.name,
      vramMb: m.size_vram_mb ?? (m.size_vram ? Math.round(m.size_vram / 1_000_000) : null),
    }))
    .filter((m) => m.name && m.vramMb);
  const ollamaMb = running.reduce((sum, m) => sum + m.vramMb, 0);
  if (ollamaMb > 0) {
    const names = running.map((m) => `${m.name} (${fmtMb(m.vramMb)})`).join(", ");
    const unattributed = Math.max(0, used - ollamaMb);
    help.textContent =
      unattributed <= 256 ? `Ollama model VRAM: ${names}.` : "";
  } else {
    help.textContent = "";
  }
}

function cpuRamRows(sys) {
  if (!sys) return "";
  const cores =
    sys.physical_cores != null
      ? `${esc(sys.physical_cores)} cores / ${esc(sys.logical_cores ?? "?")} threads`
      : `${esc(sys.logical_cores ?? "?")} logical cores`;
  const ram =
    sys.ram_total_mb != null
      ? `${fmtMb(sys.ram_total_mb)} total · ${fmtMb(sys.ram_available_mb ?? 0)} available`
      : `<span class="muted">RAM details unavailable.</span>`;
  return `
    <span class="k">CPU</span><span>${esc(sys.cpu_model || "Unknown")}</span>
    <span class="k">Cores</span><span>${cores}</span>
    <span class="k">RAM</span><span>${ram}</span>`;
}

// Read the VRAM budget the user wants to validate against (manual or probed).

function targetVram() {
  const raw = $("#vram-budget-gb")?.value?.trim() || "";
  if (raw !== "") {
    const n = Number(raw);
    return Number.isFinite(n) && n > 0 ? Math.round(n * 1024) : null;
  }
  return state.vramBudgetMb ?? state.vramTotalMb ?? state.vramFreeMb;
}

const MONITOR_POLL_MS = 5000;

const MONITOR_TPS_HISTORY_MAX = 180;

function svgSparkline(series, { unit = "", suffix = "", color = "var(--accent)", max = null } = {}) {
  const values = series.map((v) => (v == null ? null : Number(v)));
  const known = values.filter((v) => v != null);
  if (!known.length) {
    return `<div class="muted small monitor-spark-empty">No data yet - stay on this tab to build history.</div>`;
  }
  const width = 320;
  const height = 64;
  const vmax = max != null ? max : Math.max(...known, 1);
  const n = values.length;
  const points = values
    .map((v, i) => {
      if (v == null) return null;
      const x = n <= 1 ? width : (i / (n - 1)) * width;
      const y = height - Math.max(0, Math.min(1, v / (vmax || 1))) * height;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .filter(Boolean)
    .join(" ");
  const last = known[known.length - 1];
  return `<svg class="monitor-spark" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" role="img" aria-label="Rolling chart, latest ${esc(String(last))}${esc(unit)}">
      <polyline points="${esc(points)}" fill="none" stroke="${color}" stroke-width="2" vector-effect="non-scaling-stroke"></polyline>
    </svg>
    <div class="muted small monitor-spark-caption">latest: ${esc(String(last))}${esc(unit)}${suffix ? ` · ${esc(suffix)}` : ""}</div>`;
}

function renderMonitorOverview(snap) {
  const hw = snap.hardware || {};
  $("#monitor-ollama-pill").textContent = snap.ollama_reachable ? "Ollama: connected" : "Ollama: unreachable";
  $("#monitor-ollama-pill").classList.toggle("chip-bad", !snap.ollama_reachable);
  const rows = [
    ["VRAM used", hw.vram_used_mb != null ? `${fmtMb(hw.vram_used_mb)} / ${fmtMb(hw.vram_total_mb)}` : "-"],
    ["GPU utilization", hw.gpu_utilization_pct != null ? `${hw.gpu_utilization_pct}%` : "not reported by this GPU"],
    ["RAM used", hw.ram_used_mb != null ? `${fmtMb(hw.ram_used_mb)} / ${fmtMb(hw.ram_total_mb)}` : "-"],
    ["CPU utilization", hw.cpu_percent != null ? `${hw.cpu_percent}%` : "-"],
  ];
  $("#monitor-overview").innerHTML = rows
    .map(([label, value]) => `<div class="monitor-stat"><span class="muted small">${esc(label)}</span><strong>${esc(value)}</strong></div>`)
    .join("");

  const hwHistory = (snap.history && snap.history.hardware) || [];
  $("#monitor-chart-vram").innerHTML = svgSparkline(hwHistory.map((h) => h.vram_pct), { unit: "%" });
  $("#monitor-chart-gpu").innerHTML = svgSparkline(hwHistory.map((h) => h.gpu_utilization_pct), { unit: "%" });

  const modelsForTps = snap.models || [];
  const activeRates = modelsForTps.map((m) => m.recent_tokens_per_second).filter((v) => v != null);
  const avgTps = activeRates.length ? activeRates.reduce((a, b) => a + b, 0) / activeRates.length : null;
  state.monitor.tpsHistory.push(avgTps);
  if (state.monitor.tpsHistory.length > MONITOR_TPS_HISTORY_MAX) state.monitor.tpsHistory.shift();
  $("#monitor-chart-tps").innerHTML = svgSparkline(state.monitor.tpsHistory, { unit: " tok/s" });
}

function renderMonitorAlerts(alerts) {
  const box = $("#monitor-alerts");
  if (!alerts || !alerts.length) {
    box.classList.add("hidden");
    box.innerHTML = "";
    return;
  }
  box.classList.remove("hidden");
  box.innerHTML = alerts
    .map((a) => `<div class="monitor-alert monitor-alert-${esc(a.level)}">${a.level === "warning" ? "⚠" : "ℹ"} ${esc(a.text)}</div>`)
    .join("");
}

function renderMonitorModelCard(m) {
  const uptime = m.uptime_seconds != null ? fmtDuration(m.uptime_seconds) : "unknown";
  const vram = m.size_vram_mb != null ? fmtMb(m.size_vram_mb) : "-";
  const placement = m.placement ? `<span class="badge ${m.placement === "GPU" ? "on" : m.placement === "Split" ? "split" : "cpu"}">${esc(m.placement)}</span>` : "";
  const deviceNote = m.requested_device && m.placement && m.requested_device.toUpperCase() !== m.placement.toUpperCase()
    ? `<div class="muted small">Requested ${esc(m.requested_device)}, actually on ${esc(m.placement)}</div>`
    : "";
  return `<div class="fit-card model-card">
    <div class="running-top">
      <div class="model-identity">
        <span class="model-mark running" aria-hidden="true">M</span>
        <div>
          <div class="model-title">${esc(m.name)}</div>
          <div class="muted small">Running for ${esc(uptime)}</div>
        </div>
      </div>
      ${placement}
    </div>
    <div class="monitor-model-stats">
      <div><span class="muted small">VRAM</span><strong>${esc(vram)}</strong></div>
      <div><span class="muted small">tok/s (recent)</span><strong>${m.recent_tokens_per_second != null ? esc(m.recent_tokens_per_second) : "-"}</strong></div>
      <div><span class="muted small">Requests</span><strong>${esc(m.request_count)}${m.failure_count ? ` (${esc(m.failure_count)} failed)` : ""}</strong></div>
      <div><span class="muted small">TTFT (median)</span><strong>${m.median_ttft_ms != null ? `${esc(m.median_ttft_ms)} ms` : "-"}</strong></div>
    </div>
    ${deviceNote}
    <div class="row gap wrap fit-card-actions">
      <span class="spacer"></span>
      <button class="btn compact monitor-stop-btn" data-model="${esc(m.name)}">Stop</button>
    </div>
  </div>`;
}

function renderMonitorModels(models) {
  const body = $("#monitor-models");
  if (!models || !models.length) {
    body.innerHTML = `<div class="monitor-empty">
      <p class="muted">Nothing to monitor yet - this fills in once a model is running.</p>
      <p class="muted small">Head to <b>Setup &amp; Deploy</b>, pull or pick a model, and deploy it. Come back here to watch its VRAM, throughput, and request history live.</p>
      <button class="btn primary compact" id="monitor-goto-deploy">Go to Setup &amp; Deploy</button>
    </div>`;
    return;
  }
  body.innerHTML = models.map(renderMonitorModelCard).join("");
}

async function handleMonitorModelsClick(event) {
  const button = event.target.closest("button");
  if (!button) return;
  if (button.id === "monitor-goto-deploy") {
    onNavigate("serve");
    return;
  }
  if (!button.classList.contains("monitor-stop-btn")) return;
  busy(button, true);
  try {
    await postJSON("/models/stop", { model: button.dataset.model });
    await onModelStateInvalidated();
    toast(`Stopped '${button.dataset.model}'.`, "success");
  } catch (err) {
    toast(`Stop failed: ${err.message}`, "error");
  } finally {
    busy(button, false);
    void refreshMonitor();
  }
}

function renderMonitorRequests(requests) {
  const tbody = $("#monitor-requests-table tbody");
  const empty = $("#monitor-requests-empty");
  if (!requests || !requests.length) {
    tbody.innerHTML = "";
    $("#monitor-requests-table").classList.add("hidden");
    empty.classList.remove("hidden");
    return;
  }
  empty.classList.add("hidden");
  $("#monitor-requests-table").classList.remove("hidden");
  tbody.innerHTML = requests
    .map((r) => {
      const time = new Date(r.ts * 1000).toLocaleTimeString();
      const result = r.success ? `<span class="badge on">ok</span>` : `<span class="badge wont" title="${esc(r.error || "")}">error</span>`;
      return `<tr>
        <td>${esc(time)}</td>
        <td>${esc(r.model || "-")}</td>
        <td><span class="badge">${esc(r.source || "-")}</span></td>
        <td>${result}</td>
        <td class="num">${r.prompt_tokens ?? "-"}</td>
        <td class="num">${r.output_tokens ?? "-"}</td>
        <td class="num">${r.ttft_ms != null ? `${r.ttft_ms} ms` : "-"}</td>
        <td class="num">${r.tokens_per_second ?? "-"}</td>
        <td class="num">${r.elapsed_seconds}s</td>
      </tr>`;
    })
    .join("");
}

async function refreshMonitor() {
  try {
    const snap = await getJSON("/system/monitor");
    renderMonitorOverview(snap);
    renderMonitorAlerts(snap.alerts);
    renderMonitorModels(snap.models);
    renderMonitorRequests(snap.requests);
  } catch (err) {
    $("#monitor-overview").innerHTML = `<div class="muted">Monitor snapshot failed: ${esc(err.message)}</div>`;
  }
}

function startMonitorPolling() {
  if (state.monitor.timer) return;
  void refreshMonitor();
  state.monitor.timer = setInterval(() => void refreshMonitor(), MONITOR_POLL_MS);
}

function stopMonitorPolling() {
  if (state.monitor.timer) {
    clearInterval(state.monitor.timer);
    state.monitor.timer = null;
  }
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

async function checkHardware() {
  const btn = $("#btn-hardware");
  busy(btn, true);
  $("#hardware-body").innerHTML = skeletonHtml(3);
  try {
    const hw = await getJSON("/system/hardware");
    updateHwChip(hw);
    syncHardwareState(hw);
    const body = $("#hardware-body");
    if (!hw.gpu_available) {
      state.vramTotalMb = null;
      state.vramFreeMb = null;
      state.hardware = { ...hw, gpu: null, vram_total_mb: null, vram_free_mb: null };
      body.innerHTML = `<div class="muted" style="margin-bottom:.5rem">${esc(hw.message || "No GPU detected.")}</div>
        <div class="kv">${cpuRamRows(hw.system)}</div>`;
      updateVramBudgetUI();
      return;
    }
    const g = hw.gpus?.[0];
    if (!g) {
      body.innerHTML = `<div class="muted">GPU reported but no details available.</div>`;
      return;
    }
    state.hardware = {
      ...hw,
      gpu: g.name,
      vram_total_mb: hw.gpu_summary?.best_pool_total_mb ?? g.vram_total_mb,
      vram_free_mb: hw.gpu_summary?.best_pool_free_mb ?? g.vram_free_mb,
    };
    const note = hw.message
      ? `<div class="muted small" style="margin-bottom:.5rem">${esc(hw.message)}</div>`
      : "";
    const gpuRows = (hw.gpus || []).map((gpu, index) => {
      const memory = gpu.unified_memory
        ? "Unified memory"
        : `${fmtMb(gpu.vram_total_mb)} total · ${fmtMb(gpu.vram_free_mb)} free`;
      const source = gpu.vram_estimated ? "estimated" : (gpu.memory_source || gpu.backend || "detected");
      return `<span class="k">GPU ${index + 1}</span><span>${esc(gpu.name)} · ${esc(gpu.vendor || "unknown")} / ${esc(gpu.backend || "unknown")}</span>
        <span class="k">Memory</span><span>${memory} · ${esc(source)}</span>`;
    }).join("");
    const pool = hw.gpu_summary?.best_pool_total_mb
      ? `<span class="k">Fit pool</span><span>${fmtMb(hw.gpu_summary.best_pool_total_mb)} total · ${fmtMb(hw.gpu_summary.best_pool_free_mb)} free compatible memory</span>`
      : "";
    body.innerHTML = `${note}<div class="kv">
      ${gpuRows}
      ${pool}
      ${cpuRamRows(hw.system)}
    </div>`;
  } catch (err) {
    toast(`Hardware check failed: ${err.message}`, "error");
  } finally {
    busy(btn, false);
    updateProgressRail();
  }
}

function setOllamaPill(installed, reachable) {
  const pill = $("#ollama-status-pill");
  if (!pill) return;
  let cls = "chip";
  let text = "";
  if (!installed) {
    cls += " bad";
    text = "Ollama: not installed";
  } else if (!reachable) {
    cls += " warn";
    text = "Ollama: not running";
  } else {
    cls += " ok";
    text = "Ollama: running";
  }
  pill.className = cls;
  pill.textContent = text;
  pill.classList.remove("hidden");
}

// ---------------------------------------------------------------------------
// Update check (Release R7 Phase A) - best-effort, silent when offline/unavailable.
// ---------------------------------------------------------------------------

async function checkForUpdates() {
  const chip = $("#update-chip");
  if (!chip) return;
  try {
    const data = await getJSON("/system/update-check");
    if (!data.checked) {
      chip.classList.add("hidden");
      return;
    }
    if (data.update_available) {
      chip.textContent = `⬆ ${data.latest_version} available`;
      chip.title = `LocalDeploy ${data.latest_version} is available (you're on ${data.current_version}). Click to view.`;
      chip.classList.remove("hidden");
      chip.onclick = () => data.url && window.open(data.url, "_blank", "noopener");
    } else {
      chip.classList.add("hidden");
    }
  } catch {
    chip.classList.add("hidden");
  }
}

async function checkOllamaAvailability() {
  try {
    const res = await getJSON("/system/ollama-status");
    state.ollamaStatus = { installed: res.installed, reachable: res.reachable };
    setOllamaPill(res.installed, res.reachable);
  } catch {
    state.ollamaStatus = { installed: null, reachable: null };
    // Non-fatal: the pull flow will surface this again if it matters.
  }
}

// Segmented control inside "Get a model": show one panel at a time.

export function setMonitorActive(active) {
  if (active) startMonitorPolling();
  else stopMonitorPolling();
}

export function isMonitorActive() {
  return state.monitor.timer != null;
}

export function initSystem(options = {}) {
  if (initialized) return;
  initialized = true;
  onNavigate = options.onNavigate || onNavigate;
  onModelStateInvalidated = options.onModelStateInvalidated || onModelStateInvalidated;
  $("#monitor-models")?.addEventListener("click", (event) => void handleMonitorModelsClick(event));
  $("#btn-hardware")?.addEventListener("click", () => void refreshSystem());
  $("#vram-budget-gb")?.addEventListener("input", () => {
    const raw = $("#vram-budget-gb").value.trim();
    state.vramBudgetMb = raw ? targetVram() : null;
    updateVramBudgetUI();
    notifySystemChanges();
  });
  $$(".vram-preset").forEach((button) => {
    button.addEventListener("click", () => {
      if (button.dataset.vramPreset === "detected") setVramBudgetMb(state.vramTotalMb);
      else if (button.dataset.vramPreset === "free") setVramBudgetMb(state.vramFreeMb);
      else if (button.dataset.vramGb) setVramBudgetMb(Number(button.dataset.vramGb) * 1024);
    });
  });
}

export async function refreshSystem() {
  const results = await Promise.allSettled([checkHardware(), checkOllamaAvailability(), checkForUpdates()]);
  notifySystemChanges();
  return { snapshot: getSystemSnapshot(), results };
}

// A lighter check for frequent triggers (every tab switch fires this via
// invalidateModelState): Ollama's reachability can change mid-session, but
// hardware doesn't, and an update check is an internet call that has no
// business re-running on every navigation. Re-running all three here meant
// every tab switch fired 3 extra requests on top of refreshModels()'s own 3 -
// and since navigation doesn't await or cancel the previous switch's refresh,
// quickly clicking through tabs could stack up several overlapping bursts.

export async function refreshOllamaStatus() {
  const results = await Promise.allSettled([checkOllamaAvailability()]);
  notifySystemChanges();
  return { snapshot: getSystemSnapshot(), results };
}
