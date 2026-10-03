"use strict";

import { $, $$, busy, currentTheme, downloadFile, esc, fmtBytes, fmtDuration, fmtMb, formatExpires, getJSON, postJSON, postMaybeStream, simpleModal, skeletonHtml, startElapsed, toast, vramBarHtml } from "./shared.js?v=20260719-ui32";

const state = {
  profiles: [], profileData: {}, profileModels: {}, defaultProfile: null,
  installedByName: {}, installedLoaded: false, installedSort: "default",
  installedSelection: new Set(), installedList: [], servedModels: [], runningDetails: [],
  runningPlacements: {}, fitRefreshTimer: null, fitCache: {},
  catalog: { rows: [], providers: [], loaded: false, page: 0 },
  remoteCatalogLoaded: false, unifiedSearchSeq: 0, unifiedSearchTimer: null,
  remoteCatalog: { sourceRows: [], rows: [], fits: {}, query: "", sort: "popularity-desc" },
  unloadingModels: new Set(), pullRetry: null, manifestToRecreate: null,
  bakeoff: { controller: null },
  systemContext: { vramTotalMb: null, vramFreeMb: null, vramBudgetMb: null, ollamaStatus: {} },
};
const listeners = new Set();
let initialized = false;

function clone(value) { return value == null ? value : structuredClone(value); }
function readonly(value) {
  if (value && typeof value === "object" && !Object.isFrozen(value)) {
    Object.values(value).forEach(readonly);
    Object.freeze(value);
  }
  return value;
}
function targetVram() { return state.systemContext.vramBudgetMb ?? state.systemContext.vramTotalMb ?? state.systemContext.vramFreeMb; }
export function fitCacheKey(model, budgetMb) { return `${model}|${budgetMb ?? "auto"}`; }
function normalizedProfiles() {
  return state.profiles.map((name) => {
    const raw = clone(state.profileData[name] || {});
    const modelId = state.profileModels[name] || name;
    const availability = installedStatusForProfile(name);
    const running = runningDetailForInstalled(modelId);
    return {
      ...raw,
      name,
      model_id: modelId,
      backend: String(raw.backend || "ollama").toLowerCase(),
      enabled: raw.enabled === true,
      supports_vision: raw.supports_vision === true,
      is_default: name === state.defaultProfile,
      is_loaded: !!running,
      running_model: running?.name || null,
      placement: running?.placement || null,
      availability: { label: availability.label, className: availability.cls },
    };
  });
}

function normalizedInventory() {
  return state.installedList.map((item) => {
    const running = runningDetailForInstalled(item.name);
    return { ...clone(item), is_loaded: !!running, running: clone(running) };
  });
}

export function getModelSnapshot() {
  return readonly({
    profiles: normalizedProfiles(), defaultProfile: state.defaultProfile,
    installed: normalizedInventory(), installedLoaded: state.installedLoaded,
    running: clone(state.runningDetails), servedModels: [...state.servedModels],
    placements: clone(state.runningPlacements),
  });
}

function notifyModelChanges(reason = "state", requiresSystemRefresh = false) {
  const snapshot = getModelSnapshot();
  const meta = { reason, requiresSystemRefresh };
  listeners.forEach((listener) => listener(snapshot, meta));
}

export function subscribeToModelChanges(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function updateSystemContext(snapshot = {}) {
  const previousBudget = targetVram();
  state.systemContext = {
    vramTotalMb: snapshot.vramTotalMb ?? null,
    vramFreeMb: snapshot.vramFreeMb ?? null,
    vramBudgetMb: snapshot.vramBudgetMb ?? null,
    ollamaStatus: { ...(snapshot.ollamaStatus || {}) },
  };
  if (state.systemContext.ollamaStatus.installed === false) {
    showOllamaHelp(null, null, "Ollama isn't installed, so pulling and serving models won't work yet. Install it, then come back here.");
  }
  if (previousBudget !== targetVram()) scheduleFitRefresh();
}

function scheduleFitRefresh() {
  clearTimeout(state.fitRefreshTimer);
  state.fitRefreshTimer = setTimeout(() => {
    $$(".mrow[data-model]", $("#installed-body")).forEach((row) => fitCheckRow(row));
    if (!$("#fit-finder-body")?.textContent.includes("not been scanned")) scanConfiguredFits();
    if (state.remoteCatalog.rows.length) {
      void loadRemoteCatalogFits().then(() => renderRemoteCatalog());
    }
  }, 350);
}

// Render the CPU + RAM block shared by the GPU and CPU-only hardware views.

async function refreshLiveModelState(includeInstalled = false, options = {}) {
  const jobs = [refreshStatus({ notify: false })];
  if (includeInstalled) jobs.push(refreshInstalled({ notify: false }));
  const results = await Promise.allSettled(jobs);
  if (options.notify !== false) notifyModelChanges("lifecycle", true);
  return results;
}

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------

async function loadProfiles(options = {}) {
  try {
    const data = await getJSON("/profiles");
    const profiles = data.profiles || {};
    state.profileData = profiles;
    state.profiles = Object.keys(profiles);
    state.profileModels = {};
    state.profiles.forEach((name) => (state.profileModels[name] = profiles[name]?.model_id || name));
    state.defaultProfile = data.default_profile || state.profiles[0] || null;
    renderProfileSelectOptions();
    setProfileActionsEnabled(state.profiles.length > 0);
    setConn(true);
    if (options.notify !== false) notifyModelChanges("profiles");
  } catch (err) {
    setConn(false);
    setProfileActionsEnabled(false);
    toast(`Could not load profiles: ${err.message}`, "error");
  }
}

// (Re)build profile <select>s, annotating profiles whose model isn't
// pulled so a dropdown never silently offers a model that can't deploy.
// Called again after the installed list loads, preserving the selection.

function renderProfileSelectOptions() {
  const build = (current) =>
    state.profiles
      .map((name) => {
        const p = state.profileData[name] || {};
        let label = p.model_id ? `${name} - ${p.model_id}` : name;
        if (profileIsUnpulled(name)) {
          label += ` (${installedStatusForProfile(name).label})`;
        }
        const selectedName = current && state.profiles.includes(current) ? current : state.defaultProfile;
        const sel = name === selectedName ? " selected" : "";
        return `<option value="${esc(name)}"${sel}>${esc(label)}</option>`;
      })
      .join("");
  ["#profile-select", "#bench-profile-select"].forEach((sel) => {
    const el = $(sel);
    if (el) el.innerHTML = build(el.value);
  });
}

// Guard actions that need a profile so they can't fire a blank profile name.

function setProfileActionsEnabled(enabled) {
  ["#btn-serve", "#btn-switch", "#btn-run"].forEach((sel) => {
    const el = $(sel);
    if (el) el.disabled = !enabled;
  });
}

function setConn(ok) {
  const pill = $("#conn-pill");
  pill.textContent = ok ? "API: connected" : "API: unreachable";
  pill.className = `conn ${ok ? "ok" : "bad"}`;
}

function installedStatusForProfile(profileName) {
  const p = state.profileData[profileName] || {};
  if ((p.backend || "ollama") === "llamacpp") {
    // GGUF file profiles: presence is reported by the server (model_file_exists).
    if (p.model_file_exists === false) return { label: "file missing", cls: "off" };
    return { label: "gguf on disk", cls: "on" };
  }
  const model = state.profileModels[profileName] || profileName;
  if (state.installedByName[model]) return { label: "pulled", cls: "on" };
  const base = model.split(":")[0];
  const hit = Object.keys(state.installedByName).some((name) => name === model || name.split(":")[0] === base);
  return hit ? { label: "variant pulled", cls: "tight" } : { label: "not pulled", cls: "off" };
}

// A profile is hidden by default when its model isn't actually on the machine:
// an Ollama profile whose model isn't pulled, or a llama.cpp profile whose
// GGUF file the server reports missing.

function profileIsUnpulled(name) {
  const p = state.profileData[name] || {};
  if ((p.backend || "ollama") === "llamacpp") return p.model_file_exists === false;
  if (!state.installedLoaded) return false;
  return installedStatusForProfile(name).label === "not pulled";
}

const QUANT_EXPLANATIONS = {
  F16: "16-bit floating-point weights. Highest memory use and a reference-quality baseline.",
  BF16: "16-bit brain floating-point weights. Reference quality with the same broad memory class as F16.",
  Q8_0: "8-bit quantization. Near-reference quality, but much larger than common 4-bit or 5-bit files.",
  Q6_K: "About 6-bit K-quant. High quality when there is enough RAM or VRAM headroom.",
  Q5_K_M: "5-bit K-quant, medium variant. More memory and usually more quality than Q4_K_M.",
  Q4_K_M: "4-bit K-quant, medium variant. A common balance of model quality, size, and speed.",
  Q4_0: "Older 4-bit quantization. Small and widely supported, but usually less accurate than Q4_K_M.",
  Q4_0_QAT: "4-bit quantization-aware-trained weights. The model was trained to preserve quality at this precision.",
  IQ4_XS: "Importance-aware 4-bit, extra-small variant. Saves memory to fit larger models or context windows.",
  Q3_K_M: "3-bit K-quant, medium variant. Very compact, with a more noticeable quality tradeoff.",
};

function quantExplanation(value) {
  const quant = String(value || "").trim().toUpperCase();
  if (!quant) return "Quantization reduces model weight precision to use less disk space and memory.";
  if (QUANT_EXPLANATIONS[quant]) return QUANT_EXPLANATIONS[quant];
  const bits = quant.match(/(?:^|_)[QI]?(\d+)(?:_|$)/)?.[1];
  const family = quant.includes("_K") ? " It uses the newer K-quant family." : "";
  const variant = quant.endsWith("_M") ? " M means the medium mixed-precision variant." : "";
  return `${bits ? `Approximately ${bits}-bit quantized weights.` : "Quantized model weights."}${family}${variant}`;
}

function quantLabelHtml(value, className = "badge") {
  if (!value) return `<span class="muted">-</span>`;
  const explanation = quantExplanation(value);
  return `<span class="${esc(className)} quant-label tooltip-target" tabindex="0" role="button" aria-expanded="false" data-tooltip="${esc(explanation)}" aria-label="${esc(value)}: ${esc(explanation)}">${esc(value)}<span class="quant-info" aria-hidden="true">?</span></span>`;
}

function ollamaModelNamesMatch(left, right) {
  const a = String(left || "").toLowerCase();
  const b = String(right || "").toLowerCase();
  if (!a || !b) return false;
  if (a === b) return true;
  if (!a.includes(":") && b === `${a}:latest`) return true;
  if (!b.includes(":") && a === `${b}:latest`) return true;
  return false;
}

function runningDetailForInstalled(modelName) {
  return (
    (state.runningDetails || []).find((item) => ollamaModelNamesMatch(item.name, modelName)) ||
    ((state.servedModels || []).some((name) => ollamaModelNamesMatch(name, modelName)) ? { name: modelName } : null)
  );
}

// ---------------------------------------------------------------------------
// Tab 1 - Status
// ---------------------------------------------------------------------------

async function refreshStatus(options = {}) {
  const btn = $("#btn-status");
  busy(btn, true);
  $("#status-body").innerHTML = skeletonHtml(2);
  try {
    const s = await getJSON("/system/status");
    state.servedModels = s.served_models || [];
    state.runningDetails = s.ollama?.running || [];
    // Record each running model's *measured* placement so a benchmark can tag
    // its report card with the device the model actually runs on (not a guess).
    state.runningPlacements = {};
    state.runningDetails.forEach((m) => {
      if (m.name) state.runningPlacements[m.name] = { placement: m.placement, gpu_percent: m.gpu_percent };
    });
    const body = $("#status-body");
    const reachable = s.ollama?.reachable
      ? `<span class="badge on">Ollama online</span>`
      : `<span class="badge off">Ollama offline</span>`;
    let served;
    if (state.servedModels.length) {
      served = s.ollama.running
        .map((m) => {
          const place =
            m.placement === "Split"
              ? `<span class="badge split">${esc(m.gpu_percent)}% GPU</span>`
              : m.placement
                ? `<span class="badge ${m.placement === "GPU" ? "fits" : "cpu"}">${esc(m.placement)}</span>`
                : "";
          const total = state.systemContext.vramTotalMb;
          const vramMb = m.size_vram_mb ?? (m.size_vram ? Math.round(m.size_vram / 1_000_000) : null);
          const diskMb = m.size_mb ?? (m.size ? Math.round(m.size / 1_000_000) : null);
          const usedLabel = total && vramMb ? `${fmtMb(vramMb)} / ${fmtMb(total)} GPU VRAM` : `VRAM ${fmtMb(vramMb)}`;
          const activity = m.activity === "loaded" ? "Loaded / warm" : esc(m.activity || "Loaded");
          return `<div class="running-card model-card" data-model="${esc(m.name)}">
            <div class="running-top">
              <div class="model-identity">
                <span class="model-mark running" aria-hidden="true">R</span>
                <div>
                  <div class="model-title">${esc(m.name)}</div>
                  <div class="muted small">${esc(activity)} · ${esc(formatExpires(m.expires_at))}</div>
                </div>
              </div>
              <div class="row gap">
                ${place}
                <button class="btn danger compact kill-model-btn" title="Unload this model from memory/VRAM">Unload</button>
              </div>
            </div>
            ${vramBarHtml(vramMb, total, usedLabel)}
            <div class="model-meta-grid">
              <span>⚡ VRAM in use</span><b>${esc(usedLabel)}</b>
              <span>💾 Size on disk</span><b>${fmtMb(diskMb)}</b>
              <span>GPU residency</span><b>${m.gpu_percent != null ? `${esc(m.gpu_percent)}%` : "?"}</b>
            </div>
            ${apiSnippetHtml(m.name)}
            <div class="row gap wrap" style="margin-top:0.4rem">
              <button class="btn compact export-manifest-btn" data-model="${esc(m.name)}">Export deployment</button>
              <button class="btn compact use-elsewhere-btn" data-model="${esc(m.name)}">Use elsewhere ↗</button>
            </div>
            <div class="muted small">${esc(m.activity_note || "Ollama keeps this model warm until the keep-alive expires.")}</div>
          </div>`;
        })
        .join("");
    } else {
      served = `<div class="muted">No model is currently loaded. Deploy a profile below; unload controls appear here after Ollama lists a loaded model.</div>`;
    }
    body.innerHTML = `<div style="margin-bottom:.5rem">${reachable}</div>${served}`;
  } catch (err) {
    state.servedModels = [];
    state.runningDetails = [];
    state.runningPlacements = {};
    toast(`Status failed: ${err.message}`, "error");
  } finally {
    busy(btn, false);
    if (state.installedLoaded) renderInstalledList();
    if (options.notify !== false) notifyModelChanges("runtime");
  }
}

// The full local API address for a served model, with one-click copy: the
// endpoint URL for OpenAI-compatible clients, and a ready-to-run curl.

function apiEndpointBase() {
  return `${window.location.origin}/v1/chat/completions`;
}

function apiDocsUrl() {
  return `${window.location.origin}/docs`;
}

function curlSnippetFor(model) {
  return [
    `curl ${apiEndpointBase()} \\`,
    `  -H "Content-Type: application/json" \\`,
    `  -d '{"model": "${model}", "messages": [{"role": "user", "content": "Hello!"}]}'`,
  ].join("\n");
}

function apiSnippetHtml(model) {
  const url = apiEndpointBase();
  const docs = apiDocsUrl();
  return `<div class="api-snippet">
    <span class="eyebrow">🔌 Use via API</span>
    <div class="api-snippet-row">
      <code class="api-url" title="OpenAI-compatible endpoint - point any client here with model: &quot;${esc(model)}&quot; and any API key">${esc(url)}</code>
      <button class="btn compact copy-btn" data-copy="${esc(url)}" title="Copy endpoint URL">⧉ URL</button>
      <button class="btn compact copy-btn" data-copy="${esc(curlSnippetFor(model))}" title="Copy a ready-to-run curl request for this model">⧉ curl</button>
      <a class="btn compact api-docs-link" href="${esc(docs)}" target="_blank" rel="noopener" title="Open Swagger UI with every endpoint and its request and response definitions">API docs ↗</a>
    </div>
  </div>`;
}

async function copyFromButton(button) {
  try {
    await navigator.clipboard.writeText(button.dataset.copy);
    const was = button.textContent;
    button.textContent = "✓ Copied";
    setTimeout(() => (button.textContent = was), 1400);
  } catch (err) {
    toast(`Copy failed: ${err.message}`, "error");
  }
}

function handleStatusBodyClick(event) {
  const button = event.target.closest("button");
  if (!button) return;
  if (button.classList.contains("kill-model-btn")) {
    void killRunningModel(button.closest(".running-card")?.dataset.model, button);
  } else if (button.classList.contains("export-manifest-btn")) {
    void exportManifest(button.dataset.model, button);
  } else if (button.classList.contains("use-elsewhere-btn")) {
    void openIntegrationSnippets(button.dataset.model, button);
  } else if (button.classList.contains("copy-btn")) {
    void copyFromButton(button);
  }
}

function waitMs(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitForModelToUnload(name, attempts = 12) {
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    if (attempt) await waitMs(400);
    const status = await getJSON("/system/status");
    const running = status.ollama?.running || [];
    state.runningDetails = running;
    state.servedModels = status.served_models || running.map((item) => item.name).filter(Boolean);
    if (!running.some((item) => ollamaModelNamesMatch(item.name, name))) return true;
  }
  return false;
}

async function killRunningModel(name, btn) {
  if (!name) return;
  if (state.unloadingModels.has(name)) return;
  state.unloadingModels.add(name);
  busy(btn, true);
  const originalText = btn?.textContent;
  if (btn) btn.textContent = "Unloading…";
  try {
    const res = await postJSON("/models/stop", { model: name });
    if (!res.success) throw new Error(res.error || res.message || "Could not unload model.");
    const confirmed = res.confirmed === true || res.status === "unloaded" || await waitForModelToUnload(name);
    if (!confirmed) throw new Error(`Ollama still reports ${name} as loaded. Retry unload or refresh status.`);
    toast(`Unloaded ${name}.`, "success");
    await refreshLiveModelState(true);
  } catch (err) {
    toast(`Unload failed: ${err.message}`, "error");
  } finally {
    state.unloadingModels.delete(name);
    busy(btn, false);
    if (btn && btn.isConnected && originalText) btn.textContent = originalText;
    if (state.installedLoaded) renderInstalledList();
  }
}

// Map a profile's running model to the device it's actually placed on, by
// matching the profile's model_id against the live placement map. Returns
// "gpu" | "cpu" | "split" | null (null = not loaded / placement unknown).

function detectDevice(profileName) {
  const modelId = String(state.profileModels[profileName] || profileName);
  const base = modelId.split(":")[0];
  for (const [name, info] of Object.entries(state.runningPlacements || {})) {
    if (name === modelId || name.split(":")[0] === base) {
      return info.placement ? info.placement.toLowerCase() : null;
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Tab 1 - Deploy / unload / replace
// ---------------------------------------------------------------------------

function showServeResult(res) {
  const node = $("#serve-result");
  const ok = res && res.success;
  node.className = `result ${ok ? "ok" : "err"}`;
  node.textContent = (res && (res.message || res.error)) || (ok ? "Done." : "Failed.");
}

// Label the warm-up wait so loading a large model (slow on CPU) shows a live
// elapsed counter instead of an apparently-frozen button.

function warmupLabel(device) {
  return device === "cpu"
    ? "Loading on CPU (large models can take a minute)"
    : "Loading model into memory";
}

async function serveModel() {
  const btn = $("#btn-serve");
  const device = $("#serve-device").value;
  busy(btn, true);
  const node = $("#serve-result");
  node.className = "result";
  const stop = startElapsed(node, warmupLabel(device));
  try {
    const res = await postJSON("/models/serve", {
      profile: $("#profile-select").value,
      keep_alive: $("#keep-alive").value.trim() || "60m",
      device,
    });
    stop();
    showServeResult(res);
    await refreshLiveModelState(true);
  } catch (err) {
    stop();
    node.className = "result err";
    node.textContent = `Deploy failed: ${err.message}`;
    toast(`Deploy failed: ${err.message}`, "error");
  } finally {
    busy(btn, false);
  }
}

async function switchModel() {
  const btn = $("#btn-switch");
  const device = $("#serve-device").value;
  busy(btn, true);
  const node = $("#serve-result");
  node.className = "result";
  const stop = startElapsed(node, warmupLabel(device));
  try {
    const res = await postJSON("/models/switch", {
      to_profile: $("#profile-select").value,
      from_model: state.servedModels[0] || null,
      keep_alive: $("#keep-alive").value.trim() || "60m",
      device,
    });
    stop();
    showServeResult(res);
    await refreshLiveModelState(true);
  } catch (err) {
    stop();
    node.className = "result err";
    node.textContent = `Deploy and replace failed: ${err.message}`;
    toast(`Deploy and replace failed: ${err.message}`, "error");
  } finally {
    busy(btn, false);
  }
}

// ---------------------------------------------------------------------------
// Tab 1 - Installed models + fit check
// ---------------------------------------------------------------------------
// Tiered fit badge (Phase 3): green = comfortable, yellow = tight / CPU-only
// (soft warnings), red = won't fit anywhere (hard). Falls back to the coarse
// verdict for older responses without a severity.

function fitBadge(res) {
  const sevCls = { ok: "fits", soft: "tight", hard: "wont", unknown: "unknown" };
  const cls = sevCls[res.severity] || (res.verdict === "FITS" ? "fits" : res.verdict === "WONT_FIT" ? "wont" : "unknown");
  const label = res.tier ? res.tier.replace(/_/g, " ") : res.verdict || "?";
  return `<span class="badge ${cls}" title="${esc(res.headline || "")}">${esc(label)}</span>`;
}

function parseParamSize(value) {
  const m = String(value || "").match(/(\d+(?:\.\d+)?)\s*B/i);
  return m ? Number(m[1]) : null;
}

function fitRequestForModel(model, details = {}, sizeBytes = null) {
  return {
    model_id: model,
    params_b: parseParamSize(details.parameter_size),
    quant: details.quantization_level || null,
    context: details.context_length ? Math.min(Number(details.context_length), 8192) : null,
    size_bytes: sizeBytes || null,
    free_vram_mb: targetVram(),
  };
}

function fitMeterHtml(res) {
  const req = res?.estimate_gb?.required;
  const budget = targetVram();
  if (req == null || !budget) return "";
  const budgetGb = budget / 1024;
  const width = Math.max(0, Math.min(130, (req / budgetGb) * 100));
  const cls = res.severity === "hard" ? "danger" : res.severity === "soft" ? "warn" : "ok";
  return `<div class="mini-meter" title="Estimated ${req} GB against ${budgetGb.toFixed(1)} GB budget">
    <div class="mini-meter-fill ${cls}" style="width:${Math.min(width, 100).toFixed(1)}%"></div>
  </div>`;
}

async function refreshInstalled(options = {}) {
  const btn = $("#btn-installed");
  busy(btn, true);
  const body = $("#installed-body");
  body.innerHTML = skeletonHtml(3);
  try {
    const data = await getJSON("/registry/installed");
    // A real refresh invalidates cached fit results (models may have been
    // pulled/deleted, or free VRAM shifted); re-sorting alone keeps the cache.
    state.fitCache = {};
    if (!data.success) {
      state.installedByName = {};
      state.installedList = [];
      state.installedLoaded = false; // unknown, so nothing gets hidden as "not pulled"
      renderProfileSelectOptions();
      body.innerHTML = `<div class="muted">${esc(data.error || "Ollama unreachable.")}</div>`;
      return;
    }
    state.installedByName = {};
    state.installedLoaded = true;
    state.installedList = data.installed;
    // Drop selections for models that no longer exist.
    const names = new Set(data.installed.map((m) => m.name));
    state.installedSelection = new Set([...state.installedSelection].filter((n) => names.has(n)));
    if (!data.installed.length) {
      renderProfileSelectOptions();
      updateDiskSummary();
      body.innerHTML = `<div class="muted">No models pulled yet - grab one from <b>Get a model</b> above.</div>`;
      return;
    }
    data.installed.forEach((m) => {
      if (m.name) state.installedByName[m.name] = m;
    });
    renderProfileSelectOptions();
    renderInstalledList();
  } catch (err) {
    state.installedByName = {};
    state.installedList = [];
    state.installedLoaded = false;
    toast(`Installed list failed: ${err.message}`, "error");
  } finally {
    busy(btn, false);
    if (state.remoteCatalog.rows.length) renderRemoteCatalog();
    if (options.notify !== false) notifyModelChanges("installed");
  }
}

function sortedInstalled() {
  const list = state.installedList.slice();
  const sort = state.installedSort;
  if (sort === "size") list.sort((a, b) => (b.size || 0) - (a.size || 0));
  else if (sort === "recent") list.sort((a, b) => String(b.modified_at || "").localeCompare(String(a.modified_at || "")));
  else if (sort === "name") list.sort((a, b) => String(a.name).localeCompare(String(b.name)));
  return list;
}

function updateDiskSummary() {
  const slot = $("#models-disk-summary");
  if (!slot) return;
  const list = state.installedList;
  if (!list.length) {
    slot.textContent = "";
    return;
  }
  const totalGb = list.reduce((sum, m) => sum + (m.size || 0), 0) / 1e9;
  slot.textContent = `${list.length} model${list.length === 1 ? "" : "s"} · ${totalGb.toFixed(1)} GB on disk`;
}

function updateBulkDeleteBar() {
  const bar = $("#bulk-delete-bar");
  if (!bar) return;
  const selected = [...state.installedSelection];
  if (!selected.length) {
    bar.classList.add("hidden");
    return;
  }
  const bytes = selected.reduce((sum, n) => sum + (state.installedByName[n]?.size || 0), 0);
  $("#bulk-delete-label").textContent =
    `${selected.length} model${selected.length === 1 ? "" : "s"} selected · ${(bytes / 1e9).toFixed(1)} GB`;
  bar.classList.remove("hidden");
}

function renderInstalledList() {
  const body = $("#installed-body");
  if (!body) return;
  updateDiskSummary();
  body.innerHTML =
    `<div class="mlist">` +
    sortedInstalled()
      .map((m) => {
        const diskGb = m.size ? (m.size / 1e9).toFixed(1) : null;
        const d = m.details || {};
        const quant = d.quantization_level ? quantLabelHtml(d.quantization_level, "badge model-quant") : "";
        const params = d.parameter_size ? `<span class="meta">${esc(d.parameter_size)}</span>` : "";
        const date = m.modified_at ? `<span class="meta">updated ${esc(m.modified_at.slice(0, 10))}</span>` : "";
        const disk = diskGb
          ? `<span class="meta disk-chip" title="Space this model takes on your drive (not the memory it needs to run)">💾 ${esc(diskGb)} GB disk</span>`
          : "";
        const running = runningDetailForInstalled(m.name);
        const unloading = [...state.unloadingModels].some((name) => ollamaModelNamesMatch(name, m.name));
        const loaded = running ? `<span class="badge on">● loaded</span>` : "";
        const loadedHint = running ? `<span class="meta">${esc(formatExpires(running.expires_at))}</span>` : "";
        const primaryAction = running || unloading
          ? `<button class="btn danger unload-installed-btn${unloading ? " loading" : ""}" title="Unload this model from RAM and VRAM"${unloading ? " disabled" : ""}>${unloading ? "Unloading…" : "■ Unload"}</button>`
          : `<button class="btn primary start-installed-btn" title="Load this model into memory and start serving it">▶ Deploy</button>`;
        const checked = state.installedSelection.has(m.name) ? " checked" : "";
        return `<div class="mrow model-row" data-model="${esc(m.name)}">
          <input type="checkbox" class="model-select" title="Select for bulk delete"${checked} />
          <div class="model-row-main">
            <span class="name">${esc(m.name)}</span>
            <span class="model-row-meta">${params}${quant}${disk}${date}${loaded}${loadedHint}</span>
            <span class="fit"><span class="muted small">⚡ checking VRAM fit…</span></span>
          </div>
          <span class="spacer"></span>
          ${primaryAction}
          <button class="btn edit-tuning-btn" title="Edit this model's run profile (context, KV cache, GPU layers…)">⚙ Tune</button>
          <button class="btn compact fit-btn" title="Re-run the VRAM fit estimate (runs automatically when the list loads)">↻</button>
          <button class="btn danger del-btn" title="Delete from disk${diskGb ? ` - frees ${esc(diskGb)} GB` : ""}">🗑 Delete</button>
        </div>`;
      })
      .join("") +
    `</div>`;
  updateBulkDeleteBar();
  // Auto-run the fit check for each row so warnings appear without a click.
  $$(".mrow", body).forEach((row) => fitCheckRow(row));
}

async function bulkDeleteSelected(btn) {
  const selected = [...state.installedSelection];
  if (!selected.length) return;
  const bytes = selected.reduce((sum, n) => sum + (state.installedByName[n]?.size || 0), 0);
  const preview = selected.slice(0, 12).join("\n  ");
  const more = selected.length > 12 ? `\n  …and ${selected.length - 12} more` : "";
  if (!window.confirm(`Delete ${selected.length} model(s) from disk, freeing ~${(bytes / 1e9).toFixed(1)} GB?\n\n  ${preview}${more}\n\nThis cannot be undone (models can be pulled again).`)) return;
  busy(btn, true);
  let deleted = 0;
  try {
    for (const model of selected) {
      const res = await postJSON("/models/delete", { model });
      if (res.success) {
        deleted += 1;
        state.installedSelection.delete(model);
      } else {
        toast(res.error || res.message || `Could not delete ${model}.`, "error");
      }
    }
    toast(`Deleted ${deleted} model(s).`, "success");
    await refreshLiveModelState(true);
  } catch (err) {
    toast(`Bulk delete failed: ${err.message}`, "error");
  } finally {
    busy(btn, false);
  }
}

async function fitCheckRow(row, force = false) {
  const model = row.dataset.model;
  const installed = state.installedByName[model] || {};
  const details = installed.details || {};
  const slot = $(".fit", row);
  const btn = $(".fit-btn", row);
  const render = (res) => {
    if (res.verdict) {
      const req = res.estimate_gb?.required;
      const free = res.free_vram_gb;
      const detail = req != null
        ? `⚡ needs ~${req} GB VRAM${free != null ? ` of ${free} GB budget` : ""}`
        : "";
      slot.innerHTML = `<div class="fit-summary">${fitBadge(res)}<span class="meta" title="Estimated memory to run this model (weights + KV cache + overhead) - different from its size on disk">${esc(detail)}</span></div>${fitMeterHtml(res)}`;
    } else {
      slot.innerHTML = `<span class="muted">${esc(res.message || "n/a")}</span>`;
    }
  };
  // Re-sorting or re-selecting re-renders the whole list; without a cache that
  // refires one fit request per model every time. Budget changes make a new key.
  const cacheKey = fitCacheKey(model, targetVram());
  if (!force && state.fitCache[cacheKey]) {
    render(state.fitCache[cacheKey]);
    return;
  }
  busy(btn, true);
  slot.textContent = "…";
  try {
    const res = await postJSON("/system/fit-check", fitRequestForModel(model, details, installed.size));
    state.fitCache[cacheKey] = res;
    render(res);
  } catch (err) {
    slot.innerHTML = `<span class="muted">error</span>`;
    toast(`Fit check failed: ${err.message}`, "error");
  } finally {
    busy(btn, false);
  }
}

async function startInstalledModel(name, btn) {
  if (!name) return;
  busy(btn, true);
  const node = $("#serve-result");
  node.className = "result";
  const stop = startElapsed(node, "Loading model into memory");
  try {
    const res = await postJSON("/models/serve", {
      model: name,
      keep_alive: $("#keep-alive").value.trim() || "60m",
      device: $("#serve-device").value,
    });
    stop();
    showServeResult(res);
    await refreshLiveModelState(true);
  } catch (err) {
    stop();
    toast(`Deploy failed: ${err.message}`, "error");
  } finally {
    busy(btn, false);
  }
}

function findInstalledModel(modelId) {
  if (!modelId) return null;
  if (state.installedByName[modelId]) return state.installedByName[modelId];
  // Match case-insensitively and tolerate the differences between what the user
  // asked to pull and how Ollama stores it: an implicit ":latest" tag, and
  // "hf.co/Org/Repo-GGUF" landing as a lowercased name. Fall back to comparing
  // the last path segment so an HF GGUF pull still resolves to its installed row.
  const want = String(modelId).toLowerCase();
  const wantNoTag = want.split(":")[0];
  const tail = (s) => s.split("/").pop();
  const names = Object.values(state.installedByName);
  const norm = (m) => String(m.name || "").toLowerCase();
  return (
    names.find((m) => norm(m) === want) ||
    names.find((m) => norm(m) === `${want}:latest`) ||
    names.find((m) => norm(m).split(":")[0] === wantNoTag) ||
    names.find((m) => tail(norm(m).split(":")[0]) === tail(wantNoTag)) ||
    null
  );
}

// The actual installed key for a model the user asked to pull (for scroll/flip
// targeting), or the original string if it isn't installed (yet).

function resolveInstalledName(modelId) {
  return findInstalledModel(modelId)?.name || modelId;
}

function fitMatchesFilter(res, filter) {
  if (filter === "gpu") return res?.success && res?.verdict === "FITS";
  if (filter === "runnable") return res?.success && res?.severity !== "hard";
  return true;
}

function severityRank(sev) {
  return { ok: 0, soft: 1, unknown: 2, hard: 3 }[sev] ?? 4;
}

async function scanConfiguredFits() {
  const btn = $("#btn-fit-profiles");
  const body = $("#fit-finder-body");
  const filter = $("#fit-filter").value;
  busy(btn, true);
  body.innerHTML = `<span class="spin-inline"></span> Scanning saved run profiles against ${esc(fmtMb(targetVram()))}...`;
  try {
    if (!Object.keys(state.installedByName).length) {
      await refreshInstalled();
    }
    const rows = await Promise.all(
      state.profiles.map(async (profile) => {
        const p = state.profileData[profile] || {};
        const modelId = p.model_id || profile;
        const installed = findInstalledModel(modelId);
        const fit = await postJSON("/system/fit-check", {
          profile,
          size_bytes: installed?.size || null,
          free_vram_mb: targetVram(),
        });
        return { profile, p, modelId, installed, fit };
      })
    );
    const filtered = rows
      .filter((r) => fitMatchesFilter(r.fit, filter))
      .sort((a, b) => {
        const s = severityRank(a.fit?.severity) - severityRank(b.fit?.severity);
        if (s !== 0) return s;
        return (a.fit?.estimate_gb?.required ?? 999) - (b.fit?.estimate_gb?.required ?? 999);
      });
    if (!filtered.length) {
      body.innerHTML = `<div class="muted">No saved run profiles match this filter at ${esc(fmtMb(targetVram()))}.</div>`;
      return;
    }
    body.innerHTML = `<div class="fit-grid">` + filtered.map(renderProfileFitCard).join("") + `</div>`;
  } catch (err) {
    body.innerHTML = `<div class="muted">Scan failed.</div>`;
    toast(`Fit scan failed: ${err.message}`, "error");
  } finally {
    busy(btn, false);
  }
}

function handleFitFinderClick(event) {
  const button = event.target.closest("button");
  if (!button) return;
  if (button.classList.contains("select-profile-btn")) selectProfile(button.dataset.profile);
  else if (button.classList.contains("fit-start-profile-btn")) void startProfile(button.dataset.profile, button);
  else if (button.classList.contains("fit-pull-btn")) void pullModel(button.dataset.model, button);
  else if (button.classList.contains("toggle-enabled-btn")) void toggleProfileEnabled(button.dataset.profile, button.dataset.enabled === "true", button);
  else if (button.classList.contains("edit-profile-btn")) openTuningEditor(button.dataset.model, button.dataset.profile);
  else if (button.classList.contains("delete-profile-btn")) void deleteProfile(button.dataset.profile, button);
}

// One-click cleanup: delete every Ollama profile whose model isn't pulled.
// Keeps config.json an honest mirror of what's actually on the machine.

async function cleanOrphanProfiles(btn) {
  if (!state.installedLoaded) await refreshInstalled();
  const orphans = state.profiles.filter((name) => {
    const p = state.profileData[name] || {};
    if ((p.backend || "ollama") === "llamacpp") return p.model_file_exists === false;
    if (!state.installedLoaded) return false;
    return !findInstalledModel(p.model_id || name);
  });
  if (!orphans.length) {
    toast("No orphan profiles - every profile's model is pulled.", "success");
    return;
  }
  const preview = orphans.slice(0, 12).join("\n  ");
  const more = orphans.length > 12 ? `\n  …and ${orphans.length - 12} more` : "";
  if (!window.confirm(`Remove ${orphans.length} profile(s) whose model is not pulled?\n\n  ${preview}${more}\n\n(Models on disk are not touched; profiles are recreated automatically if you pull the model again.)`)) return;
  busy(btn, true);
  let removed = 0;
  try {
    for (const profile of orphans) {
      const res = await postJSON("/profiles/delete", { profile });
      if (res.success) removed += 1;
      else toast(res.error || `Could not delete ${profile}.`, "error");
    }
    toast(`Removed ${removed} orphan profile(s).`, "success");
    await loadProfiles();
    if (!$("#fit-finder-body").textContent.includes("have not been scanned")) {
      await scanConfiguredFits();
    }
  } catch (err) {
    toast(`Cleanup failed: ${err.message}`, "error");
  } finally {
    busy(btn, false);
  }
}

async function deleteProfile(profile, btn) {
  if (!profile) return;
  if (!window.confirm(`Remove the profile "${profile}" from config.json? (This does not delete the model from disk.)`)) return;
  busy(btn, true);
  try {
    const res = await postJSON("/profiles/delete", { profile });
    if (!res.success) {
      toast(res.error || "Could not delete profile.", "error");
      return;
    }
    toast(`Removed profile ${profile}.`, "success");
    await loadProfiles();
    await scanConfiguredFits();
  } catch (err) {
    toast(`Delete failed: ${err.message}`, "error");
  } finally {
    busy(btn, false);
  }
}

// ---------------------------------------------------------------------------
// Tab 1 - Tuning editor (edit a model's run profile: context, KV cache, GPU…)
// ---------------------------------------------------------------------------
// The fields a user can tune, with input types. `model_id` is upserted so the
// profile is created on the fly for models pulled before auto-create existed.

const TUNING_FIELDS = [
  { key: "description", label: "Description", type: "text" },
  { key: "supports_vision", label: "Vision (multimodal)", type: "checkbox" },
  { key: "think", label: "Thinking mode", type: "checkbox" },
  { key: "max_images", label: "Max images per request", type: "number" },
  { key: "context_limit", label: "Context limit", type: "number" },
  { key: "safe_context_limit", label: "Safe context limit", type: "number" },
  { key: "max_output_tokens", label: "Max output tokens", type: "number" },
  { key: "temperature", label: "Temperature", type: "number", step: "0.05" },
  { key: "top_p", label: "Top-p", type: "number", step: "0.05" },
  { key: "repeat_penalty", label: "Repeat penalty", type: "number", step: "0.05" },
  { key: "timeout_seconds", label: "Timeout (s)", type: "number" },
  { key: "flash_attention", label: "Flash attention (llama.cpp)", type: "checkbox" },
  { key: "kv_cache_type_k", label: "KV cache type K (e.g. q8_0)", type: "text" },
  { key: "kv_cache_type_v", label: "KV cache type V (e.g. q8_0)", type: "text" },
  { key: "gpu_layers", label: "GPU layers (number or 'all')", type: "text" },
];

// ---------------------------------------------------------------------------
// Deployment manifests + integration snippets (Release R5)
// ---------------------------------------------------------------------------

async function exportManifest(modelId, btn) {
  busy(btn, true);
  try {
    const { name } = profileForModel(modelId);
    const out = await postJSON("/system/manifest/export", { profile: name, model_id: modelId });
    if (!out.success) throw new Error(out.error || "Export failed.");
    const overlay = simpleModal(
      "Deployment manifest",
      `<code>${esc(modelId)}</code> - reproducible, human-readable YAML (JSON also available below).`,
      `<pre class="log manifest-yaml">${esc(out.yaml)}</pre>
       <div class="row gap wrap" style="margin-top:0.5rem">
         <button class="btn primary" id="manifest-copy-yaml">Copy YAML</button>
         <button class="btn" id="manifest-download-yaml">Download .yaml</button>
         <button class="btn" id="manifest-download-json">Download .json</button>
       </div>`
    );
    $("#manifest-copy-yaml", overlay)?.addEventListener("click", async () => {
      await navigator.clipboard.writeText(out.yaml);
      toast("Manifest YAML copied.", "success");
    });
    $("#manifest-download-yaml", overlay)?.addEventListener("click", () =>
      downloadFile(`localdeploy-manifest-${modelId.replace(/[^\w.-]+/g, "_")}.yaml`, out.yaml, "text/yaml")
    );
    $("#manifest-download-json", overlay)?.addEventListener("click", () =>
      downloadFile(`localdeploy-manifest-${modelId.replace(/[^\w.-]+/g, "_")}.json`, out.json, "application/json")
    );
  } catch (err) {
    toast(`Export failed: ${err.message}`, "error");
  } finally {
    busy(btn, false);
  }
}

async function openIntegrationSnippets(modelId, btn) {
  busy(btn, true);
  try {
    const { data } = profileForModel(modelId);
    const context = data?.safe_context_limit || data?.context_limit || 8192;
    const out = await getJSON(`/system/integration-snippets?model=${encodeURIComponent(modelId)}&context=${context}`);
    if (!out.success) throw new Error("Could not load integration snippets.");
    const cardsHtml = out.cards
      .map(
        (c, i) => `<div class="fit-card integration-card">
          <div class="model-title">${esc(c.label)}</div>
          <pre class="log integration-snippet">${esc(c.snippet)}</pre>
          <button class="btn compact integration-copy-btn" data-idx="${i}">Copy</button>
        </div>`
      )
      .join("");
    const overlay = simpleModal(
      "Use elsewhere",
      `<code>${esc(modelId)}</code> - copy-paste config for common tools, using this app's OpenAI-compatible <code>/v1</code> endpoints.`,
      `<div class="fit-grid">${cardsHtml}</div>`
    );
    $$(".integration-copy-btn", overlay).forEach((b) =>
      b.addEventListener("click", async () => {
        await navigator.clipboard.writeText(out.cards[Number(b.dataset.idx)].snippet);
        toast("Copied.", "success");
      })
    );
  } catch (err) {
    toast(`Could not load integration snippets: ${err.message}`, "error");
  } finally {
    busy(btn, false);
  }
}

function compatibilityReportHtml(report) {
  const symbolFor = { ok: "✓", bad: "✕", info: "△", unknown: "?" };
  const rows = (report.diffs || [])
    .map((d) => `<div class="compat-row compat-${esc(d.symbol)}"><span class="compat-mark">${symbolFor[d.symbol] || "•"}</span> ${esc(d.text)}</div>`)
    .join("");
  const subs = (report.substitutions || []).map((s) => `<div class="compat-row compat-info"><span class="compat-mark">△</span> ${esc(s)}</div>`).join("");
  const verdict = report.can_recreate
    ? `<span class="badge on">Can recreate here</span>`
    : `<span class="badge wont">Cannot recreate as-is</span>`;
  return `<div class="row gap wrap" style="margin-bottom:0.5rem">${verdict}</div>${rows}${subs}`;
}

function parseManifestInput(text) {
  const trimmed = (text || "").trim();
  if (!trimmed) return null;
  try {
    const parsed = JSON.parse(trimmed);
    return parsed?.manifest || parsed; // accept either the raw manifest or an export{...} envelope
  } catch {
    return null;
  }
}

async function validateManifest(btn) {
  const text = $("#manifest-import-text")?.value;
  const manifest = parseManifestInput(text);
  const body = $("#manifest-body");
  if (!manifest) {
    body.innerHTML = `<div class="muted">Paste or load a manifest JSON first (YAML export files should be re-saved/copied as JSON, or use "Download .json" from Export deployment).</div>`;
    $("#btn-manifest-recreate").disabled = true;
    return;
  }
  busy(btn, true);
  body.innerHTML = `<div class="muted"><span class="spin-inline"></span> Checking compatibility…</div>`;
  try {
    const out = await postJSON("/system/manifest/validate", { manifest });
    if (!out.success) throw new Error(out.error || "Validation failed.");
    body.innerHTML = compatibilityReportHtml(out);
    state.manifestToRecreate = out.can_recreate ? manifest : null;
    $("#btn-manifest-recreate").disabled = !out.can_recreate;
  } catch (err) {
    body.innerHTML = `<div class="muted">${esc(err.message)}</div>`;
    $("#btn-manifest-recreate").disabled = true;
  } finally {
    busy(btn, false);
  }
}

async function recreateManifest(btn) {
  const manifest = state.manifestToRecreate;
  if (!manifest) return;
  const body = $("#manifest-body");
  busy(btn, true);
  body.innerHTML = `<div class="muted"><span class="spin-inline"></span> Recreating deployment…</div>`;
  const log = [];
  try {
    await postMaybeStream("/system/manifest/recreate", { manifest }, (evt) => {
      if (evt.event === "pull_start") log.push(`Pulling ${evt.model}…`);
      else if (evt.event === "pull_end") log.push(`Pulled ${evt.model}.`);
      else if (evt.event === "serve_start") log.push(`Serving at context ${evt.context}…`);
      else if (evt.event === "recreate_end") {
        log.push(`Placement: ${evt.placement_observed || "unknown"} (manifest recorded ${evt.placement_expected || "unknown"}).`);
        if (evt.observed_vram_gb != null) log.push(`Observed VRAM here: ${evt.observed_vram_gb} GB (manifest: ${evt.manifest_observed_vram_gb ?? "n/a"} GB).`);
      } else if (evt.event === "error") log.push(`Error: ${evt.error}`);
      body.innerHTML = `<div class="log">${log.map(esc).join("<br/>")}</div>`;
    });
    toast("Deployment recreated.", "success");
    void refreshLiveModelState(true);
  } catch (err) {
    toast(`Recreate failed: ${err.message}`, "error");
  } finally {
    busy(btn, false);
  }
}

function wireManifestZone() {
  $("#btn-manifest-validate")?.addEventListener("click", (e) => validateManifest(e.currentTarget));
  $("#btn-manifest-recreate")?.addEventListener("click", (e) => recreateManifest(e.currentTarget));
  $("#manifest-import-file")?.addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    $("#manifest-import-text").value = await file.text();
  });
}

function profileForModel(modelId, profileName) {
  if (profileName && state.profileData[profileName]) return { name: profileName, data: state.profileData[profileName] };
  const hit = Object.entries(state.profileData).find(([, v]) => v.model_id === modelId);
  return hit ? { name: hit[0], data: hit[1] } : { name: null, data: {} };
}

function openTuningEditor(modelId, profileName) {
  const { name, data } = profileForModel(modelId, profileName);
  const heading = name || modelId;
  const fieldsHtml = TUNING_FIELDS.map((f) => {
    const val = data[f.key];
    if (f.type === "checkbox") {
      return `<label class="check tuning-field"><input type="checkbox" data-key="${f.key}" ${val ? "checked" : ""} /><span>${esc(f.label)}</span></label>`;
    }
    const step = f.step ? ` step="${f.step}"` : "";
    const v = val == null ? "" : esc(String(val));
    return `<label class="field tuning-field"><span>${esc(f.label)}</span><input type="${f.type}"${step} data-key="${f.key}" value="${v}" /></label>`;
  }).join("");
  const overlay = document.createElement("div");
  overlay.className = "modal-overlay";
  overlay.innerHTML = `<div class="modal-card" role="dialog" aria-modal="true">
      <div class="card-head">
        <div><h3 class="sub">Edit tuning</h3><div class="muted small">Model: <code>${esc(modelId)}</code>${name ? ` · profile <code>${esc(name)}</code>` : " · a profile will be created"}</div></div>
        <button class="btn compact modal-close">✕</button>
      </div>
      <div class="tuning-grid">${fieldsHtml}</div>
      <div class="row gap" style="justify-content:flex-end;margin-top:0.75rem">
        <button class="btn modal-close">Cancel</button>
        <button class="btn primary modal-save">Save profile</button>
      </div>
    </div>`;
  document.body.appendChild(overlay);
  const close = () => overlay.remove();
  overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
  $$(".modal-close", overlay).forEach((b) => b.addEventListener("click", close));
  $(".modal-save", overlay).addEventListener("click", async (e) => {
    const saveBtn = e.currentTarget;
    const fields = {};
    $$("input[data-key]", overlay).forEach((inp) => {
      const key = inp.dataset.key;
      if (inp.type === "checkbox") fields[key] = inp.checked;
      else if (inp.type === "number") fields[key] = inp.value.trim() === "" ? null : Number(inp.value);
      else fields[key] = inp.value.trim() === "" ? null : inp.value.trim();
    });
    busy(saveBtn, true);
    try {
      const payload = name ? { profile: name, fields } : { model_id: modelId, fields };
      const res = await postJSON("/profiles/upsert", payload);
      if (!res.success) {
        toast(res.error || "Could not save profile.", "error");
        return;
      }
      toast(`Saved profile ${res.profile}.`, "success");
      close();
      await loadProfiles();
    } catch (err) {
      toast(`Save failed: ${err.message}`, "error");
    } finally {
      busy(saveBtn, false);
    }
  });
}

function renderProfileFitCard(row) {
  const { profile, p, modelId, installed, fit } = row;
  const req = fit?.estimate_gb?.required;
  const margin = fit?.margin_gb;
  const installedBadge = installed ? `<span class="badge on">installed</span>` : `<span class="badge off">not pulled</span>`;
  const vision = p.supports_vision ? `<span class="badge">vision</span>` : "";
  const backend = p.backend ? `<span class="badge">${esc(p.backend)}</span>` : "";
  const enabledNow = p.enabled !== false;
  const enabledBadge = enabledNow ? `<span class="badge on">enabled</span>` : `<span class="badge off">disabled</span>`;
  const toggleBtn = `<button class="btn compact toggle-enabled-btn" data-profile="${esc(profile)}" data-enabled="${enabledNow ? "false" : "true"}" title="${enabledNow ? "Disable" : "Enable"} this profile for Auto-pick">${enabledNow ? "Disable" : "Enable"}</button>`;
  const action = installed
    ? `<button class="btn fit-start-profile-btn" data-profile="${esc(profile)}">Deploy</button>`
    : `<button class="btn fit-pull-btn" data-model="${esc(modelId)}">Pull</button>`;
  const detail = [
    req != null ? `needs ~${req} GB` : null,
    margin != null ? `${margin >= 0 ? "+" : ""}${margin} GB headroom` : null,
    p.safe_context_limit ? `${p.safe_context_limit} ctx` : null,
    p.max_output_tokens ? `${p.max_output_tokens} max output` : null,
  ].filter(Boolean).join(" · ");
  const desc = p.description ? `<div class="muted small">${esc(p.description)}</div>` : "";
  return `<div class="fit-card">
    <div class="running-top">
      <div>
        <div class="model-title">${esc(profile)}</div>
        <div class="muted small">Model: <code>${esc(modelId)}</code></div>
      </div>
      ${fitBadge(fit || {})}
    </div>
    ${fitMeterHtml(fit)}
    ${desc}
    <div class="muted small">${esc(fit?.headline || "No fit estimate available.")}</div>
    <div class="muted small">${esc(detail)}</div>
    <div class="row gap wrap fit-card-actions">
      ${enabledBadge}${installedBadge}${vision}${backend}
      <span class="spacer"></span>
      ${toggleBtn}
      <button class="btn compact edit-profile-btn" data-profile="${esc(profile)}" data-model="${esc(modelId)}" title="Edit tuning">Edit</button>
      <button class="btn compact danger delete-profile-btn" data-profile="${esc(profile)}" title="Remove this profile from config.json">Delete</button>
      <button class="btn select-profile-btn" data-profile="${esc(profile)}">Select</button>
      ${action}
    </div>
  </div>`;
}

async function toggleProfileEnabled(profile, enabled, btn) {
  busy(btn, true);
  try {
    const res = await postJSON("/system/set-enabled", { profile, enabled });
    if (!res.success) {
      toast(res.error || "Could not update profile.", "error");
      return;
    }
    if (state.profileData[profile]) state.profileData[profile].enabled = enabled;
    toast(`${profile} ${enabled ? "enabled" : "disabled"} for Auto-pick.`, "success");
    await scanConfiguredFits();
  } catch (err) {
    toast(`Update failed: ${err.message}`, "error");
  } finally {
    busy(btn, false);
  }
}

function selectProfile(profile) {
  if (!profile) return;
  $("#profile-select").value = profile;
  $("#bench-profile-select").value = profile;
  toast(`Selected ${profile}.`, "success");
}

async function startProfile(profile, btn) {
  if (!profile) return;
  selectProfile(profile);
  busy(btn, true);
  const node = $("#serve-result");
  node.className = "result";
  const stop = startElapsed(node, "Loading model into memory");
  try {
    const res = await postJSON("/models/serve", {
      profile,
      keep_alive: $("#keep-alive").value.trim() || "60m",
      device: $("#serve-device").value,
    });
    stop();
    showServeResult(res);
    await refreshLiveModelState(true);
  } catch (err) {
    stop();
    toast(`Deploy failed: ${err.message}`, "error");
  } finally {
    busy(btn, false);
  }
}

// ---------------------------------------------------------------------------
// Tab 1 - Starter pack (curated first-pull picks for the detected budget)
// ---------------------------------------------------------------------------

const REASON_KIND_LABEL = { estimated: "estimated", published: "published spec", measured: "measured here" };

function reasonListHtml(reasons) {
  if (!reasons || !reasons.length) return "";
  return `<ul class="reason-list">${reasons
    .map((r) => {
      const mark = r.kind === "measured" ? "⚡" : r.kind === "published" ? "△" : "✓";
      return `<li class="reason-item reason-${esc(r.kind)}"><span class="reason-mark">${mark}</span> ${esc(r.text)} <span class="badge reason-kind-badge" title="Provenance">${esc(REASON_KIND_LABEL[r.kind] || r.kind)}</span></li>`;
    })
    .join("")}</ul>`;
}

const RECOMMEND_BUCKET_TITLE = { recommended: "★ Recommended", faster: "⚡ Faster", higher_quality: "◆ Higher quality" };

function renderRecommendCard(c) {
  if (!c) return "";
  const installed = !!findInstalledModel(c.pull_name);
  const action = installed
    ? `<span class="badge on">installed</span><button class="btn primary starter-deploy-btn" data-model="${esc(c.pull_name)}">Deploy</button>`
    : `<button class="btn primary starter-pull-btn" data-model="${esc(c.pull_name)}">Download and start</button>`;
  const vision = c.vision ? `<span class="badge">vision</span>` : "";
  const confBadge = `<span class="badge confidence-${esc(c.confidence)}" title="How much this estimate should be trusted">confidence: ${esc(c.confidence)}</span>`;
  return `<div class="fit-card model-card recommendation-card">
    <div class="bucket-label">${esc(RECOMMEND_BUCKET_TITLE[c.bucket] || c.bucket)}</div>
    <div class="running-top">
      <div class="model-identity">
        <span class="model-mark" aria-hidden="true">M</span>
        <div>
          <div class="model-title">${esc(c.id)}</div>
          <div class="muted small">${esc(c.use_case || "")}</div>
        </div>
      </div>
      <span class="badge" title="Hand-curated quality rating; 5 is strongest in its size class">quality ${esc(c.tier)}/5</span>
    </div>
    <div class="model-card-copy">${esc(c.description || "")}</div>
    <div class="muted small model-card-reason">${esc(c.why_summary || "")}</div>
    <details class="why-recommended">
      <summary>Why this model?</summary>
      ${reasonListHtml(c.reasons)}
    </details>
    <div class="row gap wrap fit-card-actions">
      ${vision}
      ${confBadge}
      <span class="spacer"></span>
      ${action}
    </div>
  </div>`;
}

function handleStarterPackClick(event) {
  const button = event.target.closest("button");
  if (!button) return;
  if (button.classList.contains("starter-pull-btn")) void pullModel(button.dataset.model, button);
  else if (button.classList.contains("starter-deploy-btn")) void startInstalledModel(resolveInstalledName(button.dataset.model), button);
}

// ---------------------------------------------------------------------------
// Automated bakeoff - "Compare top models for me" (Release R6)
// Disabled per product decision - commented out rather than deleted so it's
// easy to re-enable later. The backend (/system/bakeoff/run) is untouched.
// ---------------------------------------------------------------------------
/*

function bakeoffDownloadBudgetGb() {
  const sel = $("#bakeoff-budget")?.value;
  if (sel === "custom") return Math.max(1, Number($("#bakeoff-budget-custom")?.value || 50));
  return Number(sel || 30);
}

function wireBakeoffBudgetSelect() {
  $("#bakeoff-budget")?.addEventListener("change", (e) => {
    $("#bakeoff-budget-custom-wrap")?.classList.toggle("hidden", e.target.value !== "custom");
  });
}

function bakeoffCandidateRowHtml(id, state_) {
  const st = state_ || { phase: "queued" };
  const phaseLabel = {
    queued: "Queued", pulling: "Downloading…", deploying: "Deploying…",
    benchmarking: "Benchmarking…", done: "Done", failed: "Failed",
  }[st.phase] || st.phase;
  const cls = st.phase === "done" ? "on" : st.phase === "failed" ? "wont" : st.phase === "queued" ? "off" : "cpu";
  const detail = st.phase === "failed" ? st.reason || "" : st.detail || "";
  return `<div class="bakeoff-row" data-model="${esc(id)}">
    <span class="badge ${cls}">${esc(phaseLabel)}</span>
    <b>${esc(id)}</b>
    <span class="muted small">${esc(detail)}</span>
  </div>`;
}

function renderBakeoffProgress(candidateOrder, candidateStates) {
  const body = $("#bakeoff-body");
  body.innerHTML = `<div class="bakeoff-progress">${candidateOrder.map((id) => bakeoffCandidateRowHtml(id, candidateStates[id])).join("")}</div>`;
}

function renderBakeoffResult(evt) {
  const body = $("#bakeoff-body");
  const rows = (evt.ranked || [])
    .map(
      (r, i) => `<tr class="${r.profile === evt.winner ? "bakeoff-winner-row" : ""}">
        <td>${i === 0 ? "🏆 " : ""}${esc(r.profile)}</td>
        <td class="num">${esc(r.passed)}/${esc(r.tests)}</td>
        <td class="num">${esc(r.avg_accuracy)}</td>
        <td class="num">${esc(r.avg_latency_s)}s</td>
        <td class="num">${r.margin_gb != null ? `${esc(r.margin_gb)} GB` : "-"}</td>
      </tr>`
    )
    .join("");
  body.innerHTML = `<div class="bakeoff-winner-card">
      <div class="eyebrow">Winner${evt.winner_deployed ? " · deployed" : ""}</div>
      <h3 class="sub">${esc(evt.winner)}</h3>
    </div>
    <div class="table-wrap"><table class="results">
      <thead><tr><th>Model</th><th class="num">Completed</th><th class="num">Accuracy</th><th class="num">Avg latency</th><th class="num">Headroom</th></tr></thead>
      <tbody>${rows}</tbody>
    </table></div>
    <div class="row gap wrap" style="margin-top:0.5rem">
      <button class="btn compact" id="bakeoff-export-winner">Export winner's deployment</button>
      ${evt.losers && evt.losers.length ? `<button class="btn compact danger" id="bakeoff-remove-losers">Remove ${evt.losers.length} other downloaded model(s)</button>` : ""}
    </div>`;
  $("#bakeoff-export-winner")?.addEventListener("click", (e) => exportManifest(evt.winner, e.currentTarget));
  $("#bakeoff-remove-losers")?.addEventListener("click", async (e) => {
    if (!window.confirm(`Delete these ${evt.losers.length} model(s) from disk?\n\n${evt.losers.join("\n")}`)) return;
    busy(e.currentTarget, true);
    for (const model of evt.losers) {
      try {
        await postJSON("/models/delete", { model });
      } catch {
        // best-effort; move on to the next loser rather than aborting the whole cleanup
      }
    }
    toast("Removed losing models.", "success");
    await refreshLiveModelState(true);
    busy(e.currentTarget, false);
  });
}

async function runBakeoff(source) {
  const btn = source?.currentTarget || source || $("#btn-bakeoff-run");
  busy(btn, true);
  const body = $("#bakeoff-body");
  body.innerHTML = `<div class="muted"><span class="spin-inline"></span> Selecting fit-safe candidates…</div>`;
  const candidateStates = {};
  let candidateOrder = [];
  try {
    await postMaybeStream(
      "/system/bakeoff/run",
      {
        use_case: $("#rec-use-case")?.value || null,
        priority: $("#rec-priority")?.value || "balanced",
        expected_context: parseInt($("#rec-context")?.value || "8192", 10),
        download_budget_gb: bakeoffDownloadBudgetGb(),
        free_vram_mb: targetVram(),
      },
      (evt) => {
        if (evt.event === "bakeoff_start") {
          candidateOrder = evt.candidates || [];
          candidateOrder.forEach((id) => (candidateStates[id] = { phase: "queued" }));
          renderBakeoffProgress(candidateOrder, candidateStates);
        } else if (evt.event === "candidate_start") {
          candidateStates[evt.model] = { phase: "pulling", detail: `~${evt.download_gb} GB` };
          renderBakeoffProgress(candidateOrder, candidateStates);
        } else if (evt.event === "pull_progress") {
          candidateStates[evt.model] = { phase: "pulling", detail: evt.status || "downloading…" };
          renderBakeoffProgress(candidateOrder, candidateStates);
        } else if (evt.event === "deploy_start") {
          candidateStates[evt.model] = { phase: "deploying", detail: "Deploying…" };
          renderBakeoffProgress(candidateOrder, candidateStates);
        } else if (evt.event === "test_start") {
          candidateStates[evt.model] = { phase: "benchmarking", detail: `Running ${evt.name}…` };
          renderBakeoffProgress(candidateOrder, candidateStates);
        } else if (evt.event === "candidate_end") {
          candidateStates[evt.model] = { phase: "done", detail: `accuracy ${evt.avg_accuracy} · ${evt.avg_latency_s}s avg` };
          renderBakeoffProgress(candidateOrder, candidateStates);
        } else if (evt.event === "candidate_failed") {
          candidateStates[evt.model] = { phase: "failed", reason: evt.reason };
          renderBakeoffProgress(candidateOrder, candidateStates);
        } else if (evt.event === "bakeoff_end") {
          renderBakeoffResult(evt);
        } else if (evt.event === "error") {
          body.innerHTML = `<div class="muted">${esc(evt.error)}</div>`;
        }
      }
    );
  } catch (err) {
    body.innerHTML = `<div class="muted">Bakeoff failed: ${esc(err.message)}</div>`;
    toast(`Bakeoff failed: ${err.message}`, "error");
  } finally {
    busy(btn, false);
  }
}
*/

async function recommendModels(source) {
  const btn = source?.currentTarget || source || $("#btn-recommend-models");
  busy(btn, true);
  const body = $("#starter-pack-body");
  body.innerHTML = `<div class="muted"><span class="spin-inline"></span> Finding models that fit your workload and hardware…</div>`;
  try {
    const data = await postJSON("/registry/recommend", {
      use_case: $("#rec-use-case")?.value || null,
      priority: $("#rec-priority")?.value || "balanced",
      expected_context: parseInt($("#rec-context")?.value || "8192", 10),
      usage_mode: $("#rec-usage-mode")?.value || "single_user_chat",
      free_vram_mb: targetVram(),
      margin_gb: 2.0,
    });
    if (!data.success || (!data.recommended && !data.faster && !data.higher_quality)) {
      body.innerHTML = `<div class="muted">${esc(data.message || "Could not determine a fit budget.")}</div>`;
      return;
    }
    const unit = data.budget_source === "vram" ? "VRAM" : "RAM";
    const marginNote = data.margin_relaxed ? "" : ` (after a 2 GB safety margin)`;
    const header = `<div class="muted small" style="margin-bottom:0.5rem">Budget: ~${esc(data.budget_gb)} GB usable ${unit}${marginNote} from ~${esc(data.raw_budget_gb)} GB detected.</div>`;
    const note = data.message ? `<div class="muted small">${esc(data.message)}</div>` : "";
    const cards = [data.recommended, data.faster, data.higher_quality].filter(Boolean);
    body.innerHTML = header + note + `<div class="fit-grid">` + cards.map(renderRecommendCard).join("") + `</div>`;
  } catch (err) {
    body.innerHTML = `<div class="muted">Recommendation lookup failed.</div>`;
    toast(`Recommendation failed: ${err.message}`, "error");
  } finally {
    busy(btn, false);
  }
}

// ---------------------------------------------------------------------------
// Tab 1 - Hugging Face model search
// ---------------------------------------------------------------------------

const CATALOG_PAGE_SIZE = 12;

async function loadProviderCatalog(source) {
  const btn = source?.currentTarget || source || $("#btn-provider-refresh");
  const body = $("#provider-catalog-body");
  busy(btn, true);
  body.innerHTML = `<div class="muted"><span class="spin-inline"></span> Checking local providers…</div>`;
  try {
    const data = await getJSON("/registry/providers");
    state.catalog.rows = data.models || [];
    state.catalog.providers = data.providers || [];
    state.catalog.loaded = true;
    state.catalog.page = 0;
    renderProviderStatuses();
    populateCatalogProviderFilter();
    renderProviderCatalog();
  } catch (err) {
    body.innerHTML = `<div class="muted">Provider inventory failed: ${esc(err.message)}</div>`;
  } finally {
    busy(btn, false);
  }
}

// How to bring an unreachable runtime online, shown in the status chip tooltip.

const PROVIDER_HINTS = {
  ollama: "Install/start Ollama from ollama.com",
  lmstudio: "In LM Studio: Developer -> Start local server (port 1234)",
  vllm: "Set VLLM_BASE_URL in .env and start your vLLM server",
  docker: "Enable Docker Model Runner (port 12434)",
  llamacpp: "Start llama-server and set ENABLE_LLAMA_CPP=true",
  openai: "Point a profile at your OpenAI-compatible server",
};

function renderProviderStatuses() {
  const slot = $("#provider-statuses");
  if (!slot) return;
  slot.innerHTML = state.catalog.providers
    .map((provider) => {
      const n = (provider.models || []).length;
      const hint = provider.reachable
        ? `${provider.base_url} - ${n} model${n === 1 ? "" : "s"}`
        : `${provider.error || "unreachable"}. ${PROVIDER_HINTS[provider.provider] || ""}`;
      return `<span class="badge ${provider.reachable ? "on" : "off"}" title="${esc(hint)}">${provider.reachable ? "●" : "○"} ${esc(provider.provider)}${provider.reachable ? ` · ${n}` : ""}</span>`;
    })
    .join(" ");
}

function populateCatalogProviderFilter() {
  const sel = $("#catalog-provider");
  if (!sel) return;
  const current = sel.value;
  const names = [...new Set(state.catalog.rows.map((m) => m.provider))].sort();
  sel.innerHTML = `<option value="">All</option>` + names.map((n) => `<option value="${esc(n)}"${n === current ? " selected" : ""}>${esc(n)}</option>`).join("");
}

function catalogParamsB(model) {
  const explicit = parseFloat(String(model.parameters || ""));
  if (Number.isFinite(explicit) && explicit > 0) return explicit;
  const match = String(model.model || "").toLowerCase().match(/(\d+(?:\.\d+)?)\s*b\b/);
  return match ? parseFloat(match[1]) : null;
}

function catalogSizeBucket(paramsB) {
  if (paramsB == null) return null;
  if (paramsB < 4) return "tiny";
  if (paramsB < 8) return "small";
  if (paramsB < 15) return "mid";
  return "large";
}

function filteredCatalogRows() {
  const q = ($("#catalog-search")?.value || "").trim().toLowerCase();
  const provider = $("#catalog-provider")?.value || "";
  const size = $("#catalog-size")?.value || "";
  const sort = $("#catalog-sort")?.value || "tps";
  const rows = state.catalog.rows.filter((m) => {
    if (provider && m.provider !== provider) return false;
    if (size && catalogSizeBucket(catalogParamsB(m)) !== size) return false;
    if (q) {
      const hay = `${m.model} ${m.publisher || ""} ${m.provider} ${m.quant || ""}`.toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });
  const by = {
    tps: (a, b) => (b.tokens_per_second ?? -1) - (a.tokens_per_second ?? -1) || a.model.localeCompare(b.model),
    name: (a, b) => a.model.localeCompare(b.model),
    params: (a, b) => (catalogParamsB(b) ?? -1) - (catalogParamsB(a) ?? -1) || a.model.localeCompare(b.model),
    provider: (a, b) => a.provider.localeCompare(b.provider) || a.model.localeCompare(b.model),
  };
  return rows.sort(by[sort] || by.tps);
}

function renderProviderCatalog() {
  const body = $("#provider-catalog-body");
  const pager = $("#catalog-pagination");
  if (!body || !state.catalog.loaded) return;
  const rows = filteredCatalogRows();
  if (!rows.length) {
    body.innerHTML = `<div class="empty-state">No models match${state.catalog.rows.length ? " these filters" : " - no reachable runtime reported any models yet"}.</div>`;
    pager?.classList.add("hidden");
    return;
  }
  const pages = Math.max(1, Math.ceil(rows.length / CATALOG_PAGE_SIZE));
  state.catalog.page = Math.min(state.catalog.page, pages - 1);
  const startIdx = state.catalog.page * CATALOG_PAGE_SIZE;
  const pageRows = rows.slice(startIdx, startIdx + CATALOG_PAGE_SIZE);
  body.innerHTML = `<div class="table-wrap provider-table-wrap"><table class="results provider-table">
      <thead><tr><th>Model</th><th>Runtime</th><th>Publisher</th><th class="num">Params</th><th>Quant</th><th class="num" title="Measured from your saved benchmark runs">tok/s ⚡</th><th class="num">Context</th><th></th></tr></thead>
      <tbody>${pageRows
        .map(
          (model) => `<tr>
        <td><b>${esc(model.model)}</b></td>
        <td><span class="badge">${esc(model.provider)}</span></td>
        <td>${esc(model.publisher || "-")}</td>
        <td class="num">${esc(model.parameters || (catalogParamsB(model) != null ? catalogParamsB(model) + "B" : "-"))}</td>
        <td>${quantLabelHtml(model.quant, "quant-code")}</td>
        <td class="num" title="${model.benchmark_samples ? `${esc(model.benchmark_samples)} saved samples` : "Not benchmarked yet - run it in Benchmark & Compare"}">${model.tokens_per_second != null ? esc(model.tokens_per_second) : "-"}</td>
        <td class="num">${model.context != null ? esc(model.context) : "-"}</td>
        <td><button class="btn compact provider-profile-btn" data-model="${esc(model.model)}" data-provider="${esc(model.provider)}" data-base-url="${esc(model.base_url)}" title="Create a run profile for this model">＋ Add profile</button></td>
      </tr>`
        )
        .join("")}</tbody>
    </table></div>`;
  if (pager) {
    pager.classList.toggle("hidden", pages <= 1);
    $("#catalog-page-label").textContent = `${startIdx + 1}–${Math.min(startIdx + CATALOG_PAGE_SIZE, rows.length)} of ${rows.length} models`;
    $("#catalog-prev").disabled = state.catalog.page === 0;
    $("#catalog-next").disabled = state.catalog.page >= pages - 1;
  }
}

async function addProviderProfile(button) {
  busy(button, true);
  try {
    const result = await postJSON("/profiles/upsert", {
      model_id: button.dataset.model,
      backend: button.dataset.provider,
      base_url: button.dataset.baseUrl,
    });
    if (!result.success) throw new Error(result.error || "Profile creation failed.");
    await loadProfiles();
    toast(`Profile ${result.profile} is ready.`, "success");
  } catch (err) {
    toast(`Could not add provider profile: ${err.message}`, "error");
  } finally {
    busy(button, false);
  }
}

function handleProviderCatalogClick(event) {
  const button = event.target.closest(".provider-profile-btn");
  if (button) void addProviderProfile(button);
}

function fmtNum(n) {
  if (n == null) return null;
  const value = Number(n);
  if (!Number.isFinite(value)) return null;
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}k`;
  return String(value);
}

function hfStatsHtml(candidate) {
  const downloads = fmtNum(candidate.downloads);
  const likes = fmtNum(candidate.likes);
  const items = [
    downloads == null ? "downloads not reported" : `${downloads} downloads`,
    likes == null ? "likes not reported" : `${likes} likes`,
  ];
  return `<span class="meta hf-stats" title="Hugging Face downloads and likes; the search API does not return a quality rating.">${esc(items.join(" · "))}</span>`;
}

function remoteCatalogSearchButton(source) {
  const candidate = source?.currentTarget || source;
  return candidate?.id === "btn-hf-search" ? candidate : $("#btn-hf-search");
}

// Looks like an exact pullable tag ("gemma3:4b", "hf.co/org/repo")? Then the
// results offer pulling it verbatim, which absorbs the old "pull by name" flow.

function exactTagQuery(query) {
  if (/^hf\.co\//i.test(query)) return query;
  if (/^modelscope\.cn\//i.test(query) && /:[\w.\-]+\.gguf$/i.test(query)) return query;
  if (/^[\w.\-]+(?:\/[\w.\-]+)?:[\w.\-]+$/.test(query)) return query;
  return null;
}

function sourceBadge(source) {
  if (source === "huggingface") {
    return `<span class="badge src-hf" title="Hugging Face models are pulled through Ollama's hf.co/ shortcut"><span class="source-dot"></span>Hugging Face</span>`;
  }
  if (source === "modelscope") {
    return `<span class="badge src-modelscope" title="ModelScope models are pulled through Ollama's modelscope.cn/ shortcut (requires Ollama 0.30+)"><span class="source-dot"></span>ModelScope</span>`;
  }
  return `<span class="badge src-ollama" title="Official Ollama model library"><span class="source-dot"></span>Ollama</span>`;
}

// "270m" / "0.8b" / "7b" -> billions of parameters (null when unparseable).

function paramsFromSizeToken(token) {
  const m = String(token || "").toLowerCase().match(/^([\d.]+)\s*([mb])$/);
  if (!m) return null;
  const value = parseFloat(m[1]);
  if (!Number.isFinite(value)) return null;
  return m[2] === "m" ? value / 1000 : value;
}

function popularityNumber(value) {
  if (value == null) return null;
  if (Number.isFinite(Number(value))) return Number(value);
  const match = String(value).trim().match(/^([\d.]+)\s*([kmb]?)$/i);
  if (!match) return null;
  return Number(match[1]) * ({ "": 1, k: 1e3, m: 1e6, b: 1e9 }[match[2].toLowerCase()] || 1);
}

export function paramsFromModelName(name) {
  const matches = [...String(name || "").toLowerCase().matchAll(/(?:^|[-_:/])(\d+(?:\.\d+)?)\s*([mb])(?:$|[-_.])/g)];
  if (!matches.length) return null;
  const match = matches[matches.length - 1];
  const value = Number(match[1]);
  return match[2] === "m" ? value / 1000 : value;
}

function expandRemoteCatalog(models) {
  return (models || []).flatMap((model) => {
    let variants = Array.isArray(model.variants) ? model.variants : [];
    if (!variants.length && model.sizes?.length) {
      variants = model.sizes.map((label) => ({
        label,
        params_b: paramsFromSizeToken(label),
        pull_name: `${model.name}:${label}`,
      }));
    }
    if (!variants.length) variants = [{ label: null, params_b: paramsFromModelName(model.name), pull_name: model.pull_name }];
    return variants.map((variant, index) => {
      const paramsB = Number(variant.params_b) > 0 ? Number(variant.params_b) : paramsFromModelName(model.name);
      const pullName = variant.pull_name || model.pull_name || model.name;
      return {
        ...model,
        row_id: `${model.source}:${model.name}:${variant.label || index}`,
        family: model.family || model.name,
        size_label: variant.label || (paramsB != null ? `${paramsB}b` : null),
        params_b: paramsB,
        quant: variant.quant || null,
        context: variant.context || null,
        download_bytes: variant.download_bytes || null,
        pull_name: pullName,
        popularity: model.popularity ?? popularityNumber(model.pulls),
      };
    });
  });
}

function remoteFit(row) {
  return row.params_b != null ? state.remoteCatalog.fits[String(row.params_b)] || null : null;
}

function fitBadgeForCatalog(row) {
  const fit = remoteFit(row);
  if (!fit) return `<span class="badge unknown" title="The parameter size is unknown, so fit cannot be estimated">unknown</span>`;
  const labels = { ok: "fits GPU", soft: fit.tier === "cpu_only" ? "CPU only" : "tight", hard: "will not fit", unknown: "unknown" };
  const classes = { ok: "fits", soft: "tight", hard: "wont", unknown: "unknown" };
  return `<span class="badge ${classes[fit.severity] || "unknown"}" data-tooltip="Estimated need: ${esc(fit.required_gb)} GB at a typical Q4 quant and 4K context. Compare quants for an exact estimate." tabindex="0">${esc(labels[fit.severity] || "unknown")}</span>`;
}

export function remoteSizeMatches(paramsB, filter) {
  if (filter === "all") return true;
  if (paramsB == null) return filter === "unknown";
  if (filter === "under4") return paramsB < 4;
  if (filter === "4to8") return paramsB >= 4 && paramsB <= 8;
  if (filter === "8to15") return paramsB > 8 && paramsB < 15;
  if (filter === "15to35") return paramsB >= 15 && paramsB < 35;
  if (filter === "over35") return paramsB >= 35;
  return true;
}

function remoteCatalogFilteredRows() {
  const source = $("#remote-source-filter")?.value || "all";
  const size = $("#remote-size-filter")?.value || "all";
  const fit = $("#remote-fit-filter")?.value || "all";
  const capability = $("#remote-cap-filter")?.value || "all";
  const installedOnly = !!$("#remote-installed-filter")?.checked;
  const rows = state.remoteCatalog.rows.filter((row) => {
    if (source !== "all" && row.source !== source) return false;
    if (!remoteSizeMatches(row.params_b, size)) return false;
    const rowFit = remoteFit(row);
    if (fit !== "all" && (rowFit?.severity || "unknown") !== fit) return false;
    if (capability !== "all" && !(row.capabilities || []).includes(capability)) return false;
    if (installedOnly && !row.installed_match && !findInstalledModel(row.pull_name)) return false;
    return true;
  });
  const sort = state.remoteCatalog.sort || $("#remote-sort")?.value || "popularity-desc";
  const [key, direction = "asc"] = sort.split("-");
  const factor = direction === "desc" ? -1 : 1;
  const value = (row) => {
    if (key === "model") return String(row.name || "").toLowerCase();
    if (key === "source") return String(row.source || "");
    if (key === "params") return row.params_b ?? null;
    if (key === "updated") return row.updated ? String(row.updated) : null;
    return row.popularity ?? null;
  };
  return rows.sort((a, b) => {
    const av = value(a);
    const bv = value(b);
    if (av == null && bv != null) return 1;
    if (bv == null && av != null) return -1;
    if (av == null && bv == null) return String(a.name).localeCompare(String(b.name));
    const compared = typeof av === "string" ? av.localeCompare(bv) : av - bv;
    return compared * factor || String(a.name).localeCompare(String(b.name));
  });
}

function catalogSortHeading(label, key, className = "") {
  const active = state.remoteCatalog.sort?.startsWith(`${key}-`);
  const direction = active && state.remoteCatalog.sort.endsWith("desc") ? "descending" : active ? "ascending" : "none";
  const arrow = direction === "ascending" ? " ↑" : direction === "descending" ? " ↓" : "";
  return `<th class="${className}" aria-sort="${direction}"><button class="catalog-sort-btn" data-sort-key="${key}">${esc(label)}${arrow}</button></th>`;
}

function remoteCatalogRow(row) {
  const wanted = String(row.pull_name || "").toLowerCase().replace(/:latest$/, "");
  const exactInstalled = Object.values(state.installedByName).some((model) =>
    String(model.name || "").toLowerCase().replace(/:latest$/, "") === wanted
  );
  const installed = exactInstalled || (row.params_b == null && row.installed_match)
    ? `<span class="badge on" title="A matching model is installed">installed</span>`
    : "";
  const caps = (row.capabilities || []).slice(0, 3).map((c) => `<span class="badge">${esc(c)}</span>`).join("");
  const params = row.params_b != null ? `${row.params_b < 1 ? row.params_b * 1000 + "M" : row.params_b + "B"}` : "-";
  const pullAction = row.pullable === false
    ? `<a class="btn compact" href="${esc(row.url)}" target="_blank" rel="noopener">View</a>`
    : `<button class="btn primary compact library-pull-btn" data-model="${esc(row.pull_name)}" title="Pull this exact ${row.size_label ? `${esc(row.size_label)} size` : "model"}">Pull</button>`;
  const quantAction = row.params_b != null
    ? `<button class="btn compact quant-jump-btn" data-model="${esc(row.source === "ollama" && row.size_label ? `${row.family}:${row.size_label}` : row.name)}" data-params="${esc(row.params_b)}" data-source="${esc(row.source)}" data-family="${esc(row.family)}">Compare quants</button>`
    : "";
  const quant = row.quant ? quantLabelHtml(row.quant, "badge model-quant") : "";
  return `<tr data-catalog-row="${esc(row.row_id)}">
    <td class="catalog-model-cell" data-label="Model">
      <div class="catalog-result-title"><a href="${esc(row.url)}" target="_blank" rel="noopener">${esc(row.name)}</a>${quant}${installed}</div>
      <div class="catalog-caps">${caps}</div>
      ${row.description ? `<div class="muted small catalog-description" title="${esc(row.description)}">${esc(row.description)}</div>` : ""}
    </td>
    <td data-label="Source">${sourceBadge(row.source)}</td>
    <td class="num" data-label="Parameters"><strong>${esc(params)}</strong></td>
    <td data-label="Best fit">${fitBadgeForCatalog(row)}</td>
    <td class="num" data-label="Popularity" title="Pulls on Ollama or downloads on Hugging Face">${esc(row.pulls || "-")}</td>
    <td class="muted small" data-label="Updated">${esc(row.updated || "-")}</td>
    <td class="catalog-actions" data-label="Actions"><div class="catalog-action-stack">${quantAction}${pullAction}</div></td>
  </tr>`;
}

function updateRemoteFilterSummary(total, shown) {
  const slot = $("#remote-active-filters");
  if (!slot) return;
  const controls = [
    ["Source", $("#remote-source-filter")], ["Parameters", $("#remote-size-filter")],
    ["Fit", $("#remote-fit-filter")], ["Capability", $("#remote-cap-filter")],
  ];
  const active = controls.filter(([, control]) => control && control.value !== "all")
    .map(([label, control]) => `<span class="filter-chip">${label}: ${esc(control.options[control.selectedIndex].textContent)}</span>`);
  if ($("#remote-installed-filter")?.checked) active.push(`<span class="filter-chip">Installed only</span>`);
  slot.classList.toggle("hidden", !active.length);
  slot.innerHTML = active.length ? `<span class="muted small">Showing ${shown} of ${total} size-specific rows</span>${active.join("")}` : "";
}

function renderRemoteCatalog(message = null) {
  const body = $("#updates-body");
  if (!body) return;
  const rows = remoteCatalogFilteredRows();
  updateRemoteFilterSummary(state.remoteCatalog.rows.length, rows.length);
  const exact = exactTagQuery(state.remoteCatalog.query);
  const exactRow = exact
    ? `<div class="exact-pull-row">Exact pullable tag detected <button class="btn primary compact library-pull-btn" data-model="${esc(exact)}">Pull ${esc(exact)}</button></div>`
    : "";
  if (!rows.length) {
    body.innerHTML = `${exactRow}<div class="empty-state">No size-specific models match the current filters.</div>`;
  } else {
    body.innerHTML = `${exactRow}<div class="catalog-result-count"><strong>${rows.length}</strong> size-specific result${rows.length === 1 ? "" : "s"}<span class="muted small">One row per model parameter size; compare quants before pulling when memory is tight.</span></div>
      <div class="table-wrap remote-catalog-wrap"><table class="results catalog-table">
        <thead><tr>${catalogSortHeading("Model", "model")}${catalogSortHeading("Source", "source")}${catalogSortHeading("Parameters", "params", "num")}<th>Best fit</th>${catalogSortHeading("Popularity", "popularity", "num")}${catalogSortHeading("Updated", "updated")}<th><span class="sr-only">Actions</span></th></tr></thead>
        <tbody>${rows.map(remoteCatalogRow).join("")}</tbody></table></div>${message ? `<p class="muted small">${esc(message)}</p>` : ""}`;
  }
}

async function loadRemoteCatalogFits() {
  const params = [...new Set(state.remoteCatalog.rows.map((row) => row.params_b).filter((value) => value > 0))];
  state.remoteCatalog.fits = {};
  if (!params.length) return;
  try {
    const res = await postJSON("/system/fit-batch", { params_b: params, free_vram_mb: targetVram() });
    (res.items || []).forEach((item) => (state.remoteCatalog.fits[String(item.params_b)] = item));
  } catch {
    // The catalog remains useful when a hardware fit estimate is unavailable.
  }
}

async function searchUnifiedModels(source) {
  const btn = remoteCatalogSearchButton(source);
  const body = $("#updates-body");
  const statusSlot = $("#catalog-source-status");
  const query = ($("#hf-search")?.value || "").trim();
  const seq = ++state.unifiedSearchSeq;
  busy(btn, true);
  state.remoteCatalogLoaded = true;
  body.innerHTML = `<div class="muted"><span class="spin-inline"></span> Searching the Ollama library, Hugging Face, and ModelScope...</div>`;
  try {
    const data = await postJSON("/registry/search-models", { query, limit: 30 });
    if (seq !== state.unifiedSearchSeq) return; // a newer keystroke superseded this
    if (statusSlot) {
      statusSlot.innerHTML = Object.entries(data.sources || {})
        .map(([name, s]) => `<span class="badge ${s.online ? "on" : "off"}" title="${esc(s.error || `${s.count} results`)}">${s.online ? "●" : "○"} ${esc(name)}${s.online ? ` · ${s.count}` : ""}</span>`)
        .join(" ");
    }
    if (!data.online && !(data.results || []).length) {
      body.innerHTML = `<div class="empty-state">${esc(data.message || "No model source is reachable.")}</div>`;
      return;
    }
    state.remoteCatalog.sourceRows = data.results || [];
    state.remoteCatalog.rows = expandRemoteCatalog(data.results || []);
    state.remoteCatalog.query = query;
    state.remoteCatalog.sort = $("#remote-sort")?.value || "popularity-desc";
    renderRemoteCatalog(data.message);
    await loadRemoteCatalogFits();
    if (seq === state.unifiedSearchSeq) renderRemoteCatalog(data.message);
  } catch (err) {
    if (seq !== state.unifiedSearchSeq) return;
    body.innerHTML = `<div class="empty-state">Model search failed.</div>`;
    toast(`Model search failed: ${err.message}`, "error");
  } finally {
    busy(btn, false);
  }
}

// Jump to the quant advisor pre-filled with a model from the results table.

const checkUpdates = searchUnifiedModels;

function openQuantAdvisor(modelId, paramsB = null, source = "ollama", family = null) {
  const input = $("#quant-model");
  if (input) input.value = modelId;
  if (input) {
    input.dataset.params = paramsB || "";
    input.dataset.source = source || "ollama";
    input.dataset.family = family || "";
  }
  $('.seg-btn[data-seg="quant"]')?.click();
  quantAdvise($("#btn-quant-advise"));
}

// Back-compat alias: older wiring and tests refer to checkUpdates.

function hideOllamaHelp() {
  $("#ollama-help")?.classList.add("hidden");
}

function showOllamaHelp(retryModel, retryBtn, message) {
  const help = $("#ollama-help");
  if (!help) return;
  const retryHtml = retryModel
    ? `<button class="btn primary compact" id="btn-retry-pull">Retry pull</button>`
    : "";
  const detail = message || `Install it (or start it if it's already installed), then ${retryModel ? "retry the pull" : "try pulling a model"}.`;
  help.innerHTML = `<div><strong>Ollama isn't reachable</strong><div class="muted small">${esc(detail)}</div></div>
    <div class="row gap">
      <button class="btn compact" id="btn-install-ollama">Install / start Ollama</button>
      ${retryHtml}
    </div>`;
  help.classList.remove("hidden");
  help.scrollIntoView({ behavior: "smooth", block: "nearest" });
  $("#btn-install-ollama").addEventListener("click", async (e) => {
    const b = e.currentTarget;
    busy(b, true);
    try {
      const res = await postJSON("/system/install-ollama", {});
      toast(res.message || res.error || "Done.", res.success ? "success" : "error");
      if (res.success) hideOllamaHelp();
    } catch (err) {
      toast(`Install failed: ${err.message}`, "error");
    } finally {
      busy(b, false);
    }
  });
  if (retryModel) {
    $("#btn-retry-pull").addEventListener("click", () => {
      hideOllamaHelp();
      pullModel(retryModel, retryBtn);
    });
  }
}

// Flip any rendered Pull button for `model` (starter pack, HF search, saved-
// profile scan) to an "installed" badge without waiting on a full re-render.

function markModelInstalledInUI(model) {
  let selector;
  try {
    selector = `[data-model="${CSS.escape(model)}"]`;
  } catch {
    return;
  }
  $$(`.starter-pull-btn${selector}, .hf-pull-btn${selector}, .fit-pull-btn${selector}`).forEach((b) => {
    const badge = document.createElement("span");
    badge.className = "badge on";
    badge.textContent = "installed";
    b.replaceWith(badge);
  });
}

// ---------------------------------------------------------------------------
// Tab 1 - Pull progress panel (%, speed, ETA, destination, completion)
// ---------------------------------------------------------------------------

function setPullProgress({ percent, status, stats }) {
  const bar = $("#pull-progress-bar");
  const pctLabel = $("#pull-progress-percent");
  if (percent === null) {
    bar.classList.add("indeterminate");
    bar.style.width = "100%";
    pctLabel.textContent = "";
  } else if (percent != null) {
    const p = Math.max(0, Math.min(100, percent));
    bar.classList.remove("indeterminate");
    bar.style.width = `${p.toFixed(1)}%`;
    bar.setAttribute("aria-valuenow", String(Math.round(p)));
    pctLabel.textContent = `${Math.round(p)}%`;
  }
  if (status != null) $("#pull-progress-status").textContent = status;
  if (stats != null) $("#pull-progress-stats").textContent = stats;
}

function showPullDone(model) {
  const done = $("#pull-progress-done");
  done.innerHTML = `<div class="pull-done-row">
      <div><strong>✓ Pulled - ${esc(model)} is ready</strong>
        <div class="muted small">A run profile was created for it. Deploy it from Your models, or right here.</div>
      </div>
      <div class="row gap wrap">
        <button class="btn compact" id="btn-view-pulled" data-model="${esc(model)}">View in Your models</button>
        <button class="btn primary compact" id="btn-deploy-pulled" data-model="${esc(model)}">Deploy now</button>
      </div>
    </div>`;
  done.classList.remove("hidden");
  $("#btn-pull-dismiss").hidden = false;
  $("#btn-deploy-pulled").addEventListener("click", (e) => {
    const b = e.currentTarget;
    startInstalledModel(resolveInstalledName(b.dataset.model), b);
  });
  $("#btn-view-pulled").addEventListener("click", (e) => highlightInstalledModel(e.currentTarget.dataset.model));
}

function dismissPullProgress() {
  state.pullRetry = null;
  $("#pull-progress")?.classList.add("hidden");
  $("#pull-progress-actions")?.classList.add("hidden");
  $("#pull-progress-done")?.classList.add("hidden");
  $("#btn-pull-dismiss").hidden = true;
}

function showPullTerminal(kind, model, message, onRetry) {
  const actions = $("#pull-progress-actions");
  const retry = kind === "cancelled" || kind === "failed" || kind === "blocked";
  actions.innerHTML = `<div>
      <strong>${kind === "cancelled" ? "Pull cancelled" : kind === "blocked" ? "Pull blocked" : "Pull failed"}</strong>
      <div class="muted small">${esc(message)}</div>
    </div>
    <div class="row gap wrap">
      ${retry ? `<button class="btn primary compact" id="btn-pull-retry">Retry</button>` : ""}
      <button class="btn compact" id="btn-pull-close">Dismiss</button>
    </div>`;
  actions.className = `pull-progress-actions ${kind}`;
  $("#btn-pull-dismiss").hidden = false;
  $("#btn-pull-close")?.addEventListener("click", dismissPullProgress);
  $("#btn-pull-retry")?.addEventListener("click", () => {
    actions.classList.add("hidden");
    (onRetry || (() => pullModel(model)))();
  });
}

// Scroll to + briefly flash the freshly-pulled model's row so it never looks
// like the pull "vanished" (the row lives lower in the page than the pull panel).

function highlightInstalledModel(model) {
  const name = resolveInstalledName(model);
  if (!name) return;
  let row;
  try {
    row = $(`#installed-body [data-model="${CSS.escape(name)}"]`);
  } catch {
    return;
  }
  if (!row) return;
  row.scrollIntoView({ behavior: "smooth", block: "center" });
  row.classList.add("just-added");
  setTimeout(() => row.classList.remove("just-added"), 2600);
}

// ---------------------------------------------------------------------------
// Tab 1 - Pull (streamed, fit-gated)
// ---------------------------------------------------------------------------

async function pullModel(modelArg, triggerBtn) {
  // Called from the manual Pull button (no args), or a candidate row's own
  // Pull button (model name + that button, so the card shows it's pulling).
  const model = typeof modelArg === "string" && modelArg ? modelArg : $("#pull-model").value.trim();
  if (!model) {
    toast("Enter a model name to pull.", "error");
    return;
  }
  $("#pull-model").value = model;
  const manualBtn = $("#btn-pull");
  const btn = triggerBtn || manualBtn;
  const usesSpinner = btn === manualBtn;
  const cardOriginalHtml = usesSpinner ? null : btn.innerHTML;
  const cancelBtn = $("#btn-pull-cancel");
  const log = $("#pull-log");

  hideOllamaHelp();
  // Reset + reveal the progress panel.
  const panel = $("#pull-progress");
  state.pullRetry = model;
  $("#pull-progress-done").classList.add("hidden");
  $("#pull-progress-done").innerHTML = "";
  $("#pull-progress-actions").classList.add("hidden");
  $("#pull-progress-actions").innerHTML = "";
  $("#btn-pull-dismiss").hidden = true;
  $("#pull-progress-title").textContent = model;
  $("#pull-progress-dest").textContent = "";
  setPullProgress({ percent: 0, status: "Starting…", stats: "" });
  panel.classList.remove("hidden");
  panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
  log.textContent = "";
  const append = (line) => {
    log.textContent += line + "\n";
    log.scrollTop = log.scrollHeight;
  };

  if (usesSpinner) busy(btn, true);
  else {
    btn.disabled = true;
    btn.textContent = "Pulling…";
  }

  // Wire a cancel that aborts the stream (closes the connection; Ollama stops).
  const controller = new AbortController();
  cancelBtn.hidden = false;
  const onCancel = () => {
    cancelBtn.disabled = true;
    cancelBtn.textContent = "Cancelling…";
    setPullProgress({ percent: null, status: "Cancelling request…", stats: "Downloaded layers may be retained by Ollama for a later retry." });
    controller.abort();
  };
  cancelBtn.addEventListener("click", onCancel);

  let ollamaUnreachable = false;
  let sawError = false;
  const checkOllamaError = (msg) => /ollama is (not running|installed but not reachable)|not reachable at/i.test(msg || "");

  // Progress accounting across the model's layers (Ollama streams per-digest
  // total/completed). Overall % = sum(completed)/sum(total); speed is a smoothed
  // byte-rate; ETA derives from the remaining bytes at that rate.
  const digests = {};
  let lastTs = performance.now();
  let lastBytes = 0;
  let speed = 0; // bytes/sec, EMA
  let overallPct = null;

  const recompute = (statusText) => {
    let total = 0;
    let completed = 0;
    for (const d of Object.values(digests)) {
      total += d.total || 0;
      completed += d.completed || 0;
    }
    const now = performance.now();
    const dt = (now - lastTs) / 1000;
    if (dt >= 0.4 && completed >= lastBytes) {
      const inst = (completed - lastBytes) / dt;
      speed = speed ? speed * 0.7 + inst * 0.3 : inst;
      lastTs = now;
      lastBytes = completed;
    }
    if (total > 0) {
      overallPct = (completed / total) * 100;
      const parts = [`${fmtBytes(completed)} / ${fmtBytes(total)}`];
      if (speed > 0) parts.push(`${(speed / 1e6).toFixed(1)} MB/s`);
      const eta = speed > 0 ? (total - completed) / speed : null;
      if (eta != null && Number.isFinite(eta) && eta > 0.5) parts.push(`ETA ${fmtDuration(eta)}`);
      setPullProgress({ percent: overallPct, status: statusText, stats: parts.join(" · ") });
    } else {
      setPullProgress({ percent: null, status: statusText, stats: "" });
    }
  };

  try {
    const out = await postMaybeStream(
      "/models/pull",
      { model, free_vram_mb: targetVram(), allow_override: $("#pull-override").checked },
      (evt) => {
        if (evt.destination_label && !$("#pull-progress-dest").textContent) {
          $("#pull-progress-dest").textContent = `→ ${evt.destination_label}`;
        }
        if (evt.error) {
          append(`error: ${evt.error}`);
          sawError = true;
          if (checkOllamaError(evt.error)) ollamaUnreachable = true;
          setPullProgress({ status: `Error: ${evt.error}` });
          return;
        }
        // A soft fit note (e.g. "won't fit GPU - will run on CPU") rides the start event.
        if (evt.note) append(`note: ${evt.note}`);
        if (evt.digest && evt.total) {
          digests[evt.digest] = { total: evt.total, completed: evt.completed || 0 };
        }
        if (evt.status) {
          const layerPct = evt.total && evt.completed ? Math.round((evt.completed / evt.total) * 100) : null;
          append(`${evt.status}${layerPct != null ? ` ${layerPct}%` : ""}`);
          recompute(evt.status);
          if (!usesSpinner) btn.textContent = overallPct != null ? `Pulling ${Math.round(overallPct)}%` : "Pulling…";
        }
      },
      controller.signal
    );
    if (!out.streamed) {
      const j = out.json || {};
      if (j.blocked_by === "fit-check") {
        const msg = j.message || "Model may not fit available memory.";
        append(`blocked by fit check: ${msg}`);
        if (j.fit?.estimate_gb) append(`  needs ~${j.fit.estimate_gb.required} GB`);
        setPullProgress({ percent: 0, status: `Blocked by fit check - tick "Warn only; pull anyway" to continue.` });
        showPullTerminal("blocked", model, msg);
      } else {
        const msg = j.error || "Pull could not start.";
        append(msg);
        if (checkOllamaError(msg)) ollamaUnreachable = true;
        setPullProgress({ status: `Error: ${msg}` });
        showPullTerminal("failed", model, msg);
      }
    } else if (sawError) {
      // Stream opened but reported a failure mid-way - don't claim success or
      // flip any card to "installed".
      showPullTerminal("failed", model, "The model source reported an error. Open Raw log for details.");
    } else {
      append("done.");
      setPullProgress({ percent: 100, status: "Pulled successfully.", stats: "" });
      await Promise.all([
        loadProfiles({ notify: false }),
        refreshLiveModelState(true, { notify: false }),
      ]);
      notifyModelChanges("pull", true);
      markModelInstalledInUI(model);
      showPullDone(model);
      highlightInstalledModel(model);
      toast(`Pulled ${model}.`, "success");
    }
  } catch (err) {
    if (err.name === "AbortError") {
      append("cancelled.");
      setPullProgress({ percent: 0, status: "Cancelled.", stats: "The transfer stopped. Ollama may keep completed layers for a faster retry." });
      showPullTerminal("cancelled", model, "The transfer stopped. Dismiss this panel or retry the same model.");
      toast("Pull cancelled.", "info");
    } else {
      append(`error: ${err.message}`);
      setPullProgress({ status: `Error: ${err.message}` });
      showPullTerminal("failed", model, err.message);
    }
  } finally {
    cancelBtn.hidden = true;
    cancelBtn.disabled = false;
    cancelBtn.textContent = "Cancel";
    cancelBtn.removeEventListener("click", onCancel);
    if (usesSpinner) busy(btn, false);
    else {
      btn.disabled = false;
      btn.innerHTML = cardOriginalHtml;
    }
    if (ollamaUnreachable) showOllamaHelp(model, triggerBtn);
  }
}

// ---------------------------------------------------------------------------
// Tab 1 - Import a GGUF file (local path, or a direct URL download)
// ---------------------------------------------------------------------------

async function importLocalGguf() {
  const btn = $("#btn-import-local-gguf");
  const input = $("#local-gguf-path");
  const path = input.value.trim();
  if (!path) {
    toast("Enter a local GGUF file path.", "error");
    return;
  }
  busy(btn, true);
  try {
    const check = await postJSON("/system/check-local-gguf", { path });
    if (!check.success || !check.exists) {
      toast(check.error || "File not found.", "error");
      return;
    }
    const sizeGb = check.size_bytes ? (check.size_bytes / 1e9).toFixed(1) : null;
    const res = await postJSON("/profiles/upsert", {
      model_id: check.path,
      backend: "llamacpp",
      fields: { description: `Imported local GGUF file${sizeGb ? ` (${sizeGb} GB)` : ""}.` },
    });
    if (!res.success) {
      toast(res.error || "Could not create a profile for this file.", "error");
      return;
    }
    toast(`Imported. Profile "${res.profile}" created - start llama-server, then Deploy.`, "success");
    input.value = "";
    await loadProfiles();
  } catch (err) {
    toast(`Import failed: ${err.message}`, "error");
  } finally {
    busy(btn, false);
  }
}

async function importGgufFromUrl() {
  const btn = $("#btn-import-url-gguf");
  const url = $("#import-gguf-url").value.trim();
  const nameOverride = $("#import-gguf-name").value.trim();
  if (!url) {
    toast("Enter a GGUF file URL.", "error");
    return;
  }
  hideOllamaHelp();
  const panel = $("#pull-progress");
  state.pullRetry = null;
  $("#pull-progress-done").classList.add("hidden");
  $("#pull-progress-done").innerHTML = "";
  $("#pull-progress-actions").classList.add("hidden");
  $("#pull-progress-actions").innerHTML = "";
  $("#btn-pull-dismiss").hidden = true;
  $("#pull-progress-title").textContent = nameOverride || url;
  $("#pull-progress-dest").textContent = "";
  setPullProgress({ percent: 0, status: "Starting…", stats: "" });
  panel.classList.remove("hidden");
  panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
  const log = $("#pull-log");
  log.textContent = "";
  const append = (line) => {
    log.textContent += line + "\n";
    log.scrollTop = log.scrollHeight;
  };

  busy(btn, true);
  const cancelBtn = $("#btn-pull-cancel");
  const controller = new AbortController();
  cancelBtn.hidden = false;
  const onCancel = () => {
    cancelBtn.disabled = true;
    cancelBtn.textContent = "Cancelling…";
    controller.abort();
  };
  cancelBtn.addEventListener("click", onCancel);

  let sawError = false;
  let ollamaUnreachable = false;
  const checkOllamaError = (msg) => /ollama is (not running|installed but not reachable)|not reachable at/i.test(msg || "");
  let importedModel = null;

  try {
    const out = await postMaybeStream(
      "/models/import-url",
      { url, model: nameOverride || null },
      (evt) => {
        if (evt.model) importedModel = evt.model;
        if (evt.error) {
          append(`error: ${evt.error}`);
          sawError = true;
          if (checkOllamaError(evt.error)) ollamaUnreachable = true;
          setPullProgress({ status: `Error: ${evt.error}` });
          return;
        }
        if (evt.status) {
          const pct = evt.total ? Math.round((evt.completed / evt.total) * 100) : null;
          append(`${evt.status}${pct != null ? ` ${pct}%` : ""}`);
          if (evt.total) {
            setPullProgress({
              percent: (evt.completed / evt.total) * 100,
              status: evt.status,
              stats: `${fmtBytes(evt.completed)} / ${fmtBytes(evt.total)}`,
            });
          } else {
            setPullProgress({ percent: evt.completed ? null : 0, status: evt.status, stats: evt.completed ? fmtBytes(evt.completed) : "" });
          }
        }
      },
      controller.signal
    );
    if (!out.streamed) {
      const msg = out.json?.error || "Import could not start.";
      append(msg);
      setPullProgress({ status: `Error: ${msg}` });
      showPullTerminal("failed", null, msg, () => importGgufFromUrl());
    } else if (sawError) {
      showPullTerminal("failed", null, "The import failed. Open Raw log for details.", () => importGgufFromUrl());
    } else {
      append("done.");
      setPullProgress({ percent: 100, status: "Imported successfully.", stats: "" });
      await Promise.all([loadProfiles({ notify: false }), refreshLiveModelState(true, { notify: false })]);
      notifyModelChanges("pull", true);
      if (importedModel) {
        markModelInstalledInUI(importedModel);
        showPullDone(importedModel);
        highlightInstalledModel(importedModel);
      }
      toast(`Imported ${importedModel || "model"}.`, "success");
    }
  } catch (err) {
    if (err.name === "AbortError") {
      append("cancelled.");
      setPullProgress({ percent: 0, status: "Cancelled." });
      showPullTerminal("cancelled", null, "The import stopped.", () => importGgufFromUrl());
      toast("Import cancelled.", "info");
    } else {
      append(`error: ${err.message}`);
      setPullProgress({ status: `Error: ${err.message}` });
      showPullTerminal("failed", null, err.message, () => importGgufFromUrl());
    }
  } finally {
    cancelBtn.hidden = true;
    cancelBtn.disabled = false;
    cancelBtn.textContent = "Cancel";
    cancelBtn.removeEventListener("click", onCancel);
    busy(btn, false);
    if (ollamaUnreachable) showOllamaHelp(null, null, "Ollama isn't reachable, so the import couldn't finish.");
  }
}

// Free all loaded models from memory/VRAM (the "unstick / reset" button).

async function freeMemory() {
  const btn = $("#btn-free");
  busy(btn, true);
  try {
    const res = await postJSON("/models/free", {});
    if (res.success) {
      toast(res.message || "Memory freed.", "success");
      await refreshLiveModelState(true);
    } else {
      toast(res.error || "Could not free memory.", "error");
    }
  } catch (err) {
    toast(`Free memory failed: ${err.message}`, "error");
  } finally {
    busy(btn, false);
  }
}

// Delete a model from disk (with confirm), then refresh the installed list.

async function deleteModel(name, btn) {
  const bytes = state.installedByName[name]?.size || 0;
  const frees = bytes ? ` This frees ${(bytes / 1e9).toFixed(1)} GB of disk space.` : "";
  if (!window.confirm(`Delete "${name}" from disk?${frees}\n\nYou can always pull it again later.`)) {
    return;
  }
  busy(btn, true);
  try {
    const res = await postJSON("/models/delete", { model: name });
    if (res.success) {
      toast(res.message || `Deleted ${name}.`, "success");
      await refreshLiveModelState(true);
    } else {
      toast(res.error || res.message || "Could not delete.", "error");
    }
  } catch (err) {
    toast(`Delete failed: ${err.message}`, "error");
  } finally {
    busy(btn, false);
  }
}

// ---------------------------------------------------------------------------
// Tab 1 - Quant advisor: fit-check every common quant of one model size
// ---------------------------------------------------------------------------

function quantBadge(v) {
  const cls = { ok: "on", soft: "tight", hard: "off" }[v.severity] || "";
  const label =
    v.severity === "ok" ? "comfortable"
    : v.tier === "tight" ? "tight"
    : v.tier === "cpu_only" ? "CPU only"
    : v.severity === "hard" ? "won't fit"
    : "unknown";
  return `<span class="badge ${cls}">${esc(label)}</span>`;
}

// Fill the "Pull it" column with real, published tags for this family+size,
// so pulling a specific quant is one click on a tag that actually exists -
// no guessing tag-name conventions.

async function attachQuantPullButtons(res, source = "ollama") {
  const family = res.model?.family;
  const paramsB = res.model?.params_b;
  const cells = $$(".quant-pull-cell");
  const extra = $("#quant-tags-extra");
  if (source !== "ollama") {
    cells.forEach((c) => (c.innerHTML = `<span class="muted small" title="Open the Hugging Face repository to choose an exact GGUF file">repository</span>`));
    return;
  }
  if (!family || !cells.length) {
    cells.forEach((c) => (c.innerHTML = `<span class="muted small" title="Not an Ollama library model - pull it by its own name">-</span>`));
    return;
  }
  let data;
  try {
    data = await postJSON("/registry/library-tags", { model: family });
  } catch {
    data = null;
  }
  if (!data?.success || !data.online || !(data.tags || []).length) {
    cells.forEach((c) => (c.innerHTML = `<span class="muted small" title="${esc(data?.message || "Tag list unavailable")}">-</span>`));
    return;
  }
  const sizeToken = `${String(paramsB).replace(/\.0$/, "")}b`;
  const norm = (s) => String(s).toLowerCase().replace(/[-_]/g, "");
  const sizeTags = data.tags.filter((tag) => norm(tag.tag).includes(norm(sizeToken)));
  cells.forEach((cell) => {
    const quant = cell.dataset.quant;
    const match = sizeTags.find((tag) => norm(tag.tag).includes(norm(quant)));
    if (!match) {
      cell.innerHTML = `<span class="muted small" title="ollama.com publishes no ${esc(sizeToken)} tag with ${esc(quant)}">not published</span>`;
      return;
    }
    cell.innerHTML = match.installed
      ? `<span class="badge on" title="${esc(match.full)} is already pulled">✓ pulled</span>`
      : `<button class="btn primary compact quant-pull-btn" data-model="${esc(match.full)}" title="Pull ${esc(match.full)}${match.size ? ` (${esc(match.size)} download)` : ""} - fit-checked first">↓ ${esc(match.size || "Pull")}</button>`;
  });
  if (extra) {
    const chips = data.tags
      .slice(0, 80)
      .map((tag) =>
        tag.installed
          ? `<span class="badge on" title="already pulled">${esc(tag.tag)}</span>`
          : `<button class="btn compact quant-pull-btn" data-model="${esc(tag.full)}" title="Pull ${esc(tag.full)}${tag.size ? ` (${esc(tag.size)} download)` : ""}">${esc(tag.tag)}${tag.size ? ` · ${esc(tag.size)}` : ""}</button>`
      )
      .join("");
    extra.innerHTML = `<details class="quant-all-tags"><summary>All ${data.tags.length} published ${esc(family)} tags (click any to pull)</summary><div class="quant-tag-grid">${chips}</div></details>`;
  }
}

async function quantAdvise(btn) {
  const modelId = $("#quant-model").value.trim();
  const input = $("#quant-model");
  const explicitParams = Number(input?.dataset.params) || undefined;
  const source = input?.dataset.source || "ollama";
  const body = $("#quant-body");
  if (!modelId) {
    body.innerHTML = `<div class="muted">Enter a model name that includes the size, e.g. <code>gemma3:12b</code>.</div>`;
    return;
  }
  busy(btn, true);
  body.innerHTML = skeletonHtml(3);
  try {
    const res = await postJSON("/system/quant-advisor", {
      model_id: modelId,
      params_b: explicitParams,
      context: Number($("#quant-context").value) || undefined,
      free_vram_mb: targetVram(),
    });
    if (!res.success) {
      body.innerHTML = `<div class="muted">${esc(res.message || "Could not estimate.")}</div>`;
      return;
    }
    const budget = res.free_vram_gb != null ? `${res.free_vram_gb} GB budget` : "no VRAM budget";
    const rows = res.variants
      .map((v) => {
        const margin = v.margin_gb != null ? `${v.margin_gb >= 0 ? "+" : ""}${v.margin_gb} GB` : "-";
        return `<tr>
          <td>${quantLabelHtml(v.quant, "quant-code")}</td>
          <td class="num">~${esc(v.weights_gb)} GB</td>
          <td class="num">~${esc(v.required_gb)} GB</td>
          <td class="num">${esc(margin)}</td>
          <td>${quantBadge(v)}</td>
          <td class="muted small">${esc(v.quality)}</td>
          <td class="quant-pull-cell" data-quant="${esc(v.quant)}"><span class="muted small">…</span></td>
        </tr>`;
      })
      .join("");
    const link = source === "ollama" && res.tags_url
      ? `<a href="${esc(res.tags_url)}" target="_blank" rel="noopener">Browse ${esc(res.model.family)}'s actual tags on ollama.com ↗</a>`
      : source === "huggingface"
        ? `<a href="https://huggingface.co/${esc(input?.dataset.family || modelId)}" target="_blank" rel="noopener">Browse exact GGUF files on Hugging Face ↗</a>`
        : "";
    const familyRows = state.remoteCatalog.rows.filter((row) => row.family === (input?.dataset.family || res.model.family) && row.params_b != null);
    const sizeSwitcher = familyRows.length > 1
      ? `<div class="quant-size-switcher"><span class="muted small">Parameter size</span>${familyRows.map((row) => `<button class="btn compact quant-size-btn${row.params_b === res.model.params_b ? " active" : ""}" data-model="${esc(row.source === "ollama" && row.size_label ? `${row.family}:${row.size_label}` : row.name)}" data-params="${esc(row.params_b)}" data-source="${esc(row.source)}" data-family="${esc(row.family)}">${esc(row.size_label || `${row.params_b}b`)}</button>`).join("")}</div>`
      : "";
    body.innerHTML = `
      <div class="next-action" style="margin-bottom:0.6rem"><b>${esc(res.recommendation)}</b></div>
      ${sizeSwitcher}
      <div class="muted small" style="margin-bottom:0.5rem">${esc(res.model.params_b)}B parameters · ${esc(res.model.context)} context · ${esc(budget)}</div>
      <div class="table-wrap"><table class="results quant-table">
        <thead><tr><th title="Lower precision uses less memory but can reduce quality">Quant</th><th class="num" title="Estimated model weight size">Weights</th><th class="num" title="Estimated weights, KV cache, and runtime overhead">Needs</th><th class="num" title="Memory budget remaining after the estimate">Headroom</th><th>Fit</th><th>Quality</th><th class="quant-pull-col">Pull it</th></tr></thead>
        <tbody>${rows}</tbody>
      </table></div>
      <div id="quant-tags-extra"></div>
      <p class="muted small quant-legend"><b>Reading quant names:</b> the number is the approximate bit class, <code>K</code> is the newer K-quant family, and <code>M</code> is its medium mixed-precision variant. Hover or focus a quant for its tradeoff.</p>
      <p class="muted small" style="margin-top:0.5rem">${esc(res.note)} ${link}</p>`;
    void attachQuantPullButtons(res, source);
  } catch (err) {
    body.innerHTML = `<div class="muted">Estimate failed.</div>`;
    toast(`Quant advisor failed: ${err.message}`, "error");
  } finally {
    busy(btn, false);
  }
}

function handleQuantBodyClick(event) {
  const button = event.target.closest("button");
  if (!button) return;
  if (button.classList.contains("quant-size-btn")) {
    openQuantAdvisor(button.dataset.model, Number(button.dataset.params), button.dataset.source, button.dataset.family);
  } else if (button.classList.contains("quant-pull-btn")) {
    void pullModel(button.dataset.model, button);
  }
}

// ---------------------------------------------------------------------------
// Tab: Chat playground - installed Ollama models with explicit load state
// ---------------------------------------------------------------------------

export async function setDefaultProfile(profile) {
  const result = await postJSON("/system/set-default", { profile });
  if (!result.success) throw new Error(result.error || "Could not set default.");
  state.defaultProfile = profile;
  renderProfileSelectOptions();
  notifyModelChanges("default-profile");
  return result;
}

function wireGetModelSegments() {
  const buttons = $$(".seg-btn");
  buttons.forEach((button) => button.addEventListener("click", () => {
    const segment = button.dataset.seg;
    buttons.forEach((candidate) => {
      const active = candidate === button;
      candidate.classList.toggle("active", active);
      candidate.setAttribute("aria-selected", active ? "true" : "false");
    });
    $$(".seg-panel").forEach((panel) => panel.classList.toggle("hidden", panel.dataset.segPanel !== segment));
    if (segment === "hf" && !state.remoteCatalogLoaded) void searchUnifiedModels();
  }));
}

function handleInstalledListClick(event) {
  const button = event.target.closest("button");
  const row = button?.closest(".mrow[data-model]");
  if (!button || !row) return;
  const model = row.dataset.model;
  if (button.classList.contains("fit-btn")) void fitCheckRow(row, true);
  else if (button.classList.contains("del-btn")) void deleteModel(model, button);
  else if (button.classList.contains("start-installed-btn")) void startInstalledModel(model, button);
  else if (button.classList.contains("unload-installed-btn")) void killRunningModel(model, button);
  else if (button.classList.contains("edit-tuning-btn")) openTuningEditor(model);
}

function handleInstalledListChange(event) {
  const checkbox = event.target.closest(".model-select");
  if (!checkbox) return;
  const model = checkbox.closest(".mrow[data-model]")?.dataset.model;
  if (!model) return;
  if (checkbox.checked) state.installedSelection.add(model);
  else state.installedSelection.delete(model);
  updateBulkDeleteBar();
}

function handleRemoteCatalogClick(event) {
  const button = event.target.closest("button");
  if (!button) return;
  if (button.classList.contains("library-pull-btn")) {
    void pullModel(button.dataset.model, button);
  } else if (button.classList.contains("quant-jump-btn")) {
    openQuantAdvisor(button.dataset.model, Number(button.dataset.params), button.dataset.source, button.dataset.family);
  } else if (button.classList.contains("catalog-sort-btn")) {
    const key = button.dataset.sortKey;
    const current = state.remoteCatalog.sort || "";
    state.remoteCatalog.sort = current.startsWith(`${key}-`) && current.endsWith("asc") ? `${key}-desc` : `${key}-asc`;
    const select = $("#remote-sort");
    if (select && [...select.options].some((option) => option.value === state.remoteCatalog.sort)) {
      select.value = state.remoteCatalog.sort;
    }
    renderRemoteCatalog();
  }
}

export function initModels() {
  if (initialized) return;
  initialized = true;
  wireGetModelSegments();
  wireManifestZone();
  $("#status-body")?.addEventListener("click", handleStatusBodyClick);
  $("#installed-body")?.addEventListener("click", handleInstalledListClick);
  $("#installed-body")?.addEventListener("change", handleInstalledListChange);
  $("#updates-body")?.addEventListener("click", handleRemoteCatalogClick);
  $("#fit-finder-body")?.addEventListener("click", handleFitFinderClick);
  $("#starter-pack-body")?.addEventListener("click", handleStarterPackClick);
  $("#provider-catalog-body")?.addEventListener("click", handleProviderCatalogClick);
  $("#quant-body")?.addEventListener("click", handleQuantBodyClick);
  $("#btn-status")?.addEventListener("click", () => void refreshModels({ profiles: false, installed: false }));
  $("#btn-serve")?.addEventListener("click", serveModel);
  $("#btn-switch")?.addEventListener("click", switchModel);
  $("#btn-installed")?.addEventListener("click", refreshInstalled);
  $("#btn-hf-search")?.addEventListener("click", (event) => searchUnifiedModels(event));
  $("#hf-search")?.addEventListener("keydown", (event) => {
    if (event.key !== "Enter") return;
    event.preventDefault(); clearTimeout(state.unifiedSearchTimer); void searchUnifiedModels(event);
  });
  $("#hf-search")?.addEventListener("input", () => {
    clearTimeout(state.unifiedSearchTimer);
    state.unifiedSearchTimer = setTimeout(() => void searchUnifiedModels(), 650);
  });
  ["#remote-source-filter", "#remote-size-filter", "#remote-fit-filter", "#remote-cap-filter"].forEach((selector) =>
    $(selector)?.addEventListener("change", () => renderRemoteCatalog()));
  $("#remote-sort")?.addEventListener("change", (event) => { state.remoteCatalog.sort = event.target.value; renderRemoteCatalog(); });
  $("#remote-installed-filter")?.addEventListener("change", () => renderRemoteCatalog());
  $("#btn-remote-clear")?.addEventListener("click", () => {
    ["#remote-source-filter", "#remote-size-filter", "#remote-fit-filter", "#remote-cap-filter"].forEach((selector) => { const control = $(selector); if (control) control.value = "all"; });
    if ($("#remote-installed-filter")) $("#remote-installed-filter").checked = false;
    state.remoteCatalog.sort = "popularity-desc";
    if ($("#remote-sort")) $("#remote-sort").value = state.remoteCatalog.sort;
    renderRemoteCatalog();
  });
  $("#btn-provider-refresh")?.addEventListener("click", (event) => loadProviderCatalog(event));
  $("#catalog-search")?.addEventListener("input", () => { state.catalog.page = 0; renderProviderCatalog(); });
  ["#catalog-provider", "#catalog-size", "#catalog-sort"].forEach((selector) => $(selector)?.addEventListener("change", () => { state.catalog.page = 0; renderProviderCatalog(); }));
  $("#catalog-prev")?.addEventListener("click", () => { state.catalog.page -= 1; renderProviderCatalog(); });
  $("#catalog-next")?.addEventListener("click", () => { state.catalog.page += 1; renderProviderCatalog(); });
  $("#btn-fit-profiles")?.addEventListener("click", scanConfiguredFits);
  $("#btn-clean-orphans")?.addEventListener("click", (event) => cleanOrphanProfiles(event.currentTarget));
  $("#btn-quant-advise")?.addEventListener("click", (event) => quantAdvise(event.currentTarget));
  $("#quant-model")?.addEventListener("input", (event) => { event.target.dataset.params = ""; event.target.dataset.source = "ollama"; event.target.dataset.family = ""; });
  $("#quant-model")?.addEventListener("keydown", (event) => { if (event.key === "Enter") { event.preventDefault(); void quantAdvise($("#btn-quant-advise")); } });
  $("#installed-sort")?.addEventListener("change", (event) => { state.installedSort = event.target.value; if (state.installedList.length) renderInstalledList(); });
  $("#btn-bulk-delete")?.addEventListener("click", (event) => bulkDeleteSelected(event.currentTarget));
  $("#btn-bulk-clear")?.addEventListener("click", () => { state.installedSelection.clear(); $$("#installed-body .model-select").forEach((checkbox) => { checkbox.checked = false; }); updateBulkDeleteBar(); });
  $("#btn-recommend-models")?.addEventListener("click", (event) => recommendModels(event));
  $("#btn-free")?.addEventListener("click", freeMemory);
  $("#btn-pull")?.addEventListener("click", () => pullModel());
  $("#btn-pull-dismiss")?.addEventListener("click", dismissPullProgress);
  $("#fit-filter")?.addEventListener("change", () => { if (!$("#fit-finder-body").textContent.includes("not been scanned")) void scanConfiguredFits(); });
  $("#pull-model")?.addEventListener("keydown", (event) => { if (event.key === "Enter") { event.preventDefault(); void pullModel(); } });
  $("#btn-import-local-gguf")?.addEventListener("click", () => void importLocalGguf());
  $("#local-gguf-path")?.addEventListener("keydown", (event) => { if (event.key === "Enter") { event.preventDefault(); void importLocalGguf(); } });
  $("#btn-import-url-gguf")?.addEventListener("click", () => void importGgufFromUrl());
  $("#import-gguf-url")?.addEventListener("keydown", (event) => { if (event.key === "Enter") { event.preventDefault(); void importGgufFromUrl(); } });
}

export async function refreshModels(options = {}) {
  const settings = { profiles: true, installed: true, runtime: true, ...options };
  const results = [];
  if (settings.profiles) results.push(await Promise.resolve(loadProfiles({ notify: false })).then((value) => ({ status: "fulfilled", value }), (reason) => ({ status: "rejected", reason })));
  const jobs = [];
  if (settings.runtime) jobs.push(refreshStatus({ notify: false }));
  if (settings.installed) jobs.push(refreshInstalled({ notify: false }));
  results.push(...await Promise.allSettled(jobs));
  notifyModelChanges("refresh");
  return { snapshot: getModelSnapshot(), results };
}
