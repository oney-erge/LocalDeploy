"use strict";

export const $ = (selector, root = document) => root.querySelector(selector);
export const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

export function initTooltips() {
  const tooltip = $("#ui-tooltip");
  if (!tooltip) return;

  let activeTrigger = null;
  let pinned = false;
  let positionFrame = null;

  const triggerFor = (node) => (node instanceof Element ? node.closest("[data-tooltip]") : null);

  const positionTooltip = () => {
    positionFrame = null;
    if (activeTrigger && !activeTrigger.isConnected) {
      hideTooltip();
      return;
    }
    if (!activeTrigger || !tooltip.classList.contains("is-visible")) return;

    const gap = 8;
    const edge = 10;
    const triggerRect = activeTrigger.getBoundingClientRect();
    const tooltipRect = tooltip.getBoundingClientRect();
    let top = triggerRect.top - tooltipRect.height - gap;
    if (top < edge) top = triggerRect.bottom + gap;
    if (top + tooltipRect.height > window.innerHeight - edge) {
      top = Math.max(edge, triggerRect.top - tooltipRect.height - gap);
    }
    const centered = triggerRect.left + triggerRect.width / 2 - tooltipRect.width / 2;
    const left = Math.max(edge, Math.min(centered, window.innerWidth - tooltipRect.width - edge));
    tooltip.style.left = `${Math.round(left)}px`;
    tooltip.style.top = `${Math.round(top)}px`;
  };

  const schedulePosition = () => {
    if (positionFrame == null) positionFrame = requestAnimationFrame(positionTooltip);
  };

  const hideTooltip = () => {
    if (positionFrame != null) cancelAnimationFrame(positionFrame);
    positionFrame = null;
    activeTrigger?.classList.remove("tooltip-open");
    if (activeTrigger?.hasAttribute("aria-expanded")) activeTrigger.setAttribute("aria-expanded", "false");
    activeTrigger = null;
    pinned = false;
    tooltip.classList.remove("is-visible");
    tooltip.setAttribute("aria-hidden", "true");
  };

  const showTooltip = (trigger, shouldPin = false) => {
    const text = trigger?.dataset.tooltip?.trim();
    if (!text) return;
    const keepPinned = activeTrigger === trigger && pinned;
    if (activeTrigger && activeTrigger !== trigger) {
      activeTrigger.classList.remove("tooltip-open");
      if (activeTrigger.hasAttribute("aria-expanded")) activeTrigger.setAttribute("aria-expanded", "false");
    }
    activeTrigger = trigger;
    pinned = shouldPin || keepPinned;
    tooltip.textContent = text;
    tooltip.classList.add("is-visible");
    tooltip.setAttribute("aria-hidden", "false");
    trigger.classList.add("tooltip-open");
    if (trigger.hasAttribute("aria-expanded")) trigger.setAttribute("aria-expanded", "true");
    positionTooltip();
  };

  document.addEventListener("pointerover", (event) => {
    const trigger = triggerFor(event.target);
    if (!trigger || trigger.contains(event.relatedTarget)) return;
    showTooltip(trigger);
  });
  document.addEventListener("pointerout", (event) => {
    const trigger = triggerFor(event.target);
    if (!trigger || trigger !== activeTrigger || trigger.contains(event.relatedTarget) || pinned || trigger.matches(":focus")) return;
    hideTooltip();
  });
  document.addEventListener("focusin", (event) => {
    const trigger = triggerFor(event.target);
    if (trigger) showTooltip(trigger);
  });
  document.addEventListener("focusout", (event) => {
    const trigger = triggerFor(event.target);
    // event.target usually isn't a tooltip trigger at all (e.g. any other
    // button losing focus) - trigger is null then, and null === activeTrigger
    // is true exactly when no tooltip is open, so this must bail out before
    // touching trigger.matches(...) instead of relying on the equality check.
    if (!trigger || trigger !== activeTrigger || pinned || trigger.matches(":hover")) return;
    hideTooltip();
  });
  document.addEventListener("click", (event) => {
    const trigger = triggerFor(event.target);
    if (!trigger) {
      if (pinned) hideTooltip();
      return;
    }
    if (trigger.classList.contains("help-tip")) {
      event.preventDefault();
      event.stopPropagation();
    }
    if (trigger === activeTrigger && pinned) hideTooltip();
    else showTooltip(trigger, true);
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && activeTrigger) {
      hideTooltip();
      return;
    }
    const trigger = triggerFor(event.target);
    if (!trigger || (event.key !== "Enter" && event.key !== " ")) return;
    event.preventDefault();
    if (trigger === activeTrigger && pinned) hideTooltip();
    else showTooltip(trigger, true);
  });
  window.addEventListener("resize", schedulePosition);
  window.addEventListener("scroll", schedulePosition, true);
  new MutationObserver(() => {
    if (activeTrigger && !activeTrigger.isConnected) hideTooltip();
  }).observe(document.body, { childList: true, subtree: true });
}

export function downloadFile(filename, content, mime) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// Extract a report card's JSON from an uploaded file (raw JSON or embedded HTML).

export function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

// Shimmering placeholder rows shown while a card's first fetch is in flight,
// so a slow local backend reads as "loading" instead of "frozen".

export function skeletonHtml(lines = 2) {
  return `<div class="skeleton-block">${Array.from({ length: lines }, () => `<div class="skeleton-line"></div>`).join("")}</div>`;
}

function dismissToast(node) {
  if (!node || node.classList.contains("toast-hide")) return;
  node.classList.add("toast-hide");
  node.addEventListener("transitionend", () => node.remove(), { once: true });
  // Fallback in case the transitionend event doesn't fire (e.g. reduced-motion).
  setTimeout(() => node.remove(), 400);
}

export function toast(message, kind = "info") {
  const node = document.createElement("div");
  node.className = `toast ${kind}`;
  node.textContent = message;
  // Errors are announced assertively and get more time on screen, but every
  // toast eventually fades on its own - a pile of undismissed error toasts
  // is worse than losing one you didn't read in time. Click still dismisses
  // early either way.
  const timeoutMs = kind === "error" ? 9000 : 5000;
  if (kind === "error") node.setAttribute("role", "alert");
  node.title = "Click to dismiss";
  node.addEventListener("click", () => dismissToast(node));
  setTimeout(() => dismissToast(node), timeoutMs);
  $("#toasts").appendChild(node);
}

// Tick a live "…Ns" counter into an element so long operations clearly show
// progress instead of looking frozen. Returns a stop() function.

export function startElapsed(el, label = "working") {
  if (!el) return () => {};
  const t0 = Date.now();
  const render = () => {
    const s = Math.round((Date.now() - t0) / 1000);
    el.innerHTML = `<span class="spin-inline"></span> ${esc(label)}… <b>${s}s</b>`;
  };
  render();
  const id = setInterval(render, 1000);
  return () => clearInterval(id);
}

// ---- optional API token (opt-in; nothing happens unless the server sets one) -

export function getToken() {
  try {
    return localStorage.getItem("localdeploy_token") || "";
  } catch {
    return "";
  }
}

export function setToken(t) {
  try {
    localStorage.setItem("localdeploy_token", t);
  } catch {
    /* ignore */
  }
}
// A `?token=…` in the URL is stored once, then stripped from the bar.
function bootstrapToken() {
  const params = new URLSearchParams(window.location.search);
  const t = params.get("token");
  if (t) {
    setToken(t);
    params.delete("token");
    const q = params.toString();
    history.replaceState(null, "", window.location.pathname + (q ? `?${q}` : ""));
  }
}

export function authHeaders() {
  const t = getToken();
  return t ? { "X-API-Token": t } : {};
}

// ---- light/dark theme (persisted; defaults to OS preference) ----------------

const THEME_KEY = "localdeploy_theme";

export function currentTheme() {
  const explicit = document.documentElement.dataset.theme;
  if (explicit === "light" || explicit === "dark") return explicit;
  return window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

export function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const btn = $("#btn-theme");
  if (btn) {
    btn.textContent = theme === "light" ? "☀" : "☾";
    btn.title = theme === "light" ? "Switch to dark theme" : "Switch to light theme";
  }
}

export function toggleTheme() {
  const next = currentTheme() === "light" ? "dark" : "light";
  try {
    localStorage.setItem(THEME_KEY, next);
  } catch {
    /* theme still applies for this session */
  }
  applyTheme(next);
}
// If the server rejects us, prompt for the token once and let the user retry.

function handle401(resp) {
  if (resp && resp.status === 401) {
    const t = window.prompt("This server requires an API token. Enter it:");
    if (t) {
      setToken(t.trim());
      toast("Token saved - retry your action.", "success");
    }
    return true;
  }
  return false;
}

// Parse a Response, throwing a useful message on any non-OK status. FastAPI
// error bodies are JSON ({"detail": ...}), so we surface that text instead of
// letting an error body masquerade as a successful payload.

async function parseOrThrow(url, resp) {
  if (resp.ok) return resp.json();
  let detail = `HTTP ${resp.status}`;
  try {
    const ct = resp.headers.get("content-type") || "";
    if (ct.includes("application/json")) {
      const body = await resp.json();
      detail = body.detail || body.error || body.message || JSON.stringify(body);
    } else {
      const text = await resp.text();
      if (text) detail = text.slice(0, 300);
    }
  } catch {
    /* keep the status-based message */
  }
  throw new Error(detail);
}

export async function getJSON(url, { isCurrent } = {}) {
  const resp = await fetch(url, { headers: authHeaders() });
  // A superseded read should not open an authentication prompt or publish
  // an error after a newer request has already taken ownership of the view.
  if (isCurrent && !isCurrent()) return null;
  if (handle401(resp)) throw new Error("unauthorized");
  return parseOrThrow(url, resp);
}

export async function postJSON(url, body) {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(body ?? {}),
  });
  if (handle401(resp)) throw new Error("unauthorized");
  return parseOrThrow(url, resp);
}

// Read a fetch Response as Server-Sent Events; calls onEvent(obj) per `data:`
// line and resolves when the stream ends or a [DONE] marker arrives.

export async function streamSSE(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  const processBlock = (block) => {
    for (const line of block.split("\n")) {
      if (!line.startsWith("data: ")) continue;
      const payload = line.slice(6);
      if (payload === "[DONE]") return true;
      try {
        onEvent(JSON.parse(payload));
      } catch {
        /* ignore non-JSON keepalives */
      }
    }
    return false;
  };
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const block = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      if (processBlock(block)) return;
    }
  }
  // Flush any trailing event that wasn't terminated by a blank line so a final
  // run_end/summary isn't dropped (which would leave the run looking hung).
  if (buf.trim()) processBlock(buf);
}

// POST that may return JSON (e.g. a blocked action) or an SSE stream.
// `signal` (optional) lets the caller abort an in-flight stream (cancel a pull).

export async function postMaybeStream(url, body, onEvent, signal) {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(body ?? {}),
    signal,
  });
  if (handle401(resp)) throw new Error("unauthorized");
  const ct = resp.headers.get("content-type") || "";
  if (ct.includes("text/event-stream")) {
    await streamSSE(resp, onEvent);
    return { streamed: true };
  }
  return { streamed: false, json: await resp.json() };
}

export function busy(button, on) {
  if (!button) return;
  button.disabled = on;
  button.classList.toggle("loading", on);
}

export function fmtMb(mb) {
  if (mb == null) return "?";
  return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb} MB`;
}

export function pct(value, total) {
  if (!value || !total) return 0;
  return Math.max(0, Math.min(100, (value / total) * 100));
}

export function vramBarHtml(usedMb, totalMb, label) {
  if (!usedMb || !totalMb) return "";
  const width = pct(usedMb, totalMb);
  const cls = width > 92 ? "danger" : width > 78 ? "warn" : "ok";
  return `<div class="mini-meter" title="${esc(label || "")}">
    <div class="mini-meter-fill ${cls}" style="width:${width.toFixed(1)}%"></div>
  </div>`;
}

export function formatExpires(expiresAt) {
  if (!expiresAt) return "No keep-alive expiry reported";
  const normalized = String(expiresAt).replace(/\.(\d{3})\d+/, ".$1");
  const t = new Date(normalized).getTime();
  if (!Number.isFinite(t)) return `Expires ${expiresAt}`;
  const delta = Math.round((t - Date.now()) / 1000);
  if (delta <= 0) return "Expiry reached; unload may be pending";
  if (delta < 90) return `Expires in ${delta}s`;
  const mins = Math.round(delta / 60);
  if (mins < 90) return `Expires in ${mins}m`;
  return `Expires in ${(mins / 60).toFixed(1)}h`;
}

export function fmtBytes(n) {
  if (n == null || !Number.isFinite(n)) return "";
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(0)} MB`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)} KB`;
  return `${n} B`;
}

export function fmtDuration(seconds) {
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${s % 60}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

// Update the pull-progress panel. `percent === null` renders an indeterminate bar
// (used while Ollama is doing sizeless work like "pulling manifest"/"verifying").

export function simpleModal(title, subtitle, bodyHtml) {
  const overlay = document.createElement("div");
  overlay.className = "modal-overlay";
  overlay.innerHTML = `<div class="modal-card" role="dialog" aria-modal="true">
      <div class="card-head">
        <div><h3 class="sub">${esc(title)}</h3>${subtitle ? `<div class="muted small">${subtitle}</div>` : ""}</div>
        <button class="btn compact modal-close">✕</button>
      </div>
      ${bodyHtml}
    </div>`;
  document.body.appendChild(overlay);
  const close = () => overlay.remove();
  overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
  $$(".modal-close", overlay).forEach((b) => b.addEventListener("click", close));
  return overlay;
}

let sharedInitialized = false;

export function initSharedUI() {
  if (sharedInitialized) return;
  sharedInitialized = true;
  bootstrapToken();
  applyTheme(currentTheme());
  $("#btn-theme")?.addEventListener("click", toggleTheme);
  initTooltips();
  $$(".btn.file").forEach((label) => {
    label.setAttribute("tabindex", "0");
    label.setAttribute("role", "button");
    label.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      label.querySelector('input[type="file"]')?.click();
    });
  });
}
