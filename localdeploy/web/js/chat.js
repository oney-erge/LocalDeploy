"use strict";

import { $, $$, downloadFile, esc, fmtBytes, formatExpires, postJSON, postMaybeStream, toast } from "./shared.js?v=20261020-ui33";

const state = {
  profiles: [], profileData: {}, profileModels: {}, defaultProfile: null,
  installedLoaded: false, installedList: [],
  chatMessages: [], chatImages: [], chatFiles: [], chatController: null,
  chatSessionBusy: false, chatSessionOperation: null,
  // True once the user picks a model in the Chat picker; until then Chat follows the loaded model.
  chatModelChosen: false,
};
let initialized = false;
let onModelStateInvalidated = async () => {};
let onNavigate = () => {};

async function loadProfiles() { await onModelStateInvalidated(); }
async function refreshStatus() { await onModelStateInvalidated(); }
async function waitForModelToUnload(name, attempts = 6) {
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    await onModelStateInvalidated();
    if (!runningChatModel(name)) return true;
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  return false;
}
async function killRunningModel(name, button) {
  if (!name) return;
  state.chatSessionOperation = "unloading"; updateChatModelState();
  try {
    const result = await postJSON("/models/stop", { model: name });
    if (!result.success) throw new Error(result.error || result.message || "Could not unload model.");
    const confirmed = result.confirmed === true || result.status === "unloaded" || await waitForModelToUnload(name);
    if (!confirmed) throw new Error(`Ollama still reports ${name} as loaded. Retry unload or refresh status.`);
    await onModelStateInvalidated(); toast(`Unloaded ${name}.`, "success");
  } catch (error) { toast(`Unload failed: ${error.message}`, "error"); }
  finally { state.chatSessionOperation = null; if (button) button.classList.remove("loading"); updateChatModelState(); }
}

export function updateModelContext(snapshot = {}) {
  state.profiles = (snapshot.profiles || []).map((profile) => profile.name);
  state.profileData = Object.fromEntries((snapshot.profiles || []).map((profile) => [profile.name, structuredClone(profile)]));
  state.profileModels = Object.fromEntries((snapshot.profiles || []).map((profile) => [profile.name, profile.model_id || profile.name]));
  state.defaultProfile = snapshot.defaultProfile || null;
  state.installedLoaded = !!snapshot.installedLoaded;
  state.installedList = structuredClone(snapshot.installed || []);
  renderChatModelOptions();
}

function chatSelectedModel() {
  return $("#chat-model")?.value || "";
}

function profileForChatModel(model = chatSelectedModel(), enabledOnly = true) {
  if (!model) return "";
  return (
    state.profiles.find((name) => {
      const profile = state.profileData[name] || {};
      return (
        (profile.backend || "ollama") === "ollama" &&
        profile.model_id === model &&
        (!enabledOnly || profile.enabled === true)
      );
    }) || ""
  );
}

function chatSelectedProfile() {
  return profileForChatModel();
}

function runningChatModel(model = chatSelectedModel()) {
  if (!model) return null;
  return state.installedList.find((item) => item.name === model)?.running || null;
}

function chatModelIsReady() {
  return !!(runningChatModel() && chatSelectedProfile());
}

function chatProfileSupportsVision() {
  const profile = state.profileData[chatSelectedProfile()] || {};
  return !!profile.supports_vision;
}

function renderChatModelOptions() {
  const select = $("#chat-model");
  if (!select) return;
  const current = select.value;
  const models = state.installedList || [];
  if (!state.installedLoaded) {
    select.innerHTML = `<option value="">Checking installed models…</option>`;
    select.disabled = true;
    updateChatModelState();
    return;
  }
  if (!models.length) {
    select.innerHTML = `<option value="">No installed models</option>`;
    select.disabled = true;
    updateChatModelState();
    return;
  }
  const names = models.map((item) => item.name).filter(Boolean);
  const byName = Object.fromEntries(models.map((item) => [item.name, item]));
  const defaultModel = state.profileModels[state.defaultProfile];
  const currentIsLoaded = names.includes(current) && !!byName[current]?.is_loaded;
  const preferred =
    (names.includes(current) && (state.chatModelChosen || currentIsLoaded) && current) ||
    names.find((name) => byName[name]?.is_loaded) ||
    (names.includes(current) && current) ||
    (names.includes(defaultModel) && defaultModel) ||
    names[0];
  select.innerHTML = names
    .map((name) => {
      const loaded = byName[name]?.is_loaded ? " · loaded" : "";
      return `<option value="${esc(name)}"${name === preferred ? " selected" : ""}>${esc(name + loaded)}</option>`;
    })
    .join("");
  select.disabled = false;
  updateChatModelState();
}

function updateChatModelState() {
  const attach = $("#chat-attach-label");
  const hint = $("#chat-hint");
  const session = $("#chat-session-state");
  const action = $("#btn-chat-session");
  const duration = $("#chat-keep-alive");
  const input = $("#chat-input");
  const send = $("#btn-chat-send");
  const progress = $("#chat-session-progress");
  const progressLabel = $("#chat-session-progress-label");
  if (!attach || !hint || !session || !action || !duration || !input || !send) return;
  const model = chatSelectedModel();
  const running = runningChatModel(model);
  const profile = chatSelectedProfile();
  const ready = !!(model && running && profile);
  const vision = chatProfileSupportsVision();
  const operation = state.chatSessionOperation || (state.chatSessionBusy ? "loading" : null);
  const canAttach = ready && !operation;
  attach.classList.toggle("disabled", !canAttach);
  attach.title = canAttach
    ? vision
      ? "Attach text files or images to your next message"
      : "Attach text files; images require a vision-capable model"
    : "Load an installed model before attaching files";
  if ((!ready || !vision) && state.chatImages.length) {
    state.chatImages = [];
    renderChatAttachments();
  }
  input.disabled = !ready || !!operation;
  send.disabled = (!ready || !!operation) && !state.chatController;
  const loadedWithoutProfile = !!(model && running && !profile);
  input.placeholder = operation
    ? `${operation === "unloading" ? "Unloading" : "Loading"} ${model}…`
    : ready
      ? "Message this local model"
      : loadedWithoutProfile
        ? "Select Enable chat above to start chatting"
        : "Load an installed model to start chatting";
  duration.disabled = ready || !!operation;
  action.disabled = !model || !!operation || !!state.chatController;
  session.className = `chat-session-state${ready && !operation ? " ready" : operation ? " loading" : ""}`;
  progress?.classList.toggle("hidden", !operation);
  if (progressLabel && operation) progressLabel.textContent = `${operation === "unloading" ? "Unloading" : "Loading"} ${model}…`;
  if (!state.installedLoaded) {
    hint.textContent = "Checking models installed in Ollama…";
    session.lastElementChild.textContent = "Checking availability";
    action.textContent = "Load model";
  } else if (!model) {
    hint.textContent = "No local models are installed. Pull one from Setup & Deploy to use Chat.";
    session.lastElementChild.textContent = "Nothing installed";
    action.textContent = "Load model";
  } else if (operation) {
    const unloading = operation === "unloading";
    hint.textContent = `${unloading ? "Releasing" : "Loading"} ${model} ${unloading ? "from" : "into"} RAM and VRAM…`;
    session.lastElementChild.textContent = unloading ? "Unloading" : "Loading";
    action.textContent = unloading ? "Unloading…" : "Loading…";
  } else if (ready) {
    const placement = running.placement ? ` · ${running.placement}${running.gpu_percent != null ? ` ${running.gpu_percent}% GPU` : ""}` : "";
    hint.textContent = `${model} is ready · ${formatExpires(running.expires_at)}${placement}`;
    session.lastElementChild.textContent = "Ready";
    action.textContent = "Unload";
  } else if (running) {
    hint.textContent = `${model} is loaded, but its chat profile is not enabled yet.`;
    session.lastElementChild.textContent = "Needs chat setup";
    action.textContent = "Enable chat";
  } else {
    const durationLabel = duration.options[duration.selectedIndex]?.textContent || "60 minutes";
    hint.textContent = `${model} is installed on disk. Load it to keep it ready for ${durationLabel}.`;
    session.lastElementChild.textContent = "On disk";
    action.textContent = "Load model";
  }
  if (!state.chatMessages.length) renderChatWelcome();
}

async function toggleChatSession() {
  const model = chatSelectedModel();
  const btn = $("#btn-chat-session");
  if (!model || state.chatSessionBusy) return;
  if (chatModelIsReady()) {
    await killRunningModel(model, btn);
    return;
  }
  state.chatSessionBusy = true;
  state.chatSessionOperation = "loading";
  updateChatModelState();
  try {
    const profileResult = await postJSON("/profiles/upsert", {
      model_id: model,
      backend: "ollama",
      fields: { enabled: true },
    });
    if (!profileResult.success) throw new Error(profileResult.error || "Could not enable chat for this model.");
    await loadProfiles();
    const keepAlive = $("#chat-keep-alive")?.value || "60m";
    const result = await postJSON("/models/serve", {
      model,
      keep_alive: keepAlive,
      device: "auto",
    });
    if (!result.success) throw new Error(result.error || result.message || "Could not load the model.");
    await refreshStatus();
    toast(result.warning || `${model} is ready for chat.`, result.warning ? "info" : "success");
  } catch (err) {
    toast(`Could not start chat: ${err.message}`, "error");
  } finally {
    state.chatSessionBusy = false;
    state.chatSessionOperation = null;
    updateChatModelState();
  }
}

function chatImageUrl(image) {
  return typeof image === "string" ? image : image?.url || "";
}

function renderChatAttachments() {
  const slot = $("#chat-attachments");
  if (!slot) return;
  if (!state.chatImages.length && !state.chatFiles.length) {
    slot.classList.add("hidden");
    slot.innerHTML = "";
    return;
  }
  slot.classList.remove("hidden");
  slot.innerHTML = "";
  state.chatFiles.forEach((file, i) => {
    const chip = document.createElement("span");
    chip.className = "chat-file-chip";
    const label = document.createElement("span");
    label.textContent = `${file.name} · ${fmtBytes(file.size || file.content?.length || 0)}`;
    const x = document.createElement("button");
    x.type = "button";
    x.textContent = "×";
    x.title = "Remove file";
    x.addEventListener("click", () => {
      state.chatFiles.splice(i, 1);
      renderChatAttachments();
    });
    chip.append(label, x);
    slot.appendChild(chip);
  });
  state.chatImages.forEach((image, i) => {
    const chip = document.createElement("span");
    chip.className = "chat-attach-chip";
    const img = document.createElement("img");
    img.src = chatImageUrl(image);
    img.alt = typeof image === "object" && image.name ? image.name : `attachment ${i + 1}`;
    const x = document.createElement("button");
    x.type = "button";
    x.textContent = "×";
    x.title = "Remove image";
    x.addEventListener("click", () => {
      state.chatImages.splice(i, 1);
      renderChatAttachments();
    });
    chip.append(img, x);
    slot.appendChild(chip);
  });
}

function addChatImages(files) {
  const maxImageBytes = 10 * 1024 * 1024;
  const maxTextBytes = 200 * 1024;
  Array.from(files || []).forEach((file) => {
    if (file.type.startsWith("image/")) {
      if (!chatProfileSupportsVision()) {
        toast("The selected profile isn't marked vision-capable - attach text files instead.", "error");
        return;
      }
      if (file.size > maxImageBytes) {
        toast(`${file.name} is over 10 MB - skipped.`, "error");
        return;
      }
      const reader = new FileReader();
      reader.onload = () => {
        state.chatImages.push({ url: String(reader.result), name: file.name, size: file.size, type: file.type });
        renderChatAttachments();
      };
      reader.readAsDataURL(file);
      return;
    }
    // Anything else is treated as text and embedded into the message.
    if (file.size > maxTextBytes) {
      toast(`${file.name} is over 200 KB - too large to embed as text.`, "error");
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      state.chatFiles.push({ name: file.name, content: String(reader.result), size: file.size, type: file.type || "text/plain" });
      renderChatAttachments();
    };
    reader.readAsText(file);
  });
}

function parseChatJson(text) {
  let raw = String(text || "").trim();
  const fenced = raw.match(/^```(?:json)?\s*\n?([\s\S]*?)\n?```$/i);
  if (fenced) raw = fenced[1].trim();
  if (!raw || !["{", "["].includes(raw[0])) return null;
  try {
    const value = JSON.parse(raw);
    return { value, raw: JSON.stringify(value, null, 2) };
  } catch {
    return null;
  }
}

function appendJsonTreeValue(parent, value, key = null, depth = 0) {
  const isContainer = value !== null && typeof value === "object";
  if (!isContainer) {
    const row = document.createElement("div");
    row.className = "json-tree-row";
    if (key !== null) {
      const keyNode = document.createElement("span");
      keyNode.className = "json-key";
      keyNode.textContent = `${key}:`;
      row.appendChild(keyNode);
    }
    const scalar = document.createElement("span");
    scalar.className = `json-value json-${value === null ? "null" : typeof value}`;
    scalar.textContent = typeof value === "string" ? `"${value}"` : String(value);
    row.appendChild(scalar);
    parent.appendChild(row);
    return;
  }
  const entries = Array.isArray(value) ? value.map((item, index) => [index, item]) : Object.entries(value);
  const details = document.createElement("details");
  details.className = "json-tree-node";
  details.open = depth < 2;
  const summary = document.createElement("summary");
  const type = Array.isArray(value) ? "array" : "object";
  summary.textContent = `${key !== null ? `${key}: ` : ""}${type} · ${entries.length} item${entries.length === 1 ? "" : "s"}`;
  details.appendChild(summary);
  const children = document.createElement("div");
  children.className = "json-tree-children";
  entries.forEach(([childKey, child]) => appendJsonTreeValue(children, child, childKey, depth + 1));
  details.appendChild(children);
  parent.appendChild(details);
}

function renderChatJson(container, parsed) {
  const shell = document.createElement("div");
  shell.className = "chat-json";
  const toolbar = document.createElement("div");
  toolbar.className = "chat-json-toolbar";
  const treeBtn = document.createElement("button");
  treeBtn.type = "button";
  treeBtn.className = "btn compact active";
  treeBtn.textContent = "Tree";
  const rawBtn = document.createElement("button");
  rawBtn.type = "button";
  rawBtn.className = "btn compact";
  rawBtn.textContent = "Raw";
  const copyBtn = document.createElement("button");
  copyBtn.type = "button";
  copyBtn.className = "btn compact";
  copyBtn.textContent = "Copy";
  const downloadBtn = document.createElement("button");
  downloadBtn.type = "button";
  downloadBtn.className = "btn compact";
  downloadBtn.textContent = "Download JSON";
  toolbar.append(treeBtn, rawBtn, copyBtn, downloadBtn);
  const tree = document.createElement("div");
  tree.className = "json-tree";
  appendJsonTreeValue(tree, parsed.value);
  const raw = document.createElement("pre");
  raw.className = "json-raw hidden";
  const code = document.createElement("code");
  code.textContent = parsed.raw;
  raw.appendChild(code);
  const selectView = (showRaw) => {
    tree.classList.toggle("hidden", showRaw);
    raw.classList.toggle("hidden", !showRaw);
    treeBtn.classList.toggle("active", !showRaw);
    rawBtn.classList.toggle("active", showRaw);
  };
  treeBtn.addEventListener("click", () => selectView(false));
  rawBtn.addEventListener("click", () => selectView(true));
  copyBtn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(parsed.raw);
      copyBtn.textContent = "Copied";
      setTimeout(() => (copyBtn.textContent = "Copy"), 1200);
    } catch { /* clipboard unavailable */ }
  });
  downloadBtn.addEventListener("click", () => downloadFile("localdeploy-response.json", parsed.raw, "application/json"));
  shell.append(toolbar, tree, raw);
  container.appendChild(shell);
}

// Render reply text into a bubble body: JSON gets a tree/raw inspector and
// fenced code blocks become styled <pre><code> elements with a copy button.
// <pre><code> elements with a copy button; everything else stays textContent
// (never innerHTML), so model output can't inject markup.

export function renderChatText(container, text) {
  container.innerHTML = "";
  const parsedJson = parseChatJson(text);
  if (parsedJson) {
    renderChatJson(container, parsedJson);
    return;
  }
  const parts = String(text).split(/```([\w+-]*)\n?([\s\S]*?)(?:```|$)/g);
  // split() with two capture groups yields [text, lang, code, text, lang, code, …]
  for (let i = 0; i < parts.length; i += 3) {
    const plain = parts[i];
    if (plain) appendMarkdownLite(container, plain);
    if (i + 2 < parts.length) {
      const code = parts[i + 2] ?? "";
      const wrap = document.createElement("div");
      wrap.className = "chat-code";
      if (parts[i + 1]) {
        const lang = document.createElement("span");
        lang.className = "chat-code-lang";
        lang.textContent = parts[i + 1];
        wrap.appendChild(lang);
      }
      const copy = document.createElement("button");
      copy.type = "button";
      copy.className = "chat-code-copy";
      copy.textContent = "⧉";
      copy.title = "Copy code";
      copy.addEventListener("click", async () => {
        try {
          await navigator.clipboard.writeText(code);
          copy.textContent = "✓";
          setTimeout(() => (copy.textContent = "⧉"), 1200);
        } catch { /* clipboard unavailable */ }
      });
      wrap.appendChild(copy);
      const pre = document.createElement("pre");
      const codeEl = document.createElement("code");
      codeEl.textContent = code;
      pre.appendChild(codeEl);
      wrap.appendChild(pre);
      container.appendChild(wrap);
    }
  }
}

// Minimal, escape-first markdown for chat text: headings, bullets, inline
// code/bold/italic and http(s) links. Everything is built with createElement +
// textContent - model output can never inject markup.

const _INLINE_MD = /(`[^`\n]+`|\*\*[^*\n]+\*\*|\*[^*\n]+\*|\[[^\]\n]+\]\(https?:\/\/[^\s)]+\))/g;

function appendInlineMarkdown(parent, text) {
  let last = 0;
  for (const match of text.matchAll(_INLINE_MD)) {
    if (match.index > last) parent.append(text.slice(last, match.index));
    const token = match[0];
    if (token.startsWith("`")) {
      const code = document.createElement("code");
      code.textContent = token.slice(1, -1);
      parent.appendChild(code);
    } else if (token.startsWith("**")) {
      const b = document.createElement("b");
      b.textContent = token.slice(2, -2);
      parent.appendChild(b);
    } else if (token.startsWith("*")) {
      const i = document.createElement("i");
      i.textContent = token.slice(1, -1);
      parent.appendChild(i);
    } else {
      const m = token.match(/^\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)$/);
      const a = document.createElement("a");
      a.textContent = m ? m[1] : token;
      if (m) {
        a.href = m[2];
        a.target = "_blank";
        a.rel = "noopener";
      }
      parent.appendChild(a);
    }
    last = match.index + token.length;
  }
  if (last < text.length) parent.append(text.slice(last));
}

function appendMarkdownLite(container, text) {
  for (const line of String(text).split("\n")) {
    const el = document.createElement("div");
    const heading = line.match(/^(#{1,3})\s+(.*)$/);
    const bullet = line.match(/^\s*[-*]\s+(.*)$/);
    if (heading) {
      el.className = `md-h${heading[1].length}`;
      appendInlineMarkdown(el, heading[2]);
    } else if (bullet) {
      el.className = "md-li";
      el.append("• ");
      appendInlineMarkdown(el, bullet[1]);
    } else if (!line.trim()) {
      el.className = "md-gap";
    } else {
      appendInlineMarkdown(el, line);
    }
    container.appendChild(el);
  }
}

// Build one message bubble with a compact role marker.

function appendChatBubble(role, text, images = [], files = []) {
  const list = $("#chat-messages");
  const row = document.createElement("div");
  row.className = `chat-row ${role}`;
  const avatar = document.createElement("div");
  avatar.className = "chat-avatar";
  avatar.textContent = role === "user" ? "U" : "AI";
  row.appendChild(avatar);
  const bubble = document.createElement("div");
  bubble.className = "chat-bubble";
  if (images.length) {
    const strip = document.createElement("div");
    strip.className = "chat-bubble-images";
    images.forEach((image) => {
      const img = document.createElement("img");
      img.src = chatImageUrl(image);
      img.alt = typeof image === "object" && image.name ? image.name : "attached image";
      strip.appendChild(img);
    });
    bubble.appendChild(strip);
  }
  if (files.length) {
    const fileStrip = document.createElement("div");
    fileStrip.className = "chat-bubble-files";
    files.forEach((file) => {
      const item = document.createElement("span");
      item.className = "chat-bubble-file";
      item.textContent = `${file.name} · ${fmtBytes(file.size || file.content?.length || 0)}`;
      item.title = "This file's text was included in the model request.";
      fileStrip.appendChild(item);
    });
    bubble.appendChild(fileStrip);
  }
  const body = document.createElement("div");
  body.className = "chat-bubble-text";
  if (text) renderChatText(body, text);
  bubble.appendChild(body);
  const meta = document.createElement("div");
  meta.className = "chat-bubble-meta muted small";
  bubble.appendChild(meta);
  row.appendChild(bubble);
  list.appendChild(row);
  list.scrollTop = list.scrollHeight;
  return { row, body, meta, bubble };
}

function chatMessagesForApi(profile) {
  const messages = [];
  const system = $("#chat-system")?.value.trim();
  if (system) messages.push({ role: "system", content: system });
  state.chatMessages.forEach((m) => {
    if (m.images?.length) {
      messages.push({
        role: m.role,
        content: [
          { type: "text", text: m.apiText || m.text },
          ...m.images.map((image) => ({ type: "image_url", image_url: { url: chatImageUrl(image) } })),
        ],
      });
    } else {
      messages.push({ role: m.role, content: m.apiText || m.text });
    }
  });
  return {
    model: profile,
    messages,
    stream: true,
    // Without this, each generate call resets Ollama's keep-alive to its 5m
    // default, silently undoing the session's "keep loaded" choice.
    keep_alive: $("#chat-keep-alive")?.value || undefined,
  };
}

async function sendChatMessage() {
  const input = $("#chat-input");
  const btn = $("#btn-chat-send");
  if (state.chatController) {
    // Button doubles as Stop while a reply is streaming.
    state.chatController.abort();
    return;
  }
  const typed = input.value.trim();
  const profile = chatSelectedProfile();
  if ((!typed && !state.chatFiles.length && !state.chatImages.length) || !chatModelIsReady() || !profile) {
    if (!chatModelIsReady()) toast("Load an installed model before sending a message.", "error");
    return;
  }
  // Attached text files ride along as fenced blocks: the model gets clear file
  // boundaries and the bubble renders them as code blocks.
  const fileBlocks = state.chatFiles
    .map((f) => "\n\n[Attached file: " + f.name + "]\n```\n" + f.content + "\n```")
    .join("");
  const displayText = typed || "Please review the attached file(s).";
  const apiText = displayText + fileBlocks;
  const images = state.chatImages.slice();
  const files = state.chatFiles.map((file) => ({ ...file }));
  state.chatImages = [];
  state.chatFiles = [];
  renderChatAttachments();
  input.value = "";
  input.style.height = "auto";

  $("#chat-messages .chat-welcome")?.remove();
  state.chatMessages.push({ role: "user", text: displayText, apiText, images, files });
  appendChatBubble("user", displayText, images, files);
  const assistant = appendChatBubble("assistant", "", []);
  assistant.bubble.classList.add("typing");

  const controller = new AbortController();
  state.chatController = controller;
  updateChatModelState();
  btn.textContent = "Stop";
  btn.classList.add("danger");
  const started = performance.now();
  let firstTokenAt = null;
  let reply = "";
  let failed = null;
  try {
    await postMaybeStream(
      "/v1/chat/completions",
      chatMessagesForApi(profile),
      (evt) => {
        if (evt.error?.message) {
          failed = evt.error.message;
          return;
        }
        const delta = evt.choices?.[0]?.delta?.content;
        if (delta) {
          if (firstTokenAt === null) firstTokenAt = performance.now();
          reply += delta;
          assistant.bubble.classList.remove("typing");
          renderChatText(assistant.body, reply);
          const list = $("#chat-messages");
          list.scrollTop = list.scrollHeight;
        }
      },
      controller.signal
    );
  } catch (err) {
    if (err.name === "AbortError") failed = "Stopped.";
    else failed = err.message;
  } finally {
    state.chatController = null;
    btn.textContent = "Send";
    btn.classList.remove("danger");
    assistant.bubble.classList.remove("typing");
    updateChatModelState();
  }

  const elapsed = (performance.now() - started) / 1000;
  if (failed && !reply) {
    assistant.bubble.classList.add("error");
    assistant.body.textContent = failed;
    // Drop the failed exchange so a retry doesn't resend a broken turn.
    state.chatMessages.pop();
    return;
  }
  state.chatMessages.push({ role: "assistant", text: reply, images: [] });
  // tok/s measured from the first streamed token, so model-load time (which
  // dominates a cold first message) doesn't poison the generation speed.
  const approxTokens = Math.max(1, Math.ceil(reply.length / 4));
  const genSeconds = firstTokenAt !== null ? (performance.now() - firstTokenAt) / 1000 : elapsed;
  const tps = genSeconds > 0.2 && approxTokens > 2 ? ` · ⚡ ~${(approxTokens / genSeconds).toFixed(1)} tok/s` : "";
  const firstTok = firstTokenAt !== null && (firstTokenAt - started) / 1000 > 2
    ? ` · first token ${((firstTokenAt - started) / 1000).toFixed(1)} s`
    : "";
  assistant.meta.textContent = `🕒 ${elapsed.toFixed(1)} s${firstTok}${tps}${failed ? ` · ${failed}` : ""}`;
}

const CHAT_SUGGESTIONS = [
  "Explain what a quantized model is, in two sentences.",
  "Write a haiku about running AI locally.",
  "Return a JSON object with three fields describing this machine's ideal use.",
];

function clearChat() {
  state.chatController?.abort();
  state.chatMessages = [];
  state.chatImages = [];
  state.chatFiles = [];
  renderChatAttachments();
  renderChatWelcome();
}

function renderChatWelcome() {
  const list = $("#chat-messages");
  if (!list || state.chatMessages.length) return;
  list.innerHTML = "";
  const welcome = document.createElement("div");
  welcome.className = "chat-welcome";
  const ready = chatModelIsReady();
  const model = chatSelectedModel();
  if (!state.installedLoaded) {
    welcome.innerHTML = `<div class="chat-welcome-mark">…</div>
      <h3>Checking local models</h3>
      <p class="muted small">Chat will show models that are actually installed in Ollama.</p>`;
    list.appendChild(welcome);
    return;
  }
  if (!model) {
    welcome.innerHTML = `<div class="chat-welcome-mark">+</div>
      <h3>No installed models</h3>
      <p class="muted small">Pull a model from Setup &amp; Deploy, then return here to load it.</p>`;
    const openCatalog = document.createElement("button");
    openCatalog.type = "button";
    openCatalog.className = "btn primary";
    openCatalog.textContent = "Open model catalog";
    openCatalog.addEventListener("click", () => onNavigate("serve"));
    welcome.appendChild(openCatalog);
    list.appendChild(welcome);
    return;
  }
  if (!ready) {
    const loaded = !!runningChatModel(model);
    welcome.innerHTML = `<div class="chat-welcome-mark">${esc(model.slice(0, 1).toUpperCase())}</div>
      <h3>${esc(model)} is ${loaded ? "loaded" : "installed"}</h3>
      <p class="muted small">${loaded
        ? "Select Enable chat above to start a conversation with it."
        : "Use Load model above to warm it for the displayed keep-loaded duration."}</p>`;
    list.appendChild(welcome);
    return;
  }
  welcome.innerHTML = `<div class="chat-welcome-mark">${esc(model.slice(0, 1).toUpperCase())}</div>
    <h3>${esc(model)} is ready</h3>
    <p class="muted small">Start with a prompt or choose one below.</p>`;
  const chips = document.createElement("div");
  chips.className = "chat-suggestion-chips";
  CHAT_SUGGESTIONS.forEach((text) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chat-suggestion";
    chip.textContent = text;
    chip.addEventListener("click", () => {
      $("#chat-input").value = text;
      sendChatMessage();
    });
    chips.appendChild(chip);
  });
  welcome.appendChild(chips);
  list.appendChild(welcome);
}

// ---------------------------------------------------------------------------
// Tab 2 - Question set: example / upload / validate
// ---------------------------------------------------------------------------

export { clearChat };

export function initChat(options = {}) {
  if (initialized) return;
  initialized = true;
  onModelStateInvalidated = options.onModelStateInvalidated || onModelStateInvalidated;
  onNavigate = options.onNavigate || onNavigate;
  $("#btn-chat-send")?.addEventListener("click", sendChatMessage);
  $("#btn-chat-clear")?.addEventListener("click", clearChat);
  $("#chat-model")?.addEventListener("change", () => { state.chatModelChosen = true; updateChatModelState(); });
  $("#chat-keep-alive")?.addEventListener("change", updateChatModelState);
  $("#btn-chat-session")?.addEventListener("click", toggleChatSession);
  $("#btn-chat-system-toggle")?.addEventListener("click", () => { const panel = $("#chat-system-panel"); panel.classList.toggle("hidden"); if (!panel.classList.contains("hidden")) $("#chat-system").focus(); });
  $("#chat-images")?.addEventListener("change", (event) => { if (!chatModelIsReady()) toast("Load an installed model before attaching files.", "error"); else addChatImages(event.target.files); event.target.value = ""; });
  const composer = $(".chat-composer");
  composer?.addEventListener("dragover", (event) => { if (!chatModelIsReady()) return; event.preventDefault(); composer.classList.add("dragging"); });
  composer?.addEventListener("dragleave", (event) => { if (!composer.contains(event.relatedTarget)) composer.classList.remove("dragging"); });
  composer?.addEventListener("drop", (event) => { event.preventDefault(); composer.classList.remove("dragging"); if (!chatModelIsReady()) { toast("Load an installed model before dropping files here.", "error"); return; } addChatImages(event.dataTransfer?.files); });
  $("#chat-input")?.addEventListener("input", (event) => { const element = event.target; element.style.height = "auto"; element.style.height = Math.min(element.scrollHeight, 180) + "px"; });
  $("#chat-input")?.addEventListener("keydown", (event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); if (!state.chatController) void sendChatMessage(); } });
  clearChat();
}
