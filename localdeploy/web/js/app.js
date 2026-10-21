"use strict";

import { $, $$, initSharedUI, toast } from "./shared.js?v=20261020-ui33";
import { getSystemSnapshot, initSystem, refreshOllamaStatus, refreshSystem, setMonitorActive, subscribeToSystemChanges, updateModelContext as updateSystemModelContext } from "./system.js?v=20261020-ui33";
import { getModelSnapshot, initModels, refreshModels, setDefaultProfile, subscribeToModelChanges, updateSystemContext as updateModelsSystemContext } from "./models.js?v=20261020-ui33";
import { initChat, updateModelContext as updateChatModelContext } from "./chat.js?v=20261020-ui33";
import { initBenchmark, refreshBenchmarkMetadata, updateModelContext as updateBenchmarkModelContext, updateSystemContext as updateBenchmarkSystemContext } from "./benchmark.js?v=20261020-ui33";

function propagateSystem(snapshot = getSystemSnapshot()) {
  updateModelsSystemContext(snapshot);
  updateBenchmarkSystemContext(snapshot);
}

function propagateModels(snapshot = getModelSnapshot()) {
  updateChatModelContext(snapshot);
  updateBenchmarkModelContext(snapshot);
  updateSystemModelContext({ installedCount: snapshot.installed.length, running: snapshot.running });
}

async function invalidateModelState() {
  await Promise.allSettled([refreshModels(), refreshOllamaStatus()]);
  propagateSystem(); propagateModels();
}

function initializeFeature(label, initializer) {
  try {
    initializer();
    return { status: "fulfilled", label };
  } catch (error) {
    console.error(`LocalDeploy ${label} initialization failed`, error);
    toast(`${label} failed to initialize: ${error.message}`, "error");
    return { status: "rejected", label, reason: error };
  }
}

function activateTab(name) {
  $$(".tab").forEach((tab) => { const active = tab.dataset.tab === name; tab.classList.toggle("active", active); tab.setAttribute("aria-selected", active ? "true" : "false"); });
  $$(".tab-panel").forEach((panel) => panel.classList.toggle("active", panel.id === `tab-${name}`));
  setMonitorActive(name === "monitor");
  if (name === "serve") void invalidateModelState();
  else if (name === "bench") void refreshModels({ profiles: false, installed: false });
  else if (name === "chat") void invalidateModelState().then(() => $("#chat-input")?.focus());
}

async function bootstrap() {
  initializeFeature("shared UI", initSharedUI);
  subscribeToSystemChanges(propagateSystem);
  subscribeToModelChanges((snapshot, meta) => { propagateModels(snapshot); if (meta?.requiresSystemRefresh) void refreshSystem(); });
  initializeFeature("system", () => initSystem({ onNavigate: activateTab, onModelStateInvalidated: invalidateModelState }));
  initializeFeature("models", initModels);
  initializeFeature("chat", () => initChat({ onModelStateInvalidated: invalidateModelState, onNavigate: activateTab }));
  initializeFeature("benchmark", () => initBenchmark({ onModelStateInvalidated: invalidateModelState, setDefaultProfile }));
  $$(".tab").forEach((tab) => tab.addEventListener("click", () => activateTab(tab.dataset.tab)));
  $("#brand-home")?.addEventListener("click", () => activateTab("serve"));
  const results = await Promise.allSettled([refreshSystem(), refreshModels(), refreshBenchmarkMetadata()]);
  propagateSystem(); propagateModels();
  results.filter((result) => result.status === "rejected").forEach((result) => console.error("LocalDeploy startup task failed", result.reason));
}

bootstrap().catch((error) => {
  console.error("LocalDeploy startup failed", error);
  toast(`LocalDeploy startup failed: ${error.message}`, "error");
});
