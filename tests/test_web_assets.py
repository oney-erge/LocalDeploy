"""Static guards for the native ES-module frontend."""
from __future__ import annotations

import shutil
import subprocess
import re
from pathlib import Path

import pytest

from tests.node_flags import node_module_flags

WEB_DIR = Path(__file__).resolve().parent.parent / "localdeploy" / "web"
JS_DIR = WEB_DIR / "js"
MODULE_NAMES = {
    "app.js",
    "shared.js",
    "system.js",
    "models.js",
    "chat.js",
    "benchmark.js",
    "benchmark-views.js",
}


def js_text(*names: str) -> str:
    selected = names or tuple(sorted(MODULE_NAMES))
    return "\n".join((JS_DIR / name).read_text(encoding="utf-8") for name in selected)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_web_modules_parse_as_esm() -> None:
    for module in sorted(JS_DIR.glob("*.js")):
        result = subprocess.run(
            ["node", "--input-type=module", "--check"],
            input=module.read_text(encoding="utf-8"),
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        assert result.returncode == 0, f"{module.name} failed to parse:\n{result.stderr}"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_feature_modules_have_no_import_time_browser_side_effects() -> None:
    for name in sorted(MODULE_NAMES - {"app.js"}):
        module_url = (JS_DIR / name).as_uri()
        result = subprocess.run(
            [
                "node",
                *node_module_flags(),
                "-e",
                f"import({module_url!r}).catch(error => {{ console.error(error); process.exit(1); }})",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        assert result.returncode == 0, f"{name} executed browser work during import:\n{result.stderr}"


def test_module_inventory_and_dependency_graph() -> None:
    assert {path.name for path in JS_DIR.glob("*.js")} == MODULE_NAMES
    allowed = {
        "app.js": {"shared.js", "system.js", "models.js", "chat.js", "benchmark.js"},
        "shared.js": set(),
        "system.js": {"shared.js"},
        "models.js": {"shared.js"},
        "chat.js": {"shared.js"},
        "benchmark.js": {"shared.js", "benchmark-views.js"},
        "benchmark-views.js": {"shared.js"},
    }
    graph: dict[str, set[str]] = {}
    for name in MODULE_NAMES:
        source = (JS_DIR / name).read_text(encoding="utf-8")
        imports = set(re.findall(r'from "\./([^"?]+\.js)(?:\?v=[\w-]+)?"', source))
        assert imports <= allowed[name], f"disallowed imports in {name}: {sorted(imports - allowed[name])}"
        assert all((JS_DIR / target).is_file() for target in imports)
        graph[name] = imports

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        assert name not in visiting, f"circular frontend import through {name}"
        if name in visited:
            return
        visiting.add(name)
        for dependency in graph[name]:
            visit(dependency)
        visiting.remove(name)
        visited.add(name)

    for name in MODULE_NAMES:
        visit(name)


def test_models_module_has_no_transitional_cross_feature_adapters() -> None:
    source = js_text("models.js")
    for legacy_name in (
        "renderChatModelOptions",
        "renderBenchmarkProfileChips",
        "updateBenchmarkSummary",
        "updateChatModelState",
        "checkHardware",
    ):
        assert legacy_name not in source


def test_model_matching_and_placement_are_owned_by_models_module() -> None:
    consumers = js_text("chat.js", "benchmark.js")
    assert "ollamaModelNamesMatch" not in consumers
    assert "runningPlacements" not in consumers
    models_source = js_text("models.js")
    assert "is_loaded" in models_source
    assert "running_model" in models_source
    assert "placement" in models_source


def test_no_smart_quotes_as_js_delimiters() -> None:
    """Catch curly quotes used as code (string/attribute delimiters) even when
    Node is unavailable. Display-text curly quotes inside straight-quoted
    strings are fine; these patterns are the fatal ones."""
    text = js_text()
    # Common fatal shapes: $(“  ”)  || “”  join(“  class=”
    offenders = [m for m in ['$(“', '”)', 'class=”', 'join(“', 'split(“'] if m in text]
    assert not offenders, f"smart quotes used as JS delimiters: {offenders}"


def test_ui_assets_are_cache_busted_and_no_favicon_404() -> None:
    """Every static asset must carry one shared ?v= token (a stale-cache guard),
    without pinning the token's value so a version bump doesn't break the test."""
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    sources = html + "\n" + js_text()
    versions = set(re.findall(r'(?:\.css|\.js|\.png|\.svg)\?v=([\w-]+)', sources))
    assert len(versions) == 1, f"expected one shared cache-bust token, found: {sorted(versions)}"
    assert '<script type="module" src="js/app.js?v=' in html
    assert 'rel="icon" type="image/png" href="favicon.png?v=' in html
    assert (WEB_DIR / "favicon.png").is_file()
    assert (WEB_DIR / "logo.svg").is_file()


def test_benchmark_workspace_v2_labels_are_present() -> None:
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    for label in ("Benchmark runner", "Leaderboard", "Category heatmap", "Compare selected"):
        assert label in html


def test_new_ui_controls_have_safe_bindings() -> None:
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    models_js = js_text("models.js")
    chat_js = js_text("chat.js")
    benchmark_js = js_text("benchmark.js")
    system_js = js_text("system.js")
    for dom_id in (
        "btn-hf-search", "btn-fit-profiles", "fit-filter", "catalog-source-status", "vram-budget-gb",
        # chat playground
        "chat-model", "chat-keep-alive", "btn-chat-session", "chat-session-state",
        "btn-chat-send", "btn-chat-clear", "chat-input", "chat-images",
        # quant advisor
        "quant-model", "btn-quant-advise", "quant-body",
        # disk usage / bulk delete
        "installed-sort", "models-disk-summary", "btn-bulk-delete", "btn-bulk-clear",
        # server-side history + orphan cleanup
        "bench-history-server", "btn-clean-orphans",
        # provider catalog + repeated benchmark variance
        "btn-provider-refresh", "provider-catalog-body", "bench-repetitions",
        # remote discovery catalog lifecycle
        "remote-source-filter", "remote-size-filter", "remote-fit-filter", "remote-cap-filter",
        "remote-sort", "remote-installed-filter", "btn-remote-clear", "remote-active-filters",
        "btn-pull-dismiss", "pull-progress-actions", "chat-session-progress",
        "local-gguf-path", "btn-import-local-gguf", "import-gguf-url", "import-gguf-name",
        "btn-import-url-gguf",
    ):
        assert f'id="{dom_id}"' in html, dom_id
    assert "btn-hf-search" in models_js and "searchUnifiedModels" in models_js
    assert "btn-fit-profiles" in models_js and "scanConfiguredFits" in models_js
    assert "sendChatMessage" in chat_js
    for binding in ("quantAdvise", "bulkDeleteSelected"):
        assert binding in models_js, binding
    for binding in ("importLocalGguf", "importGgufFromUrl", "modelscope"):
        assert binding in models_js + html, binding
    assert "initServerHistoryToggle" in benchmark_js
    assert "vram-budget-gb" in system_js
    assert '["#profile-select", "#bench-profile-select"]' in models_js
    assert "#chat-profile" not in js_text()


def test_chat_tab_present() -> None:
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    assert 'data-tab="chat"' in html
    assert 'id="tab-chat"' in html


def test_api_docs_quant_help_and_model_lifecycle_controls_are_present() -> None:
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    models_js = js_text("models.js")
    chat_js = js_text("chat.js")
    shared_js = js_text("shared.js")
    css = (WEB_DIR / "styles.css").read_text(encoding="utf-8")

    assert "apiDocsUrl" in models_js
    assert "${window.location.origin}/docs" in models_js
    assert 'class="btn compact api-docs-link"' in models_js
    assert "QUANT_EXPLANATIONS" in models_js
    assert "4-bit K-quant, medium variant" in models_js
    assert html.count('class="help-tip"') >= 4
    assert 'id="ui-tooltip"' in html
    assert "initTooltips" in shared_js
    assert ".ui-tooltip.is-visible" in css
    assert ".help-tip::after" not in css
    assert ".quant-label" in css
    assert "unload-installed-btn" in models_js
    assert 'postJSON("/models/stop"' in models_js
    assert "waitForModelToUnload" in models_js
    assert "expandRemoteCatalog" in models_js
    assert "renderChatJson" in chat_js
    assert "Download JSON" in chat_js


def test_system_card_holds_hardware_and_fit_budget() -> None:
    """New IA: hardware, live VRAM, and the model fit budget all live together in
    the top 'System' card; deploy options moved to 'Your models'."""
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    system = html.split("1. SYSTEM", 1)[1].split("2. GET A MODEL", 1)[0]
    assert "Live VRAM" in system
    assert "Model fit budget" in system
    assert "Custom GB" in system
    # Deploy options live with Your models, not a top-of-page deploy card.
    assert 'id="keep-alive" value="60m"' in html
    assert 'id="btn-stop"' not in html
    assert "<h2>Deploy a profile</h2>" not in html  # old top card is gone


def test_profiles_and_benchmark_copy_are_clear() -> None:
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    # Profiles now live in the Advanced zone as "All run profiles".
    assert "All run profiles" in html
    assert "<code>config.json</code> recipes" in html
    assert "Scan saved profiles" in html
    assert "Use LocalDeploy test bench" in html
    assert "Load JSON sample" in html
    assert "Leave the editor empty to run the built-in LocalDeploy test bench" in html


def test_model_discovery_readability_and_tune_progress_styles_exist() -> None:
    css = (WEB_DIR / "styles.css").read_text(encoding="utf-8")
    assert ".mrow a.name" in css
    assert "#c7ddff" in css
    assert ".discover-section .mrow" in css
    assert ".run-progress-fill.indeterminate" in css
    assert ".tune-steps" in css


def test_ui_does_not_guess_unattributed_vram_sources() -> None:
    js = js_text("models.js", "system.js")
    assert "Windows, the display" not in js
    assert "browser/GPU apps" not in js
    assert "cannot attribute this VRAM to exact processes" not in js
    assert "LocalDeploy cannot attribute" not in js
    assert "Ollama reports" not in js
    assert "Ollama model VRAM" in js
    assert "Unload this model from RAM and VRAM" in js
    assert "downloads not reported" in js
