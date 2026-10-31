"""Tests for profile CRUD + auto-create-on-pull (config.json ⇄ UI sync).

config.json is a live mirror of what the user has pulled: pulling a model
auto-creates its profile, and the UI can edit/delete profiles. These verify the
new /profiles/* routes and the _config helpers, and that the shipped
config.example.json fixture is never written.
"""
from __future__ import annotations

import json

import pytest

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover
    pytest.skip("FastAPI TestClient requires httpx", allow_module_level=True)

import api_server
from api_server import app
from localdeploy.control import _config

client = TestClient(app)

EXAMPLE_CONFIG = str(api_server.APP_DIR / "config.example.json")


@pytest.fixture()
def live_config(monkeypatch, tmp_path):
    """Point CONFIG_PATH at a writable minimal config.json for the test."""
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps({"version": 1, "default_profile": None, "profiles": {}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("CONFIG_PATH", str(cfg))
    return cfg


def _read(cfg):
    return json.loads(cfg.read_text(encoding="utf-8"))


# --- slug / default-profile helpers -----------------------------------------


def test_slugify_profile_name():
    assert _config.slugify_profile_name("gemma3:4b") == "gemma3_4b"
    assert _config.slugify_profile_name("hf.co/Org/Repo-GGUF") == "hf_co_org_repo_gguf"


def test_default_profile_detects_vision():
    assert _config.default_profile_for("qwen3-vl:8b-instruct")["supports_vision"] is True
    assert _config.default_profile_for("qwen3.6:27b")["supports_vision"] is True
    assert _config.default_profile_for("qwen3.6:27b")["max_images"] == 4
    assert _config.default_profile_for("qwen3.6:27b")["think"] is False
    assert "think" not in _config.default_profile_for("gemma3:4b")
    assert _config.default_profile_for("qwen2.5:7b")["supports_vision"] is False
    assert _config.default_profile_for("qwen2.5:7b")["max_images"] == 1


# --- upsert (create + edit) --------------------------------------------------


def test_upsert_creates_profile_from_model(live_config):
    body = client.post(
        "/profiles/upsert",
        json={"model_id": "gemma3:4b", "fields": {"context_limit": 2048}},
    ).json()
    assert body["success"] is True
    assert body["profile"] == "gemma3_4b"
    saved = _read(live_config)["profiles"]["gemma3_4b"]
    assert saved["model_id"] == "gemma3:4b"
    assert saved["backend"] == "ollama"
    assert saved["context_limit"] == 2048


def test_upsert_edits_existing_and_ignores_structural_fields(live_config):
    client.post("/profiles/upsert", json={"model_id": "qwen3:8b"})
    body = client.post(
        "/profiles/upsert",
        json={
            "profile": "qwen3_8b",
            "fields": {"temperature": 0.9, "max_images": 8, "model_id": "evil:99b", "backend": "hax"},
        },
    ).json()
    assert body["success"] is True
    saved = _read(live_config)["profiles"]["qwen3_8b"]
    assert saved["temperature"] == 0.9
    assert saved["max_images"] == 8
    # structural fields are not editable via upsert
    assert saved["model_id"] == "qwen3:8b"
    assert saved["backend"] == "ollama"


def test_upsert_can_set_extra_llama_server_flags(live_config):
    client.post("/profiles/upsert", json={"model_id": "qwen3:8b"})
    client.post(
        "/profiles/upsert",
        json={"profile": "qwen3_8b", "fields": {"server_args": ["--reasoning", "off", "-np", "1"]}},
    )
    assert _read(live_config)["profiles"]["qwen3_8b"]["server_args"] == ["--reasoning", "off", "-np", "1"]


def test_upsert_same_model_twice_reuses_profile(live_config):
    a = client.post("/profiles/upsert", json={"model_id": "llama3.1:8b"}).json()
    b = client.post("/profiles/upsert", json={"model_id": "llama3.1:8b"}).json()
    assert a["profile"] == b["profile"]
    assert len(_read(live_config)["profiles"]) == 1


def test_upsert_creates_loopback_provider_profile(live_config):
    body = client.post(
        "/profiles/upsert",
        json={
            "model_id": "local/model",
            "backend": "lmstudio",
            "base_url": "http://127.0.0.1:1234",
        },
    ).json()
    assert body["success"] is True
    saved = _read(live_config)["profiles"][body["profile"]]
    assert saved["backend"] == "lmstudio"
    assert saved["base_url"] == "http://127.0.0.1:1234"


def test_upsert_rejects_remote_provider_url(live_config):
    body = client.post(
        "/profiles/upsert",
        json={"model_id": "remote", "backend": "openai", "base_url": "https://api.example.com"},
    ).json()
    assert body["success"] is False
    assert "localhost" in body["error"]


def test_same_provider_model_on_different_loopback_ports_gets_distinct_profiles(live_config):
    first = client.post(
        "/profiles/upsert",
        json={"model_id": "shared/model", "backend": "openai", "base_url": "http://127.0.0.1:8001"},
    ).json()
    second = client.post(
        "/profiles/upsert",
        json={"model_id": "shared/model", "backend": "openai", "base_url": "http://127.0.0.1:8002"},
    ).json()
    assert first["profile"] != second["profile"]


def test_upsert_requires_profile_or_model(live_config):
    body = client.post("/profiles/upsert", json={}).json()
    assert body["success"] is False


def test_upsert_refuses_example_config(monkeypatch):
    # The shipped example fixture must never be written to.
    monkeypatch.setenv("CONFIG_PATH", EXAMPLE_CONFIG)
    body = client.post("/profiles/upsert", json={"model_id": "gemma3:4b"}).json()
    assert body["success"] is False
    assert "config.example.json" in body["error"]


# --- delete ------------------------------------------------------------------


def test_delete_profile(live_config):
    client.post("/profiles/upsert", json={"model_id": "mistral:7b"})
    body = client.post("/profiles/delete", json={"profile": "mistral_7b"}).json()
    assert body["success"] is True
    assert "mistral_7b" not in _read(live_config)["profiles"]


def test_delete_clears_stale_default(live_config):
    client.post("/profiles/upsert", json={"model_id": "phi4-mini"})
    client.post("/system/set-default", json={"profile": "phi4_mini"})
    assert _read(live_config)["default_profile"] == "phi4_mini"
    client.post("/profiles/delete", json={"profile": "phi4_mini"})
    assert _read(live_config)["default_profile"] is None


def test_delete_unknown_profile(live_config):
    body = client.post("/profiles/delete", json={"profile": "nope"}).json()
    assert body["success"] is False


# --- auto-create-on-pull helper ---------------------------------------------


def test_ensure_profile_for_model_is_idempotent(live_config):
    name1, created1, err1 = _config.ensure_profile_for_model("gemma3:12b")
    assert err1 is None and created1 is True and name1 == "gemma3_12b"
    # First model with no prior default becomes the default.
    assert _read(live_config)["default_profile"] == "gemma3_12b"
    name2, created2, err2 = _config.ensure_profile_for_model("gemma3:12b")
    assert name2 == "gemma3_12b" and created2 is False and err2 is None
    assert len(_read(live_config)["profiles"]) == 1


def test_ensure_profile_skips_example_fixture(monkeypatch):
    monkeypatch.setenv("CONFIG_PATH", EXAMPLE_CONFIG)
    name, created, err = _config.ensure_profile_for_model("some-brand-new:9b")
    assert created is False and err is None  # silently skipped, no write
