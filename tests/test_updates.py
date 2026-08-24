"""Tests for the GitHub-release update check (Release R7 Phase A)."""
from __future__ import annotations

import pytest

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover
    pytest.skip("FastAPI TestClient requires httpx", allow_module_level=True)

from api_server import app
from localdeploy import __version__
from localdeploy.control import updates as updates_mod

client = TestClient(app)


def test_version_parsing():
    assert updates_mod._parse_version("v0.6.0") == (0, 6, 0)
    assert updates_mod._parse_version("0.6.0") == (0, 6, 0)
    assert updates_mod._parse_version("v0.6.0-beta1") == (0, 6, 0)
    assert updates_mod._parse_version("not-a-version") == ()


def test_is_newer():
    assert updates_mod._is_newer("v0.6.0", "0.5.1") is True
    assert updates_mod._is_newer("v0.5.1", "0.5.1") is False
    assert updates_mod._is_newer("v0.5.0", "0.5.1") is False
    assert updates_mod._is_newer("not-a-version", "0.5.1") is None


def test_channel_defaults_to_stable(monkeypatch):
    monkeypatch.delenv("UPDATE_CHANNEL", raising=False)
    assert updates_mod._channel() == "stable"


def test_channel_reads_env(monkeypatch):
    monkeypatch.setenv("UPDATE_CHANNEL", "preview")
    assert updates_mod._channel() == "preview"
    monkeypatch.setenv("UPDATE_CHANNEL", "garbage")
    assert updates_mod._channel() == "stable"  # unrecognized value falls back safely


def test_update_check_offline_mode_skips_egress(monkeypatch):
    monkeypatch.setenv("OFFLINE", "true")
    body = client.get("/system/update-check").json()
    assert body["success"] is True
    assert body["checked"] is False
    assert "offline" in body["message"].lower()
    assert body["current_version"] == __version__
    monkeypatch.setenv("OFFLINE", "false")


def test_update_check_reports_update_available(monkeypatch):
    monkeypatch.setattr(
        updates_mod,
        "_fetch_latest",
        lambda channel: (
            {"tag_name": "v99.0.0", "html_url": "https://github.com/oney-erge/LocalDeploy/releases/tag/v99.0.0",
             "published_at": "2026-01-01T00:00:00Z", "body": "Big release.", "prerelease": False},
            None,
        ),
    )
    body = client.get("/system/update-check").json()
    assert body["checked"] is True
    assert body["latest_version"] == "99.0.0"
    assert body["update_available"] is True
    assert body["url"].endswith("v99.0.0")


def test_update_check_reports_up_to_date(monkeypatch):
    monkeypatch.setattr(
        updates_mod,
        "_fetch_latest",
        lambda channel: ({"tag_name": f"v{__version__}", "html_url": "x", "published_at": None, "body": None, "prerelease": False}, None),
    )
    body = client.get("/system/update-check").json()
    assert body["checked"] is True
    assert body["update_available"] is False


def test_update_check_network_failure_is_graceful(monkeypatch):
    monkeypatch.setattr(updates_mod, "_fetch_latest", lambda channel: (None, "Could not reach GitHub (offline or no network)."))
    body = client.get("/system/update-check").json()
    assert body["success"] is True
    assert body["checked"] is False
    assert "GitHub" in body["message"]


def test_update_check_never_sends_query_params_or_body():
    # No telemetry: this endpoint takes no request body and forwards nothing
    # about the local machine - GET with no params is the entire contract.
    import inspect

    sig = inspect.signature(updates_mod.update_check)
    assert len(sig.parameters) == 0
