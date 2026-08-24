"""GitHub-release update check (Release R7, Phase A).

GET /system/update-check compares the running LocalDeploy version against
the project's own GitHub releases feed. Like the Hugging Face model search,
this is the one deliberate outbound egress path in this feature, gated by
the same OFFLINE=true switch that disables all other internet calls, and
called once when the UI starts. Fully offline installations can disable it.

No telemetry: this is a plain GET against GitHub's public releases API for
this project's own repository. Nothing about the local machine, its models,
or its usage is sent - the request carries no query parameters or body.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter

from .. import __version__
from ..utils import offline_mode

router = APIRouter()

_REPO = "oney-erge/LocalDeploy"
_RELEASES_API = f"https://api.github.com/repos/{_REPO}/releases"
_REQUEST_TIMEOUT = 6


def _channel() -> str:
    value = (os.getenv("UPDATE_CHANNEL") or "stable").strip().lower()
    return value if value in ("stable", "preview") else "stable"


def _parse_version(text: str) -> Tuple[int, ...]:
    """'v0.6.0' / '0.6.0-beta1' -> (0, 6, 0). Non-numeric segments are dropped
    rather than raising, so an odd tag name degrades to 'unknown' instead of
    crashing the whole check."""
    cleaned = text.strip().lstrip("vV")
    parts = re.split(r"[.\-+]", cleaned)
    numeric: List[int] = []
    for part in parts:
        if part.isdigit():
            numeric.append(int(part))
        else:
            break
    return tuple(numeric)


def _is_newer(latest: str, current: str) -> Optional[bool]:
    latest_v, current_v = _parse_version(latest), _parse_version(current)
    if not latest_v or not current_v:
        return None
    return latest_v > current_v


def _fetch_latest(channel: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    import requests

    try:
        if channel == "preview":
            resp = requests.get(f"{_RELEASES_API}?per_page=5", timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
            releases = resp.json()
            if not isinstance(releases, list) or not releases:
                return None, "No releases found."
            return releases[0], None  # GitHub lists releases newest-first, prereleases included
        resp = requests.get(f"{_RELEASES_API}/latest", timeout=_REQUEST_TIMEOUT)
        if resp.status_code == 404:
            return None, "No stable release has been published yet."
        resp.raise_for_status()
        return resp.json(), None
    except requests.ConnectionError:
        return None, "Could not reach GitHub (offline or no network)."
    except requests.Timeout:
        return None, "GitHub release check timed out."
    except requests.RequestException as exc:
        return None, str(exc)


@router.get("/system/update-check")
def update_check() -> Dict[str, Any]:
    channel = _channel()
    if offline_mode():
        return {
            "success": True,
            "checked": False,
            "current_version": __version__,
            "channel": channel,
            "message": "offline mode (OFFLINE=true): update check skipped - no egress.",
        }

    release, error = _fetch_latest(channel)
    if error or not release:
        return {
            "success": True,
            "checked": False,
            "current_version": __version__,
            "channel": channel,
            "message": error or "Could not determine the latest release.",
        }

    tag = str(release.get("tag_name") or "")
    update_available = _is_newer(tag, __version__)
    return {
        "success": True,
        "checked": True,
        "current_version": __version__,
        "latest_version": tag.lstrip("vV") or None,
        "update_available": bool(update_available),
        "version_comparable": update_available is not None,
        "channel": channel,
        "prerelease": bool(release.get("prerelease")),
        "url": release.get("html_url"),
        "published_at": release.get("published_at"),
        "notes": release.get("body"),
    }
