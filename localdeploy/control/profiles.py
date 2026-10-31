"""Profile CRUD for the live config.json.

`config.json` mirrors what the user actually has: pulling a model auto-creates a
profile (see ``models.py`` + ``_config.ensure_profile_for_model``); these routes
let the UI edit a profile's tuning and delete profiles (e.g. orphans whose model
was never pulled, or a profile whose model the user just deleted).

All writes go through the shared atomic writer and refuse to touch the shipped
config.example.json fixture.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from ..utils import is_loopback_url, strip_trailing_slash
from ._config import default_profile_for, refuse_example, slugify_profile_name, write_config_atomic

router = APIRouter()

# Tuning fields the UI is allowed to patch on an existing profile. Structural
# fields (name/backend/model_id/base_url) are intentionally excluded so an edit
# can't silently repoint a profile at a different model or backend.
_EDITABLE_FIELDS = {
    "enabled",
    "description",
    "supports_vision",
    "max_images",
    "context_limit",
    "safe_context_limit",
    "max_prompt_chars",
    "max_output_tokens",
    "temperature",
    "top_p",
    "repeat_penalty",
    "think",
    "timeout_seconds",
    "slow_response_seconds",
    "quantization",
    "gpu_layers",
    "flash_attention",
    "kv_cache_type_k",
    "kv_cache_type_v",
    "batch_size",
    "ubatch_size",
    "threads",
    "mmap",
    "mlock",
    "server_args",
}


class UpsertProfileRequest(BaseModel):
    # Either edit an existing profile by name, or create one for a model_id.
    profile: Optional[str] = None
    model_id: Optional[str] = None
    backend: Optional[str] = None
    base_url: Optional[str] = None
    fields: Dict[str, Any] = {}


@router.post("/profiles/upsert")
def upsert_profile(req: UpsertProfileRequest) -> Dict[str, Any]:
    """Create a profile for `model_id`, or patch an existing profile's tuning.

    Create: pass ``model_id`` (optionally ``fields`` to override defaults).
    Edit:   pass ``profile`` plus ``fields`` (only whitelisted keys are applied).
    """
    from api_server import get_config_path, load_config

    path = get_config_path()
    refusal = refuse_example(path)
    if refusal:
        return {"success": False, "error": refusal}
    config = load_config()
    profiles = config.setdefault("profiles", {})
    requested_backend = str(req.backend or "ollama").strip().lower()
    supported = {"ollama", "llamacpp", "lmstudio", "vllm", "docker", "openai"}
    if requested_backend not in supported:
        return {"success": False, "error": f"Unsupported backend '{requested_backend}'."}
    if req.base_url and not is_loopback_url(req.base_url):
        return {"success": False, "error": "Provider base_url must use localhost, 127.0.0.1, or ::1."}

    if req.profile:
        if req.backend or req.base_url:
            return {"success": False, "error": "backend and base_url can only be set when creating a profile."}
        prof = profiles.get(req.profile)
        if not prof:
            return {"success": False, "error": f"Unknown profile '{req.profile}'."}
        name = req.profile
    elif req.model_id:
        # Reuse an existing profile for this exact model if present, else create.
        name = next(
            (
                p
                for p, v in profiles.items()
                if v.get("model_id") == req.model_id
                and str(v.get("backend") or "ollama").lower() == requested_backend
                and (
                    not req.base_url
                    or strip_trailing_slash(str(v.get("base_url") or ""))
                    == strip_trailing_slash(req.base_url)
                )
            ),
            None,
        )
        if name is None:
            name = slugify_profile_name(req.model_id)
            if name in profiles:
                suffix = 2
                while f"{name}_{suffix}" in profiles:
                    suffix += 1
                name = f"{name}_{suffix}"
            profiles[name] = default_profile_for(req.model_id, backend=requested_backend)
            if req.base_url:
                profiles[name]["base_url"] = strip_trailing_slash(req.base_url)
        prof = profiles[name]
    else:
        return {"success": False, "error": "Pass 'profile' (to edit) or 'model_id' (to create)."}

    for key, value in (req.fields or {}).items():
        if key in _EDITABLE_FIELDS:
            prof[key] = value

    # Same rule as the pull auto-create path: the first profile ever created
    # becomes the default, so /chat works without further setup.
    if not config.get("default_profile"):
        config["default_profile"] = name

    err = write_config_atomic(config, path)
    if err:
        return {"success": False, "error": err}
    return {"success": True, "profile": name, "profile_data": prof, "path": str(path)}


class DeleteProfileRequest(BaseModel):
    profile: str


@router.post("/profiles/delete")
def delete_profile(req: DeleteProfileRequest) -> Dict[str, Any]:
    """Remove a profile from the live config (e.g. an orphan or a deleted model)."""
    from api_server import get_config_path, load_config

    path = get_config_path()
    refusal = refuse_example(path)
    if refusal:
        return {"success": False, "error": refusal}
    config = load_config()
    profiles = config.get("profiles", {})
    if req.profile not in profiles:
        return {"success": False, "error": f"Unknown profile '{req.profile}'."}
    del profiles[req.profile]
    # If we just deleted the default, drop the stale pointer.
    if config.get("default_profile") == req.profile:
        config["default_profile"] = next(iter(profiles), None)
    err = write_config_atomic(config, path)
    if err:
        return {"success": False, "error": err}
    return {"success": True, "deleted": req.profile, "path": str(path)}
