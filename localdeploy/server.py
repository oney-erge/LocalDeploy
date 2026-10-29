from __future__ import annotations

import base64
import hmac
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import localdeploy
from localdeploy.backends import decision
from localdeploy.backends.llamacpp import call_llamacpp, llama_health
from localdeploy.backends.ollama import (
    call_ollama_detailed,
    embed_ollama,
    encode_embedding_base64,
    ollama_models,
    stream_ollama_events,
)
from localdeploy.backends.openai_compatible import (
    call_openai_compatible,
    embed_openai_compatible,
    stream_openai_compatible,
)
from localdeploy.utils import (
    BackendCallError,
    api_token,
    app_home,
    enable_web_ui,
    env_bool,
    env_float,
    env_int,
    get_backend_base_url,
    model_dump_compat,
    model_validate_compat,
    require_gpu_only,
    web_dir,
)


# The repo root in a source checkout (this file is localdeploy/server.py).
APP_DIR = Path(__file__).resolve().parents[1]
# Runtime state (.env, config.json, logs/, reports/) lives in the app home:
# the repo root for a source checkout, ~/.localdeploy for pip installs.
APP_HOME = app_home()
load_dotenv(APP_HOME / ".env")

SUPPORTED_BACKENDS = {"ollama", "llamacpp", "lmstudio", "vllm", "docker", "openai", decision.DECISION_BACKEND}


class ChatRequest(BaseModel):
    profile: Optional[str] = None
    model: Optional[str] = None
    backend: Optional[str] = None
    # Ollama keep-alive for this request (e.g. "5m", "1h", "-1" = until unloaded).
    keep_alive: Optional[str] = None
    prompt: str = ""
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    repeat_penalty: Optional[float] = None
    max_output_tokens: Optional[int] = None
    context_limit: Optional[int] = None
    safe_mode: bool = True
    allow_clamp: bool = False
    timeout_seconds: Optional[int] = None
    # Advanced: pin GPU layers for this request (0 = force CPU, large = force GPU).
    # Threaded into the Ollama options so a CPU/GPU benchmark measures the device it
    # asked for instead of letting Ollama silently re-place the model.
    num_gpu: Optional[int] = None
    # Passed to backends that support grammar/schema-constrained generation.
    # Native and OpenAI-compatible callers use the same response-format shape.
    response_format: Optional[Dict[str, Any]] = None
    messages: Optional[List[Dict[str, Any]]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None


class VisionRequest(ChatRequest):
    images_base64: Optional[List[str]] = None


class BenchmarkRequest(ChatRequest):
    all_profiles: bool = False
    prompts: Optional[List[str]] = None
    image_path: Optional[str] = None
    images_base64: Optional[List[str]] = None


class EstimateRequest(VisionRequest):
    pass


class OpenAIChatMessage(BaseModel):
    role: str
    content: Any = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class OpenAIChatCompletionRequest(BaseModel):
    model: str = Field(default="gemma3_4b_ollama_safe")
    messages: List[OpenAIChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    stream: bool = False
    response_format: Optional[Dict[str, Any]] = None
    profile: Optional[str] = None
    keep_alive: Optional[str] = None
    context_limit: Optional[int] = None
    safe_mode: bool = True
    allow_clamp: bool = False
    timeout_seconds: Optional[int] = None
    repeat_penalty: Optional[float] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    parallel_tool_calls: Optional[bool] = None


class OpenAIEmbeddingRequest(BaseModel):
    input: Any
    model: str
    encoding_format: str = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None


class SystemOneRequest(BaseModel):
    """Typed questions about one piece of text, answered by a decision model."""

    state: Any = None
    questions: Dict[str, Any]
    model: Optional[str] = None
    profile: Optional[str] = None


class OpenAIResponseRequest(BaseModel):
    model: str
    input: Any
    instructions: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    max_output_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: bool = False
    profile: Optional[str] = None


class LocalLLMResponse(BaseModel):
    success: bool
    backend: Optional[str] = None
    profile: Optional[str] = None
    model: Optional[str] = None
    response: Optional[Any] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    metrics: Optional[Dict[str, Any]] = None
    elapsed_seconds: float = 0.0
    estimated_prompt_chars: int = 0
    estimated_prompt_tokens: int = 0
    context_limit_used: Optional[int] = None
    max_output_tokens_used: Optional[int] = None
    warning: Optional[str] = None
    error: Optional[str] = None


def get_config_path() -> Path:
    configured = os.getenv("CONFIG_PATH", "config.json")
    path = Path(configured)
    if not path.is_absolute():
        path = APP_HOME / path
    return path


def load_config() -> Dict[str, Any]:
    """Load the live config, or a minimal empty one when none exists yet.

    A fresh clone deliberately starts with zero profiles: config.json mirrors
    what the user actually has, and pulling a model auto-creates its profile.
    (config.example.json is documentation/reference only - falling back to it
    here would show a dozen models the user never pulled.)
    """
    path = get_config_path()
    if not path.exists():
        return {
            "version": 1,
            "default_profile": None,
            "global_defaults": {"safe_mode": True, "stream": False, "require_gpu_only": False},
            "profiles": {},
        }
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_global_limits() -> Dict[str, Any]:
    return {
        "max_prompt_chars": env_int("GLOBAL_MAX_PROMPT_CHARS", 20000),
        "max_output_tokens": env_int("GLOBAL_MAX_OUTPUT_TOKENS", 2048),
        # Per-profile limits still default to one for existing profiles. This is
        # only the hard server ceiling, allowing qualified vision profiles to opt
        # into evidence bundles without weakening legacy 8 GB defaults.
        "max_images": env_int("GLOBAL_MAX_IMAGES", 8),
        "max_image_mb": env_float("GLOBAL_MAX_IMAGE_MB", 10.0),
        "request_timeout_seconds": env_int("REQUEST_TIMEOUT_SECONDS", 180),
        "slow_response_seconds": env_int("SLOW_RESPONSE_SECONDS", 60),
    }


def profile_warning(profile: Dict[str, Any]) -> Optional[str]:
    recommended = profile.get("recommended_for_8gb_vram")
    notes = str(profile.get("notes", "")).lower()
    name = profile.get("name", "selected profile")
    if recommended == "experimental" or "experimental" in notes:
        return f"{name} is experimental on 8 GB VRAM; benchmark stability before relying on it."
    if recommended == "fallback":
        return f"{name} is a fallback profile and may be much slower due to partial CPU offload."
    if recommended is False:
        return f"{name} is not the recommended 8 GB VRAM default; use safe mode and conservative context."
    return None


def estimate_tokens_from_chars(char_count: int) -> int:
    return max(1, math.ceil(char_count / 4))


def normalize_images(images: Optional[Any]) -> List[str]:
    if images is None:
        return []
    if isinstance(images, str):
        return [images]
    return [str(item) for item in images if item is not None]


def extract_openai_content(content: Any) -> Tuple[str, List[str]]:
    if isinstance(content, str):
        return content, []

    if not isinstance(content, list):
        return str(content), []

    text_parts: List[str] = []
    images: List[str] = []
    for item in content:
        if isinstance(item, str):
            text_parts.append(item)
            continue
        if not isinstance(item, dict):
            text_parts.append(str(item))
            continue

        item_type = item.get("type")
        if item_type == "text":
            text_parts.append(str(item.get("text", "")))
        elif item_type == "image_url":
            image_url = item.get("image_url")
            if isinstance(image_url, dict):
                url = str(image_url.get("url", ""))
            else:
                url = str(image_url or "")
            if url.startswith("data:") and "," in url:
                images.append(url)
        elif item_type == "input_text":
            text_parts.append(str(item.get("text", "")))
        elif item_type == "input_image":
            image_value = str(item.get("image_url") or item.get("image_base64") or "")
            if image_value:
                images.append(image_value)

    return "\n".join(part for part in text_parts if part), images


def clean_base64_image(value: str) -> str:
    cleaned = "".join(str(value).split())
    if cleaned.startswith("data:") and "," in cleaned:
        cleaned = cleaned.split(",", 1)[1]
    return cleaned


def decoded_image_size_bytes(value: str) -> int:
    cleaned = clean_base64_image(value)
    try:
        return len(base64.b64decode(cleaned, validate=True))
    except Exception as exc:
        raise ValueError("Image input is not valid base64.") from exc


def clamp_or_error(
    requested: int,
    allowed: int,
    field_name: str,
    allow_clamp: bool,
    errors: List[str],
) -> int:
    if requested <= 0:
        errors.append(f"{field_name} must be greater than zero.")
        return requested
    if requested > allowed:
        if allow_clamp:
            return allowed
        errors.append(f"{field_name} {requested} exceeds configured limit {allowed}. Set allow_clamp=true to clamp safely.")
    return requested


def resolve_profile(
    config: Dict[str, Any],
    request_data: Dict[str, Any],
    require_enabled: bool = True,
) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
    profiles = config.get("profiles", {})
    if not profiles:
        return None, None, (
            "No model profiles configured yet. Pull a model first (UI → Get a model, "
            "or POST /models/pull) - a profile is created automatically."
        )
    profile_name = (
        request_data.get("profile")
        or os.getenv("DEFAULT_MODEL_PROFILE")
        or config.get("default_profile")
        or "gemma3_4b_ollama_safe"
    )
    if profile_name not in profiles:
        return None, None, f"Unknown profile '{profile_name}'. Check config.json."

    profile = dict(profiles[profile_name])
    profile["name"] = profile.get("name") or profile_name

    if require_enabled and not profile.get("enabled", False):
        return profile_name, profile, f"Profile '{profile_name}' is disabled in config.json."

    requested_backend = request_data.get("backend")
    if requested_backend:
        original_backend = str(profile.get("backend") or "ollama").strip().lower()
        backend = str(requested_backend).strip().lower()
        if backend not in SUPPORTED_BACKENDS:
            return profile_name, profile, f"backend must be one of: {', '.join(sorted(SUPPORTED_BACKENDS))}."
        profile["backend"] = backend
        if backend != original_backend:
            profile.pop("base_url", None)

    requested_model = request_data.get("model")
    if requested_model:
        profile["model_id"] = str(requested_model)

    backend = str(profile.get("backend", "")).lower()
    if backend not in SUPPORTED_BACKENDS:
        return profile_name, profile, f"Unsupported backend '{backend}' in profile '{profile_name}'."

    if require_gpu_only() and backend == decision.DECISION_BACKEND:
        if str(profile.get("device") or "auto").lower() != "cuda":
            return profile_name, profile, (
                f"GPU-only mode is enabled, but decision profile '{profile_name}' does not set device to 'cuda'."
            )
    elif require_gpu_only():
        gpu_layers = str(profile.get("gpu_layers") or "").strip().lower()
        if backend != "llamacpp":
            return profile_name, profile, (
                f"GPU-only mode is enabled, but profile '{profile_name}' uses backend '{backend}'. "
                "Select a llama.cpp profile with gpu_layers=all."
            )
        if gpu_layers != "all":
            return profile_name, profile, (
                f"GPU-only mode is enabled, but profile '{profile_name}' has gpu_layers={profile.get('gpu_layers')!r}. "
                "Set gpu_layers to 'all' so startup fails instead of falling back to CPU."
            )

    if backend == "llamacpp" and not env_bool("ENABLE_LLAMA_CPP", False):
        return (
            profile_name,
            profile,
            "llama.cpp backend is disabled. Set ENABLE_LLAMA_CPP=true after starting a local llama-server.",
        )

    if backend != decision.DECISION_BACKEND:  # decision models run in-process; the HTTP adapter checks its own URL
        try:
            profile["base_url"] = get_backend_base_url(profile, backend)
        except BackendCallError as exc:
            return profile_name, profile, str(exc)

    return profile_name, profile, None


def profile_for_openai_model(config: Dict[str, Any], model_name: str) -> Tuple[str, Optional[str]]:
    profiles = config.get("profiles", {})
    if model_name in profiles:
        return model_name, None

    for profile_name, profile in profiles.items():
        if str(profile.get("model_id")) == model_name:
            return profile_name, None

    default_profile = (
        os.getenv("DEFAULT_MODEL_PROFILE")
        or config.get("default_profile")
        or "gemma3_4b_ollama_safe"
    )
    return default_profile, model_name


def openai_request_to_local_payload(request_model: OpenAIChatCompletionRequest) -> Tuple[Dict[str, Any], bool]:
    config = load_config()
    requested_profile, model_override = profile_for_openai_model(config, request_model.profile or request_model.model)
    system_parts: List[str] = []
    conversation_parts: List[str] = []
    images: List[str] = []

    for message in request_model.messages:
        role = message.role.lower()
        text, message_images = extract_openai_content(message.content)
        images.extend(message_images)
        if role == "system":
            if text:
                system_parts.append(text)
        else:
            if text:
                conversation_parts.append(f"{role}: {text}" if role != "user" else text)

    prompt = "\n\n".join(conversation_parts).strip()
    if request_model.response_format:
        prompt = f"{prompt}\n\n{response_format_instruction(request_model.response_format)}".strip()

    payload: Dict[str, Any] = {
        "profile": requested_profile,
        "prompt": prompt,
        "system_prompt": "\n\n".join(system_parts).strip() or None,
        "temperature": request_model.temperature,
        "top_p": request_model.top_p,
        "repeat_penalty": request_model.repeat_penalty,
        "max_output_tokens": request_model.max_tokens or request_model.max_completion_tokens,
        "context_limit": request_model.context_limit,
        "safe_mode": request_model.safe_mode,
        "allow_clamp": request_model.allow_clamp,
        "timeout_seconds": request_model.timeout_seconds,
        "response_format": request_model.response_format,
        "keep_alive": request_model.keep_alive,
        "messages": [model_dump_compat(message) for message in request_model.messages],
        "tools": request_model.tools,
        "tool_choice": request_model.tool_choice,
    }
    if model_override:
        payload["model"] = model_override
    if images:
        payload["images_base64"] = images
    return payload, bool(images)


def response_format_instruction(response_format: Dict[str, Any]) -> str:
    format_type = response_format.get("type")
    if format_type == "json_schema":
        json_schema = response_format.get("json_schema") or {}
        schema = json_schema.get("schema") or {}
        required = schema.get("required") or []
        properties = schema.get("properties") or {}
        property_lines = []
        for key, details in properties.items():
            if isinstance(details, dict):
                value_type = details.get("type", "value")
                description = details.get("description")
                suffix = f" - {description}" if description else ""
                property_lines.append(f"- {key}: {value_type}{suffix}")
            else:
                property_lines.append(f"- {key}")
        return (
            "Return only valid JSON. Do not use markdown or code fences. "
            "Return a data object that conforms to this schema; do not return the schema itself. "
            f"Required keys: {', '.join(required) if required else 'use the schema properties'}. "
            f"Properties:\n{chr(10).join(property_lines) if property_lines else json.dumps(properties, ensure_ascii=False)}"
        )
    if format_type == "json_object":
        return "Return only one valid JSON object. Do not use markdown or code fences."
    return (
        "Return only valid JSON when possible. Do not use markdown or code fences. "
        f"Response format request: {json.dumps(response_format, ensure_ascii=False)}"
    )


def prepare_request(
    kind: str,
    request_data: Dict[str, Any],
    require_enabled: bool = True,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    config = load_config()
    limits = get_global_limits()
    errors: List[str] = []

    model_cls = VisionRequest if kind == "vision" else ChatRequest
    if kind == "benchmark":
        model_cls = BenchmarkRequest
    if kind == "estimate":
        model_cls = EstimateRequest

    try:
        request_model = model_validate_compat(model_cls, request_data)
    except Exception as exc:
        return None, make_error_response(error=f"Invalid request schema: {exc}")

    data = model_dump_compat(request_model)
    data["images_base64"] = normalize_images(data.get("images_base64"))
    profile_name, profile, profile_error = resolve_profile(config, data, require_enabled=require_enabled)
    if profile_error:
        return None, make_error_response(
            error=profile_error,
            backend=profile.get("backend") if profile else data.get("backend"),
            profile=profile_name,
            model=profile.get("model_id") if profile else data.get("model"),
            warning=profile_warning(profile) if profile else None,
        )

    if profile is None:
        return None, make_error_response(error="Profile resolution returned no profile.")
    backend = str(profile.get("backend")).lower()
    if backend == decision.DECISION_BACKEND:
        return None, make_error_response(
            error=(
                f"Invalid request: profile '{profile_name}' is a decision model. It answers typed questions at "
                "POST /v1/systemone and does not generate text."
            ),
            backend=backend,
            profile=profile_name,
            model=profile.get("model_id"),
        )
    prompt = data.get("prompt") or ""
    system_prompt = data.get("system_prompt") or ""
    response_format = data.get("response_format")
    response_format_chars = len(json.dumps(response_format, ensure_ascii=False)) if response_format else 0
    estimated_prompt_chars = len(prompt) + len(system_prompt) + response_format_chars
    estimated_prompt_tokens = estimate_tokens_from_chars(estimated_prompt_chars)
    allow_clamp = bool(data.get("allow_clamp", False))
    safe_mode = bool(data.get("safe_mode", True))

    profile_prompt_limit = int(profile.get("max_prompt_chars") or limits["max_prompt_chars"])
    prompt_limit = min(profile_prompt_limit, int(limits["max_prompt_chars"]))
    if estimated_prompt_chars > prompt_limit:
        errors.append(
            f"Prompt is too large: {estimated_prompt_chars} characters exceeds configured limit {prompt_limit}."
        )

    profile_context_limit = int(profile.get("context_limit") or 2048)
    profile_safe_context_limit = int(profile.get("safe_context_limit") or profile_context_limit)
    allowed_context = min(profile_context_limit, profile_safe_context_limit) if safe_mode else profile_context_limit
    # `or` would silently replace an explicit 0 with the default instead of
    # letting the "must be greater than zero" validation below catch it.
    requested_context_raw = data.get("context_limit")
    requested_context = int(requested_context_raw if requested_context_raw is not None else allowed_context)
    context_limit_used = clamp_or_error(
        requested_context,
        allowed_context,
        "context_limit",
        allow_clamp,
        errors,
    )

    profile_output_limit = int(profile.get("max_output_tokens") or limits["max_output_tokens"])
    allowed_output = min(profile_output_limit, int(limits["max_output_tokens"]))
    requested_output_raw = data.get("max_output_tokens")
    requested_output = int(requested_output_raw if requested_output_raw is not None else allowed_output)
    max_output_tokens_used = clamp_or_error(
        requested_output,
        allowed_output,
        "max_output_tokens",
        allow_clamp,
        errors,
    )

    images = data.get("images_base64") or []
    if images:
        if not profile.get("supports_vision", False):
            errors.append(f"Profile '{profile_name}' does not support vision requests.")
        raw_profile_image_limit = profile.get("max_images")
        profile_image_limit = int(raw_profile_image_limit if raw_profile_image_limit is not None else 1)
        image_limit = min(profile_image_limit, int(limits["max_images"]))
        if image_limit <= 0:
            errors.append(f"Profile '{profile_name}' max_images must be greater than zero.")
        elif len(images) > image_limit:
            errors.append(f"Too many images: {len(images)} exceeds configured limit {image_limit}.")
        max_image_bytes = float(limits["max_image_mb"]) * 1024 * 1024
        cleaned_images: List[str] = []
        for idx, image in enumerate(images, start=1):
            try:
                image_size = decoded_image_size_bytes(image)
            except ValueError as exc:
                errors.append(f"Image {idx}: {exc}")
                continue
            if image_size > max_image_bytes:
                errors.append(
                    f"Image {idx} is too large: {image_size / (1024 * 1024):.2f} MB exceeds "
                    f"configured limit {limits['max_image_mb']} MB."
                )
            cleaned_images.append(clean_base64_image(image))
        data["images_base64"] = cleaned_images

    requested_timeout_raw = data.get("timeout_seconds")
    default_timeout = profile.get("timeout_seconds") or limits["request_timeout_seconds"]
    timeout_seconds = int(requested_timeout_raw if requested_timeout_raw is not None else default_timeout)
    if timeout_seconds <= 0:
        errors.append("timeout_seconds must be greater than zero.")

    if errors:
        return None, make_error_response(
            error=" ".join(errors),
            backend=backend,
            profile=profile_name,
            model=str(profile.get("model_id")),
            estimated_prompt_chars=estimated_prompt_chars,
            estimated_prompt_tokens=estimated_prompt_tokens,
            context_limit_used=context_limit_used,
            max_output_tokens_used=max_output_tokens_used,
            warning=profile_warning(profile),
        )

    prepared = {
        "kind": kind,
        "profile_name": profile_name,
        "profile": profile,
        "backend": backend,
        "model": str(profile.get("model_id")),
        "prompt": prompt,
        "system_prompt": system_prompt,
        "temperature": data.get("temperature") if data.get("temperature") is not None else profile.get("temperature"),
        "top_p": data.get("top_p") if data.get("top_p") is not None else profile.get("top_p"),
        "repeat_penalty": data.get("repeat_penalty")
        if data.get("repeat_penalty") is not None
        else profile.get("repeat_penalty"),
        "context_limit_used": context_limit_used,
        "max_output_tokens_used": max_output_tokens_used,
        "num_gpu": data.get("num_gpu"),
        "timeout_seconds": timeout_seconds,
        "estimated_prompt_chars": estimated_prompt_chars,
        "estimated_prompt_tokens": estimated_prompt_tokens,
        "images_base64": data.get("images_base64") or [],
        "response_format": response_format,
        "keep_alive": data.get("keep_alive"),
        "messages": data.get("messages") or [],
        "tools": data.get("tools") or [],
        "tool_choice": data.get("tool_choice"),
        "safe_mode": safe_mode,
        "warning": profile_warning(profile),
        "slow_response_seconds": int(profile.get("slow_response_seconds") or limits["slow_response_seconds"]),
    }
    return prepared, None


def make_error_response(
    error: str,
    backend: Optional[str] = None,
    profile: Optional[str] = None,
    model: Optional[str] = None,
    elapsed_seconds: float = 0.0,
    estimated_prompt_chars: int = 0,
    estimated_prompt_tokens: int = 0,
    context_limit_used: Optional[int] = None,
    max_output_tokens_used: Optional[int] = None,
    warning: Optional[str] = None,
) -> Dict[str, Any]:
    return model_dump_compat(
        LocalLLMResponse(
            success=False,
            backend=backend,
            profile=profile,
            model=model,
            response=None,
            elapsed_seconds=round(elapsed_seconds, 3),
            estimated_prompt_chars=estimated_prompt_chars,
            estimated_prompt_tokens=estimated_prompt_tokens,
            context_limit_used=context_limit_used,
            max_output_tokens_used=max_output_tokens_used,
            warning=warning,
            error=error,
        )
    )


def make_success_response(
    prepared: Dict[str, Any], content: Any, elapsed_seconds: float, details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    warning = prepared.get("warning")
    if elapsed_seconds >= prepared.get("slow_response_seconds", 60):
        slow_warning = f"Response exceeded slow threshold of {prepared.get('slow_response_seconds')} seconds."
        warning = f"{warning} {slow_warning}" if warning else slow_warning
    return model_dump_compat(
        LocalLLMResponse(
            success=True,
            backend=prepared["backend"],
            profile=prepared["profile_name"],
            model=prepared["model"],
            response=content,
            tool_calls=(details or {}).get("tool_calls") or None,
            metrics=(details or {}).get("metrics") or None,
            elapsed_seconds=round(elapsed_seconds, 3),
            estimated_prompt_chars=prepared["estimated_prompt_chars"],
            estimated_prompt_tokens=prepared["estimated_prompt_tokens"],
            context_limit_used=prepared["context_limit_used"],
            max_output_tokens_used=prepared["max_output_tokens_used"],
            warning=warning,
            error=None,
        )
    )


def _note_monitor_request(
    prepared: Dict[str, Any], kind: str, success: bool, elapsed: float,
    details: Optional[Dict[str, Any]] = None, error: Optional[str] = None,
) -> None:
    """Best-effort numeric-only request record for the Monitor tab (Release
    R3). Never raises - monitoring must never affect the actual response."""
    monitor_mod = None
    try:
        from .control import monitor as monitor_mod

        monitor_mod.record_request(
            profile=prepared.get("profile_name"),
            model=prepared.get("model"),
            backend=prepared.get("backend"),
            kind=kind,
            success=success,
            elapsed_seconds=elapsed,
            metrics=(details or {}).get("metrics"),
            context_limit=prepared.get("context_limit_used"),
            error=error,
        )
    except Exception:
        pass
    finally:
        if monitor_mod is not None:
            try:
                monitor_mod.note_request_end(prepared.get("model"), prepared.get("_monitor_request_id"))
            except Exception:
                pass


def _note_monitor_request_started(prepared: Dict[str, Any]) -> None:
    """Best-effort in-flight counter paired with ``_note_monitor_request``."""
    try:
        from .control import monitor as monitor_mod

        prepared["_monitor_request_id"] = monitor_mod.note_request_start(prepared.get("model"))
    except Exception:
        pass


def run_local_request(kind: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
    prepared, error_response = prepare_request(kind, request_data, require_enabled=True)
    if error_response:
        return error_response
    if prepared is None:
        return make_error_response(error="Request preparation returned no request.")

    _note_monitor_request_started(prepared)
    start = time.perf_counter()
    try:
        if prepared["backend"] == "ollama":
            details = call_ollama_detailed(prepared)
            content = details.get("content") or ""
        elif prepared["backend"] == "llamacpp":
            if prepared.get("tools"):
                details = call_openai_compatible(prepared)
                content = details.get("content") or ""
            else:
                content = call_llamacpp(prepared)
                details = {"content": content, "tool_calls": [], "metrics": {}}
        elif prepared["backend"] in {"lmstudio", "vllm", "docker", "openai"}:
            details = call_openai_compatible(prepared)
            content = details.get("content") or ""
        else:
            raise BackendCallError(f"Unsupported backend '{prepared['backend']}'.")
    except BackendCallError as exc:
        elapsed = time.perf_counter() - start
        _note_monitor_request(prepared, kind, False, elapsed, error=str(exc))
        return make_error_response(
            error=str(exc),
            backend=prepared["backend"],
            profile=prepared["profile_name"],
            model=prepared["model"],
            elapsed_seconds=elapsed,
            estimated_prompt_chars=prepared["estimated_prompt_chars"],
            estimated_prompt_tokens=prepared["estimated_prompt_tokens"],
            context_limit_used=prepared["context_limit_used"],
            max_output_tokens_used=prepared["max_output_tokens_used"],
            warning=prepared.get("warning"),
        )
    except Exception as exc:  # noqa: BLE001 - last-resort guard
        # Any unexpected backend error degrades to a graceful failure response
        # instead of a raw 500 (e.g. an unwrapped JSON/parse error from a backend).
        elapsed = time.perf_counter() - start
        _note_monitor_request(prepared, kind, False, elapsed, error=str(exc))
        return make_error_response(
            error=f"Unexpected backend error: {exc}",
            backend=prepared["backend"],
            profile=prepared["profile_name"],
            model=prepared["model"],
            elapsed_seconds=elapsed,
            estimated_prompt_chars=prepared["estimated_prompt_chars"],
            estimated_prompt_tokens=prepared["estimated_prompt_tokens"],
            context_limit_used=prepared["context_limit_used"],
            max_output_tokens_used=prepared["max_output_tokens_used"],
            warning=prepared.get("warning"),
        )
    elapsed = time.perf_counter() - start
    _note_monitor_request(prepared, kind, True, elapsed, details=details)
    return make_success_response(prepared, content, elapsed, details)


def estimate_request_safety(request_data: Dict[str, Any]) -> Dict[str, Any]:
    prepared, error_response = prepare_request("estimate", request_data, require_enabled=False)
    if error_response:
        error_response["response"] = {
            "likely_safe": False,
            "errors": [error_response.get("error")],
            "warnings": [error_response.get("warning")] if error_response.get("warning") else [],
        }
        return error_response

    if prepared is None:
        response = make_error_response(error="Request preparation returned no request.")
        response["response"] = {
            "likely_safe": False,
            "errors": ["Request preparation returned no request."],
            "warnings": [],
        }
        return response
    warnings: List[str] = []
    profile = prepared["profile"]
    limits = get_global_limits()
    profile_prompt_limit = min(int(profile.get("max_prompt_chars") or limits["max_prompt_chars"]), limits["max_prompt_chars"])
    prompt_ratio = prepared["estimated_prompt_chars"] / max(1, profile_prompt_limit)
    if prompt_ratio >= 0.8:
        warnings.append("Prompt is close to the configured character limit.")

    estimated_total_tokens = prepared["estimated_prompt_tokens"] + prepared["max_output_tokens_used"]
    if estimated_total_tokens > prepared["context_limit_used"]:
        warnings.append(
            "Estimated prompt plus output tokens may exceed the selected context. Tokenization is approximate; lower prompt size or output tokens if generation fails."
        )
    if prepared.get("warning"):
        warnings.append(prepared["warning"])

    response = {
        "likely_safe": True,
        "errors": [],
        "warnings": warnings,
        "selected_profile": prepared["profile_name"],
        "backend": prepared["backend"],
        "model": prepared["model"],
        "safe_mode": prepared["safe_mode"],
        "estimated_prompt_chars": prepared["estimated_prompt_chars"],
        "estimated_prompt_tokens": prepared["estimated_prompt_tokens"],
        "estimated_total_tokens_with_output": estimated_total_tokens,
        "context_limit_used": prepared["context_limit_used"],
        "max_output_tokens_used": prepared["max_output_tokens_used"],
        "global_limits": limits,
    }
    return make_success_response(prepared, response, 0.0)


def benchmark_prompts() -> List[str]:
    return [
        "Explain what this local LLM server is doing in 3 bullet points.",
        "A laptop can process 18 images per minute. How long will it take to process 153 images? Show your reasoning briefly.",
        "Return only valid JSON with keys: model_capability, strengths, weaknesses, recommended_use. No markdown.",
    ]


def run_benchmark(request_data: Dict[str, Any]) -> Dict[str, Any]:
    config = load_config()
    request_model = model_validate_compat(BenchmarkRequest, request_data)
    data = model_dump_compat(request_model)
    prompts = data.get("prompts") or ([data["prompt"]] if data.get("prompt") else benchmark_prompts())
    if data.get("all_profiles"):
        profile_names = [
            name
            for name, profile in config.get("profiles", {}).items()
            if profile.get("enabled", False) and profile.get("backend") != decision.DECISION_BACKEND
        ]
    else:
        profile_names = [
            data.get("profile")
            or os.getenv("DEFAULT_MODEL_PROFILE")
            or config.get("default_profile")
            or "gemma3_4b_ollama_safe"
        ]

    if not profile_names:
        return make_error_response(error="No enabled profiles are available in config.json.", profile="all")

    started = time.perf_counter()
    results: List[Dict[str, Any]] = []
    for profile_name in profile_names:
        profile_results: List[Dict[str, Any]] = []
        for prompt in prompts:
            call_data = dict(data)
            call_data["profile"] = profile_name
            call_data["prompt"] = prompt
            call_data["images_base64"] = data.get("images_base64") or []
            result = run_local_request("vision" if call_data["images_base64"] else "chat", call_data)
            profile_results.append(
                {
                    "prompt": prompt[:120],
                    "success": result.get("success"),
                    "elapsed_seconds": result.get("elapsed_seconds"),
                    "response_length": len(str(result.get("response") or "")),
                    "warning": result.get("warning"),
                    "error": result.get("error"),
                }
            )
        successes = sum(1 for item in profile_results if item["success"])
        failures = len(profile_results) - successes
        elapsed_values = [float(item["elapsed_seconds"] or 0.0) for item in profile_results if item["success"]]
        average_elapsed = sum(elapsed_values) / len(elapsed_values) if elapsed_values else None
        results.append(
            {
                "profile": profile_name,
                "successes": successes,
                "failures": failures,
                "average_elapsed_seconds": round(average_elapsed, 3) if average_elapsed is not None else None,
                "tests": profile_results,
            }
        )

    elapsed = time.perf_counter() - started
    response = {
        "profiles_tested": profile_names,
        "results": results,
    }
    if len(profile_names) > 1:
        top_backend = "mixed"
    else:
        top_backend = (config.get("profiles", {}).get(profile_names[0]) or {}).get("backend")
    return model_dump_compat(
        LocalLLMResponse(
            success=any(item["successes"] > 0 for item in results),
            backend=top_backend,
            profile="all" if len(profile_names) > 1 else profile_names[0],
            model=None,
            response=response,
            elapsed_seconds=round(elapsed, 3),
            estimated_prompt_chars=sum(len(prompt) for prompt in prompts),
            estimated_prompt_tokens=estimate_tokens_from_chars(sum(len(prompt) for prompt in prompts)),
            context_limit_used=None,
            max_output_tokens_used=data.get("max_output_tokens"),
            warning=None,
            error=None,
        )
    )


def sanitize_profiles(config: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    for name, profile in config.get("profiles", {}).items():
        item = dict(profile)
        item["warning"] = profile_warning(item)
        # llama.cpp profiles reference a local GGUF path the browser can't
        # verify; report file presence so the UI can flag/hide dead profiles
        # the same way it does for un-pulled Ollama models.
        if str(item.get("backend", "")).lower() == "llamacpp":
            try:
                item["model_file_exists"] = Path(str(item.get("model_id") or "")).is_file()
            except OSError:
                item["model_file_exists"] = False
        sanitized[name] = item
    return sanitized


def openai_error_status(error: str) -> int:
    lowered = error.lower()
    if (
        "too large" in lowered
        or "exceeds configured limit" in lowered
        or "invalid" in lowered
        or "streaming is not implemented" in lowered
    ):
        return 400
    if "not running" in lowered or "unreachable" in lowered or "not reachable" in lowered:
        return 503
    if "not available" in lowered or "not found" in lowered or "disabled" in lowered:
        return 404
    return 502


def openai_error_response(result: Dict[str, Any]) -> JSONResponse:
    error = str(result.get("error") or "LocalDeploy request failed.")
    return JSONResponse(
        status_code=openai_error_status(error),
        content={
            "error": {
                "message": error,
                "type": "localdeploy_error",
                "param": None,
                "code": "localdeploy_request_failed",
            },
            "localdeploy": {
                "backend": result.get("backend"),
                "profile": result.get("profile"),
                "model": result.get("model"),
                "warning": result.get("warning"),
                "context_limit_used": result.get("context_limit_used"),
                "max_output_tokens_used": result.get("max_output_tokens_used"),
            },
        },
    )


def openai_sse_chunk(completion_id: str, model_name: str, delta: Dict[str, Any], finish_reason: Optional[str]) -> str:
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def openai_sse_error(completion_id: str, model_name: str, message: str) -> str:
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
        "error": {
            "message": message,
            "type": "localdeploy_error",
            "code": "localdeploy_request_failed",
        },
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def openai_stream_response(request_model: OpenAIChatCompletionRequest) -> StreamingResponse:
    completion_id = f"chatcmpl-localdeploy-{int(time.time() * 1000)}"
    model_name = request_model.model

    payload, has_images = openai_request_to_local_payload(request_model)
    prepared, error_response = prepare_request("vision" if has_images else "chat", payload, require_enabled=True)

    def emit_prepared() -> Iterator[str]:
        if error_response:
            yield openai_sse_error(completion_id, model_name, str(error_response.get("error") or "LocalDeploy request failed."))
            yield "data: [DONE]\n\n"
            return
        if prepared is None:
            yield openai_sse_error(completion_id, model_name, "Request preparation returned no request.")
            yield "data: [DONE]\n\n"
            return

        started = time.perf_counter()

        if prepared["backend"] == "llamacpp" and not prepared.get("tools"):
            try:
                content = call_llamacpp(prepared)
            except BackendCallError as exc:
                _note_monitor_request(prepared, prepared["kind"], False, time.perf_counter() - started, error=str(exc))
                yield openai_sse_error(completion_id, model_name, str(exc))
                yield "data: [DONE]\n\n"
                return
            except Exception as exc:
                _note_monitor_request(prepared, prepared["kind"], False, time.perf_counter() - started, error=str(exc))
                yield openai_sse_error(completion_id, model_name, f"Unexpected error: {exc}")
                yield "data: [DONE]\n\n"
                return
            _note_monitor_request(prepared, prepared["kind"], True, time.perf_counter() - started)
            yield openai_sse_chunk(completion_id, model_name, {"role": "assistant"}, None)
            if content:
                yield openai_sse_chunk(completion_id, model_name, {"content": content}, None)
            yield openai_sse_chunk(completion_id, model_name, {}, "stop")
            yield "data: [DONE]\n\n"
            return

        yield openai_sse_chunk(completion_id, model_name, {"role": "assistant"}, None)
        finish_reason = "stop"
        metrics_seen: Dict[str, Any] = {}
        try:
            events = (
                stream_ollama_events(prepared)
                if prepared["backend"] == "ollama"
                else stream_openai_compatible(prepared)
            )
            for event in events:
                if event.get("content"):
                    yield openai_sse_chunk(completion_id, model_name, {"content": event["content"]}, None)
                if event.get("tool_calls"):
                    finish_reason = "tool_calls"
                    streamed_tool_calls = (
                        normalize_tool_calls(event["tool_calls"])
                        if prepared["backend"] == "ollama"
                        else event["tool_calls"]
                    )
                    if prepared["backend"] == "ollama":
                        streamed_tool_calls = [
                            {**tool_call, "index": index}
                            for index, tool_call in enumerate(streamed_tool_calls)
                        ]
                    yield openai_sse_chunk(
                        completion_id, model_name, {"tool_calls": streamed_tool_calls}, None
                    )
                if event.get("done_reason") == "tool_calls":
                    finish_reason = "tool_calls"
                if isinstance(event.get("metrics"), dict):
                    metrics_seen.update(event["metrics"])
        except BackendCallError as exc:
            _note_monitor_request(prepared, prepared["kind"], False, time.perf_counter() - started, error=str(exc))
            yield openai_sse_error(completion_id, model_name, str(exc))
            yield "data: [DONE]\n\n"
            return
        except Exception as exc:
            # Last-resort safety net: any unexpected mid-stream failure (a dropped
            # connection, a malformed chunk) must still end the SSE stream cleanly
            # instead of crashing the generator and leaving the client hanging.
            _note_monitor_request(prepared, prepared["kind"], False, time.perf_counter() - started, error=str(exc))
            yield openai_sse_error(completion_id, model_name, f"Unexpected error: {exc}")
            yield "data: [DONE]\n\n"
            return
        _note_monitor_request(prepared, prepared["kind"], True, time.perf_counter() - started, details={"metrics": metrics_seen})
        yield openai_sse_chunk(completion_id, model_name, {}, finish_reason)
        yield "data: [DONE]\n\n"

    def emit() -> Iterator[str]:
        if error_response or prepared is None:
            yield from emit_prepared()
            return
        _note_monitor_request_started(prepared)
        try:
            yield from emit_prepared()
        finally:
            # Normal completion already decrements via _note_monitor_request;
            # this second, idempotent finish also covers client disconnects.
            try:
                from .control import monitor as monitor_mod

                monitor_mod.note_request_end(prepared.get("model"), prepared.get("_monitor_request_id"))
            except Exception:
                pass

    return StreamingResponse(emit(), media_type="text/event-stream")


def openai_chat_completion_response(request_model: OpenAIChatCompletionRequest, result: Dict[str, Any]) -> Dict[str, Any]:
    content = str(result.get("response") or "")
    tool_calls = normalize_tool_calls(result.get("tool_calls") or [])
    if request_model.response_format:
        content = normalize_structured_content(content)
    metrics = result.get("metrics") or {}
    prompt_tokens = int(
        metrics.get("prompt_eval_count")
        or result.get("estimated_prompt_tokens")
        or estimate_tokens_from_chars(result.get("estimated_prompt_chars") or 0)
    )
    completion_tokens = int(metrics.get("eval_count") or (estimate_tokens_from_chars(len(content)) if content else 0))
    return {
        "id": f"chatcmpl-localdeploy-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request_model.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content or None,
                    **({"tool_calls": tool_calls} if tool_calls else {}),
                },
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "localdeploy": {
            "backend": result.get("backend"),
            "profile": result.get("profile"),
            "model": result.get("model"),
            "elapsed_seconds": result.get("elapsed_seconds"),
            "warning": result.get("warning"),
            "context_limit_used": result.get("context_limit_used"),
            "max_output_tokens_used": result.get("max_output_tokens_used"),
            "metrics": result.get("metrics"),
        },
    }


def normalize_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for index, call in enumerate(tool_calls):
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments or {}, ensure_ascii=False)
        normalized.append(
            {
                "id": call.get("id") or f"call_localdeploy_{index}",
                "type": "function",
                "function": {"name": function.get("name"), "arguments": arguments},
            }
        )
    return normalized


def normalize_structured_content(content: str) -> str:
    stripped = content.strip()
    fence_match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.IGNORECASE | re.DOTALL)
    candidates = [fence_match.group(1).strip()] if fence_match else []
    candidates.append(stripped)

    object_match = re.search(r"(\{.*\})", stripped, flags=re.DOTALL)
    if object_match:
        candidates.append(object_match.group(1).strip())
    array_match = re.search(r"(\[.*\])", stripped, flags=re.DOTALL)
    if array_match:
        candidates.append(array_match.group(1).strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        return json.dumps(parsed, ensure_ascii=False)
    return content


app = FastAPI(title="LocalDeploy", version=localdeploy.__version__)


def _host_is_loopback(host: str) -> bool:
    return host.strip().lower() in {"127.0.0.1", "localhost", "::1"}


def check_lan_exposure() -> None:
    """Warn (or hard-fail under REQUIRE_TOKEN_ON_LAN) when API_HOST is bound to
    a non-loopback address without an API_TOKEN set. Runs at import time so it
    covers both `python api_server.py` and Docker's `uvicorn api_server:app`
    entry points, neither of which share a single call site otherwise.
    """
    host = os.getenv("API_HOST", "127.0.0.1")
    if _host_is_loopback(host):
        return
    token_configured = bool(api_token())
    in_docker = os.path.exists("/.dockerenv")
    lines = [
        "=" * 70,
        f"WARNING: API_HOST={host} is not loopback-only.",
    ]
    if in_docker:
        lines.append(
            "Inside Docker this is expected (the container always binds 0.0.0.0 "
            "internally) - what actually controls exposure is the *host* port "
            "mapping in docker-compose.yml. If you changed it to reach this from "
            "your LAN, return the host mapping to 127.0.0.1."
        )
    else:
        lines.append(
            "The control-plane (pull/delete/unload/set-default/benchmark) is "
            "reachable by network clients. LocalDeploy is a single-user local tool, not an internet-facing server."
        )
    if token_configured:
        lines.append(
            "API_TOKEN is configured, but it is one shared token and does not provide TLS or multi-user isolation."
        )
    else:
        lines.append("No API_TOKEN is configured; any reachable client can use the control plane.")
    if in_docker:
        lines.append("Do not expose the host port mapping beyond 127.0.0.1, and never put LocalDeploy behind a public tunnel.")
    else:
        lines.append("Bind API_HOST=127.0.0.1. Do not expose LocalDeploy directly to the internet or a public tunnel.")
    lines.append("=" * 70)
    message = "\n".join(lines)
    if env_bool("REQUIRE_TOKEN_ON_LAN", False) and not token_configured:
        raise SystemExit(message + "\nRefusing to start (REQUIRE_TOKEN_ON_LAN=true).")
    print(message, file=sys.stderr)


check_lan_exposure()

if env_bool("ENABLE_CORS", False):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost", "http://127.0.0.1", "http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _extract_token(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("x-api-token") or request.query_params.get("token", "")


@app.middleware("http")
async def api_token_guard(request: Request, call_next):
    """Opt-in auth: enforced only when API_TOKEN is set. Zero overhead otherwise.

    The static UI, health check, and CORS preflight stay open so the page can
    load and prompt for the token; every data endpoint requires it.
    """
    token = api_token()
    if token and request.method != "OPTIONS":
        path = request.url.path
        if not (
            path == "/health"
            or path == "/favicon.ico"
            or path == "/ui"
            or path.startswith("/ui/")
        ):
            if not hmac.compare_digest(_extract_token(request), token):
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": {
                            "message": "Unauthorized: provide the API token via the "
                            "'X-API-Token' header, 'Authorization: Bearer', or '?token='.",
                            "code": "unauthorized",
                        }
                    },
                )
    return await call_next(request)


@app.get("/health")
def health() -> Dict[str, Any]:
    config = load_config()
    ollama_base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    if require_gpu_only():
        models, ollama_error = [], "disabled by GPU-only mode"
        ollama_reachable = False
    else:
        models, ollama_error = ollama_models(ollama_base)
        ollama_reachable = ollama_error is None
    llama_base = os.getenv("LLAMACPP_BASE_URL", "http://127.0.0.1:8080")
    llama_status = llama_health(llama_base) if env_bool("ENABLE_LLAMA_CPP", False) else {"enabled": False}
    return {
        "success": True,
        "server": "ok",
        "config_path": str(get_config_path()),
        "default_profile": os.getenv("DEFAULT_MODEL_PROFILE") or config.get("default_profile"),
        "require_gpu_only": require_gpu_only(),
        "ollama": {
            "base_url": ollama_base,
            "reachable": ollama_reachable,
            "models": models,
            "error": ollama_error,
        },
        "llamacpp": llama_status,
        "decision": {"loaded": decision.status()},
        "limits": get_global_limits(),
    }


@app.get("/models")
def models() -> Dict[str, Any]:
    config = load_config()
    ollama_base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    if require_gpu_only():
        models_available, ollama_error = [], "disabled by GPU-only mode"
    else:
        models_available, ollama_error = ollama_models(ollama_base)
    llama_profiles = [
        {
            "profile": name,
            "model_id": profile.get("model_id"),
            "enabled": profile.get("enabled", False),
            "recommended_for_8gb_vram": profile.get("recommended_for_8gb_vram"),
            "quantization": profile.get("quantization"),
        }
        for name, profile in config.get("profiles", {}).items()
        if profile.get("backend") == "llamacpp"
    ]
    return {
        "success": True,
        "ollama": {
            "base_url": ollama_base,
            "models": models_available,
            "error": ollama_error,
        },
        "llamacpp_profiles": llama_profiles,
    }


@app.get("/v1/models")
def openai_models() -> Dict[str, Any]:
    config = load_config()
    data = []
    for name, profile in config.get("profiles", {}).items():
        if not profile.get("enabled", False):
            continue
        data.append(
            {
                "id": name,
                "object": "model",
                "created": 0,
                "owned_by": "localdeploy",
                "localdeploy": {
                    "backend": profile.get("backend"),
                    "kind": "decision" if profile.get("backend") == decision.DECISION_BACKEND else "chat",
                    "model_id": profile.get("model_id"),
                    "supports_vision": profile.get("supports_vision"),
                    "recommended_for_8gb_vram": profile.get("recommended_for_8gb_vram"),
                    "warning": profile_warning(profile),
                },
            }
        )
    return {"object": "list", "data": data}


@app.get("/profiles")
def profiles() -> Dict[str, Any]:
    config = load_config()
    return {
        "success": True,
        "default_profile": os.getenv("DEFAULT_MODEL_PROFILE") or config.get("default_profile"),
        "profiles": sanitize_profiles(config),
    }


@app.post("/estimate", response_model=LocalLLMResponse)
def estimate(request: EstimateRequest) -> Dict[str, Any]:
    return estimate_request_safety(model_dump_compat(request))


@app.post("/chat", response_model=LocalLLMResponse)
def chat(request: ChatRequest) -> Dict[str, Any]:
    return run_local_request("chat", model_dump_compat(request))


@app.post("/v1/chat/completions", response_model=None)
def openai_chat_completions(request: OpenAIChatCompletionRequest) -> Any:
    if request.stream:
        return openai_stream_response(request)
    payload, has_images = openai_request_to_local_payload(request)
    result = run_local_request("vision" if has_images else "chat", payload)
    if not result.get("success"):
        return openai_error_response(result)
    return openai_chat_completion_response(request, result)


@app.post("/v1/embeddings")
def openai_embeddings(request: OpenAIEmbeddingRequest) -> Any:
    values = [request.input] if isinstance(request.input, str) else request.input
    if not isinstance(values, list) or not values or not all(isinstance(value, str) for value in values):
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "input must be a string or non-empty list of strings.", "type": "invalid_request_error"}},
        )
    if request.encoding_format not in {"float", "base64"}:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "encoding_format must be 'float' or 'base64'.", "type": "invalid_request_error"}},
        )
    config = load_config()
    profile_name, model_override = profile_for_openai_model(config, request.model)
    resolved_name, profile, error = resolve_profile(
        config,
        {"profile": profile_name, **({"model": model_override} if model_override else {})},
        require_enabled=True,
    )
    if error or profile is None:
        return openai_error_response(
            make_error_response(error=error or "Profile resolution failed.", profile=resolved_name, model=request.model)
        )
    backend = str(profile.get("backend") or "ollama").lower()
    model = str(profile.get("model_id") or request.model)
    timeout = int(profile.get("timeout_seconds") or get_global_limits()["request_timeout_seconds"])
    try:
        if backend == "ollama":
            result = embed_ollama(str(profile["base_url"]), model, values, timeout)
            usage = {
                "prompt_tokens": int(result.get("prompt_eval_count") or sum(estimate_tokens_from_chars(len(v)) for v in values)),
                "total_tokens": int(result.get("prompt_eval_count") or sum(estimate_tokens_from_chars(len(v)) for v in values)),
            }
        else:
            result = embed_openai_compatible(profile, backend, model, values, timeout)
            usage = result.get("usage") or {}
    except BackendCallError as exc:
        return openai_error_response(make_error_response(error=str(exc), backend=backend, profile=resolved_name, model=model))
    embeddings = result.get("embeddings") or []
    data = []
    for index, vector in enumerate(embeddings):
        encoded = encode_embedding_base64(vector) if request.encoding_format == "base64" else vector
        data.append({"object": "embedding", "embedding": encoded, "index": index})
    return {"object": "list", "data": data, "model": request.model, "usage": usage}


def _decision_error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": "localdeploy_decision_error", "param": None, "code": code}},
    )


def _default_decision_profile(config: Dict[str, Any]) -> Optional[str]:
    configured = os.getenv("DEFAULT_DECISION_PROFILE") or config.get("default_decision_profile")
    if configured:
        return str(configured)
    for name, profile in config.get("profiles", {}).items():
        if profile.get("enabled", False) and profile.get("backend") == decision.DECISION_BACKEND:
            return name
    return None


@app.post("/v1/systemone")
def system_one(request: SystemOneRequest) -> Any:
    """Typed answers (with probabilities) to named questions about one state.

    Wire-compatible with the System One request/response shape: ``state`` plus
    ``questions`` in, ``answers`` plus ``usage`` out. ``model`` (or ``profile``)
    names a decision profile; without one, the default decision profile is used.
    """
    config = load_config()
    requested = request.profile or request.model
    profiles = config.get("profiles", {})
    if requested and requested not in profiles:
        requested = next((n for n, p in profiles.items() if str(p.get("model_id")) == requested), requested)
    name = requested or _default_decision_profile(config)
    if not name:
        return _decision_error(
            404, "no_decision_profile",
            "No enabled decision profile. Enable one in config.json (see docs/API_OPTIONS.md, 'Decision models').",
        )
    resolved_name, profile, error = resolve_profile(config, {"profile": name}, require_enabled=True)
    if error or profile is None:
        return _decision_error(404, "profile_unavailable", error or f"Profile '{name}' could not be resolved.")
    if profile.get("backend") != decision.DECISION_BACKEND:
        return _decision_error(
            400, "invalid_request",
            f"Invalid request: profile '{resolved_name}' is not a decision model. Use /v1/chat/completions for it.",
        )
    try:
        result = decision.decide(resolved_name, profile, request.state, request.questions)
    except decision.DecisionRequestError as exc:
        return _decision_error(400, "invalid_request", f"Invalid request: {exc}")
    except decision.DecisionUnavailable as exc:
        return _decision_error(503, "decision_unavailable", str(exc))
    except BackendCallError as exc:
        return _decision_error(502, "decision_failed", str(exc))
    body = {
        "answers": result["answers"],
        "model": resolved_name,
        "usage": result.get("usage") or {},
        "latency_ms": result["latency_ms"],
    }
    if "load_ms" in result:
        body["load_ms"] = result["load_ms"]
    return body


def _responses_chat_request(request: OpenAIResponseRequest) -> OpenAIChatCompletionRequest:
    messages: List[Dict[str, Any]] = []
    if request.instructions:
        messages.append({"role": "system", "content": request.instructions})
    if isinstance(request.input, str):
        messages.append({"role": "user", "content": request.input})
    elif isinstance(request.input, list):
        for item in request.input:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "user")
            content = item.get("content")
            if isinstance(content, list):
                translated = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    value = dict(part)
                    if value.get("type") == "input_text":
                        value["type"] = "text"
                    elif value.get("type") == "input_image":
                        value["type"] = "image_url"
                        value["image_url"] = value.get("image_url") or value.get("image")
                    translated.append(value)
                content = translated
            if item.get("type") == "function_call_output":
                role = "tool"
                content = item.get("output") or ""
            messages.append(
                {
                    "role": role,
                    "content": content,
                    "tool_call_id": item.get("call_id"),
                    "name": item.get("name"),
                }
            )
    tools = []
    for tool in request.tools or []:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "function" and "function" not in tool:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.get("name"),
                        "description": tool.get("description"),
                        "parameters": tool.get("parameters") or {},
                    },
                }
            )
        else:
            tools.append(tool)
    return OpenAIChatCompletionRequest(
        model=request.model,
        profile=request.profile,
        messages=[OpenAIChatMessage(**item) for item in messages],
        tools=tools or None,
        tool_choice=request.tool_choice,
        max_tokens=request.max_output_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    )


def _responses_payload(
    request: OpenAIResponseRequest, result: Dict[str, Any], response_id: Optional[str] = None
) -> Dict[str, Any]:
    response_id = response_id or f"resp_localdeploy_{int(time.time() * 1000)}"
    output: List[Dict[str, Any]] = []
    content = str(result.get("response") or "")
    if content:
        output.append(
            {
                "type": "message",
                "id": f"msg_{int(time.time() * 1000)}",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": content, "annotations": []}],
            }
        )
    for index, tool_call in enumerate(normalize_tool_calls(result.get("tool_calls") or [])):
        function = tool_call.get("function") or {}
        output.append(
            {
                "type": "function_call",
                "id": tool_call.get("id") or f"fc_{index}",
                "call_id": tool_call.get("id") or f"call_{index}",
                "name": function.get("name"),
                "arguments": function.get("arguments") or "{}",
                "status": "completed",
            }
        )
    metrics = result.get("metrics") or {}
    input_tokens = int(
        metrics.get("prompt_eval_count")
        or metrics.get("prompt_tokens")
        or result.get("estimated_prompt_tokens")
        or 0
    )
    output_tokens = int(
        metrics.get("eval_count")
        or metrics.get("completion_tokens")
        or estimate_tokens_from_chars(len(content))
    )
    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": request.model,
        "output": output,
        "output_text": content,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens},
        "error": None,
        "localdeploy": {
            "backend": result.get("backend"),
            "profile": result.get("profile"),
            "model": result.get("model"),
            "metrics": metrics,
        },
    }


def _responses_sse(event_type: str, payload: Dict[str, Any]) -> str:
    data = {"type": event_type, **payload}
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _merge_tool_call_deltas(
    state: Dict[int, Dict[str, Any]], tool_calls: List[Dict[str, Any]]
) -> None:
    for position, raw in enumerate(tool_calls):
        if not isinstance(raw, dict):
            continue
        try:
            index = int(raw.get("index", position))
        except (TypeError, ValueError):
            index = position
        item = state.setdefault(
            index,
            {"id": raw.get("id") or f"call_localdeploy_{index}", "type": "function", "function": {}},
        )
        if raw.get("id"):
            item["id"] = raw["id"]
        function = raw.get("function") if isinstance(raw.get("function"), dict) else {}
        if function.get("name"):
            item["function"]["name"] = function["name"]
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            item["function"]["arguments"] = item["function"].get("arguments", "") + arguments
        elif arguments is not None:
            item["function"]["arguments"] = json.dumps(arguments, ensure_ascii=False)


def _responses_stream_response(
    request: OpenAIResponseRequest, chat_request: OpenAIChatCompletionRequest
) -> StreamingResponse:
    response_id = f"resp_localdeploy_{int(time.time() * 1000)}"
    payload, has_images = openai_request_to_local_payload(chat_request)
    prepared, error_response = prepare_request(
        "vision" if has_images else "chat", payload, require_enabled=True
    )

    def emit_prepared() -> Iterator[str]:
        started_response = {
            "id": response_id,
            "object": "response",
            "created_at": int(time.time()),
            "status": "in_progress",
            "model": request.model,
            "output": [],
            "output_text": "",
            "error": None,
        }
        yield _responses_sse("response.created", {"response": started_response})
        yield _responses_sse("response.in_progress", {"response": started_response})
        if error_response or prepared is None:
            message = str((error_response or {}).get("error") or "Request preparation failed.")
            failed = {**started_response, "status": "failed", "error": {"message": message}}
            yield _responses_sse("response.failed", {"response": failed})
            return

        started = time.perf_counter()
        content_chunks: List[str] = []
        tool_state: Dict[int, Dict[str, Any]] = {}
        metrics: Dict[str, Any] = {}
        message_started = False
        message_id = f"msg_{int(time.time() * 1000)}"
        try:
            if prepared["backend"] == "llamacpp" and not prepared.get("tools"):
                backend_events: Iterator[Dict[str, Any]] = iter(
                    [{"content": call_llamacpp(prepared), "tool_calls": [], "metrics": {}}]
                )
            elif prepared["backend"] == "ollama":
                backend_events = stream_ollama_events(prepared)
            else:
                backend_events = stream_openai_compatible(prepared)

            for event in backend_events:
                chunk = str(event.get("content") or "")
                if chunk:
                    if not message_started:
                        message_started = True
                        item = {
                            "type": "message",
                            "id": message_id,
                            "role": "assistant",
                            "status": "in_progress",
                            "content": [],
                        }
                        yield _responses_sse(
                            "response.output_item.added", {"output_index": 0, "item": item}
                        )
                        yield _responses_sse(
                            "response.content_part.added",
                            {
                                "item_id": message_id,
                                "output_index": 0,
                                "content_index": 0,
                                "part": {"type": "output_text", "text": "", "annotations": []},
                            },
                        )
                    content_chunks.append(chunk)
                    yield _responses_sse(
                        "response.output_text.delta",
                        {
                            "item_id": message_id,
                            "output_index": 0,
                            "content_index": 0,
                            "delta": chunk,
                        },
                    )
                if event.get("tool_calls"):
                    _merge_tool_call_deltas(tool_state, event["tool_calls"])
                if isinstance(event.get("metrics"), dict):
                    metrics.update(event["metrics"])
        except Exception as exc:  # noqa: BLE001 - preserve the SSE error contract
            _note_monitor_request(prepared, prepared["kind"], False, time.perf_counter() - started, error=str(exc))
            failed = {**started_response, "status": "failed", "error": {"message": str(exc)}}
            yield _responses_sse("response.failed", {"response": failed})
            return

        _note_monitor_request(prepared, prepared["kind"], True, time.perf_counter() - started, details={"metrics": metrics})
        content = "".join(content_chunks)
        result = {
            "success": True,
            "backend": prepared["backend"],
            "profile": prepared["profile_name"],
            "model": prepared["model"],
            "response": content,
            "tool_calls": [tool_state[index] for index in sorted(tool_state)],
            "metrics": metrics,
            "estimated_prompt_tokens": prepared["estimated_prompt_tokens"],
        }
        completed = _responses_payload(request, result, response_id=response_id)
        for output_index, item in enumerate(completed["output"]):
            if item["type"] == "message":
                yield _responses_sse(
                    "response.output_text.done",
                    {
                        "item_id": item["id"],
                        "output_index": output_index,
                        "content_index": 0,
                        "text": content,
                    },
                )
                yield _responses_sse(
                    "response.content_part.done",
                    {
                        "item_id": item["id"],
                        "output_index": output_index,
                        "content_index": 0,
                        "part": item["content"][0],
                    },
                )
            else:
                yield _responses_sse(
                    "response.output_item.added", {"output_index": output_index, "item": item}
                )
                yield _responses_sse(
                    "response.function_call_arguments.done",
                    {
                        "item_id": item["id"],
                        "output_index": output_index,
                        "arguments": item["arguments"],
                    },
                )
            yield _responses_sse(
                "response.output_item.done", {"output_index": output_index, "item": item}
            )
        yield _responses_sse("response.completed", {"response": completed})

    def emit() -> Iterator[str]:
        if error_response or prepared is None:
            yield from emit_prepared()
            return
        _note_monitor_request_started(prepared)
        try:
            yield from emit_prepared()
        finally:
            try:
                from .control import monitor as monitor_mod

                monitor_mod.note_request_end(prepared.get("model"), prepared.get("_monitor_request_id"))
            except Exception:
                pass

    return StreamingResponse(emit(), media_type="text/event-stream")


@app.post("/v1/responses", response_model=None)
def openai_responses(request: OpenAIResponseRequest) -> Any:
    chat_request = _responses_chat_request(request)
    if request.stream:
        return _responses_stream_response(request, chat_request)
    payload, has_images = openai_request_to_local_payload(chat_request)
    result = run_local_request("vision" if has_images else "chat", payload)
    if not result.get("success"):
        return openai_error_response(result)
    response = _responses_payload(request, result)
    return response


def parse_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        if lowered.isdigit():
            try:
                return int(lowered)
            except ValueError:
                return value
        try:
            return float(value)
        except ValueError:
            return value
    return value


@app.post("/vision", response_model=LocalLLMResponse)
async def vision(request: Request) -> Dict[str, Any]:
    content_type = request.headers.get("content-type", "")
    try:
        if "multipart/form-data" in content_type:
            form = await request.form()
            data: Dict[str, Any] = {}
            images: List[str] = []
            for key, value in form.multi_items():
                if hasattr(value, "read"):
                    file_bytes = await value.read()
                    images.append(base64.b64encode(file_bytes).decode("ascii"))
                elif key == "images_base64":
                    images.append(str(value))
                else:
                    data[key] = parse_scalar(value)
            data["images_base64"] = images
        else:
            data = await request.json()
    except Exception as exc:
        return make_error_response(error=f"Invalid vision request body: {exc}")

    return run_local_request("vision", data)


@app.post("/benchmark", response_model=LocalLLMResponse)
def benchmark(request: BenchmarkRequest) -> Dict[str, Any]:
    return run_benchmark(model_dump_compat(request))


# Web UI and control-plane endpoints are additive and opt-out via ENABLE_WEB_UI.
# When disabled, the server behaves exactly as it did before this package existed.
if enable_web_ui():
    from fastapi.responses import FileResponse

    from localdeploy.control import router as web_router

    app.include_router(web_router)
    app.mount("/ui", StaticFiles(directory=str(web_dir()), html=True), name="ui")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> FileResponse:
        # Browsers request this on /docs and the API root; without it every
        # page view logs a 404.
        return FileResponse(web_dir() / "favicon.png", media_type="image/png")


def serve() -> None:
    """Run the API with uvicorn (used by `python api_server.py` and the shim)."""
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=env_int("API_PORT", 8000),
    )


if __name__ == "__main__":
    serve()
