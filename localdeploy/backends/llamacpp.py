from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import requests

from . import llamacpp_managed
from ..utils import (
    BackendCallError,
    get_backend_base_url,
    is_loopback_url,
    strip_trailing_slash,
)


def gguf_path_exists(model_id: str) -> bool:
    if not model_id.lower().endswith(".gguf"):
        return True
    return Path(model_id).expanduser().exists()


def llama_cpp_error_message(response: requests.Response) -> str:
    body = response.text.strip()
    try:
        parsed = response.json()
        body = parsed.get("error") or parsed.get("message") or body
    except Exception:
        pass
    lowered = str(body).lower()
    if "out of memory" in lowered or "cuda" in lowered or "memory" in lowered:
        return f"llama.cpp reported a memory-related failure. Lower context/output limits or reduce GPU layers. Details: {body}"
    return f"llama.cpp request failed with HTTP {response.status_code}. Details: {body}"


def llama_completion_prompt(system_prompt: str, prompt: str) -> str:
    parts: List[str] = []
    if system_prompt:
        parts.append(f"System:\n{system_prompt}")
    parts.append(f"User:\n{prompt}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def call_llamacpp(prepared: Dict[str, Any]) -> str:
    profile = prepared["profile"]
    model_id = prepared["model"]
    if not gguf_path_exists(model_id):
        raise BackendCallError(f"GGUF file path not found: {model_id}. Update config.json or start the matching local llama-server.")

    if llamacpp_managed.is_managed(profile):
        # LocalDeploy starts (or switches to) this profile's llama-server itself and forwards to it.
        base_url = llamacpp_managed.ensure(str(profile.get("name") or prepared.get("profile_name") or "managed"), profile)
    else:
        base_url = get_backend_base_url(profile, "llamacpp")
    messages: List[Dict[str, str]] = []
    if prepared["system_prompt"]:
        messages.append({"role": "system", "content": prepared["system_prompt"]})
    messages.append({"role": "user", "content": prepared["prompt"]})

    openai_payload: Dict[str, Any] = {
        "model": Path(model_id).stem if model_id.lower().endswith(".gguf") else model_id,
        "messages": messages,
        "temperature": prepared["temperature"],
        "top_p": prepared["top_p"],
        "max_tokens": prepared["max_output_tokens_used"],
        "stream": False,
    }
    if prepared.get("repeat_penalty") is not None:
        openai_payload["repeat_penalty"] = prepared["repeat_penalty"]
    if prepared.get("response_format"):
        openai_payload["response_format"] = prepared["response_format"]

    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=openai_payload,
            timeout=prepared["timeout_seconds"],
        )
    except requests.Timeout as exc:
        raise BackendCallError(
            f"llama.cpp request timed out after {prepared['timeout_seconds']} seconds. Lower context/output limits or GPU layers."
        ) from exc
    except requests.ConnectionError as exc:
        raise BackendCallError(f"llama.cpp server is not running or is unreachable at {base_url}.") from exc

    if response.status_code != 404:
        if not response.ok:
            raise BackendCallError(llama_cpp_error_message(response))
        try:
            data = response.json()
        except Exception as exc:
            raise BackendCallError(f"llama.cpp returned invalid JSON: {response.text[:500]}") from exc
        try:
            return str(data["choices"][0]["message"]["content"])
        except Exception:
            return json.dumps(data, ensure_ascii=False)

    completion_payload = {
        "prompt": llama_completion_prompt(prepared["system_prompt"], prepared["prompt"]),
        "n_predict": prepared["max_output_tokens_used"],
        "temperature": prepared["temperature"],
        "top_p": prepared["top_p"],
        "repeat_penalty": prepared["repeat_penalty"],
        "stream": False,
    }
    try:
        response = requests.post(
            f"{base_url}/completion",
            json=completion_payload,
            timeout=prepared["timeout_seconds"],
        )
    except requests.Timeout as exc:
        raise BackendCallError(
            f"llama.cpp completion timed out after {prepared['timeout_seconds']} seconds. Lower context/output limits."
        ) from exc
    except requests.ConnectionError as exc:
        raise BackendCallError(f"llama.cpp server is not running or is unreachable at {base_url}.") from exc

    if not response.ok:
        raise BackendCallError(llama_cpp_error_message(response))

    try:
        data = response.json()
    except Exception as exc:
        raise BackendCallError(f"llama.cpp returned invalid JSON: {response.text[:500]}") from exc
    if "content" in data:
        return str(data.get("content", ""))
    if "response" in data:
        return str(data.get("response", ""))
    return json.dumps(data, ensure_ascii=False)


def llama_health(base_url: str) -> Dict[str, Any]:
    if not is_loopback_url(base_url):
        return {"reachable": False, "error": f"Refusing non-local llama.cpp URL: {base_url}"}
    base_url = strip_trailing_slash(base_url)
    for path in ("/health", "/v1/models"):
        try:
            response = requests.get(f"{base_url}{path}", timeout=2)
            if response.ok:
                return {"reachable": True, "endpoint": f"{base_url}{path}"}
        except requests.ConnectionError:
            return {"reachable": False, "error": f"llama.cpp is not reachable at {base_url}."}
        except requests.Timeout:
            return {"reachable": False, "error": f"llama.cpp health check timed out at {base_url}."}
    return {"reachable": False, "error": f"llama.cpp did not answer /health or /v1/models at {base_url}."}
