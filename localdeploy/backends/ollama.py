from __future__ import annotations

import base64
import json
import struct
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

import requests

from ..utils import (
    BackendCallError,
    get_backend_base_url,
    is_loopback_url,
    require_gpu_only,
    strip_trailing_slash,
)


def ollama_format(response_format: Any) -> Any:
    """Translate OpenAI response_format into Ollama's native `format` value."""
    if not isinstance(response_format, dict):
        return None
    format_type = response_format.get("type")
    if format_type == "json_object":
        return "json"
    if format_type == "json_schema":
        wrapper = response_format.get("json_schema")
        if isinstance(wrapper, dict) and isinstance(wrapper.get("schema"), dict):
            return wrapper["schema"]
    return None


def options_payload(prepared: Dict[str, Any]) -> Dict[str, Any]:
    options: Dict[str, Any] = {
        "num_ctx": prepared["context_limit_used"],
        "num_predict": prepared["max_output_tokens_used"],
    }
    for key in ("temperature", "top_p", "repeat_penalty"):
        if prepared.get(key) is not None:
            options[key] = prepared[key]
    # Optional CPU/GPU placement: num_gpu is the number of layers offloaded to GPU
    # (0 = force CPU). Absent -> Ollama's default (auto), preserving prior behaviour.
    if prepared.get("num_gpu") is not None:
        options["num_gpu"] = prepared["num_gpu"]
    return options


def _message_content_for_ollama(content: Any) -> Tuple[str, List[str]]:
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        return str(content or ""), []
    text: List[str] = []
    images: List[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") in {"text", "input_text"}:
            text.append(str(part.get("text") or ""))
        elif part.get("type") in {"image_url", "input_image"}:
            image = part.get("image_url") or part.get("image")
            url = image.get("url") if isinstance(image, dict) else image
            if isinstance(url, str):
                images.append(url.split(",", 1)[1] if url.startswith("data:") and "," in url else url)
    return "\n".join(value for value in text if value), images


def _ollama_messages(prepared: Dict[str, Any]) -> List[Dict[str, Any]]:
    supplied = prepared.get("messages") or []
    if supplied:
        messages: List[Dict[str, Any]] = []
        tool_names: Dict[str, str] = {}
        for raw in supplied:
            if not isinstance(raw, dict):
                continue
            content, images = _message_content_for_ollama(raw.get("content"))
            message: Dict[str, Any] = {"role": str(raw.get("role") or "user"), "content": content}
            if images:
                message["images"] = images
            if raw.get("tool_calls"):
                normalized_calls = []
                for call in raw["tool_calls"]:
                    if not isinstance(call, dict):
                        continue
                    normalized = dict(call)
                    function = dict(call.get("function") or {})
                    if isinstance(function.get("arguments"), str):
                        try:
                            function["arguments"] = json.loads(function["arguments"])
                        except ValueError:
                            pass
                    normalized["function"] = function
                    normalized_calls.append(normalized)
                    if call.get("id") and function.get("name"):
                        tool_names[str(call["id"])] = str(function["name"])
                message["tool_calls"] = normalized_calls
            if raw.get("tool_call_id"):
                call_id = str(raw.get("tool_call_id"))
                message["tool_name"] = raw.get("name") or tool_names.get(call_id) or call_id
            messages.append(message)
        return messages

    messages = []
    if prepared["system_prompt"]:
        messages.append({"role": "system", "content": prepared["system_prompt"]})
    user_message: Dict[str, Any] = {"role": "user", "content": prepared["prompt"]}
    if prepared["images_base64"]:
        user_message["images"] = prepared["images_base64"]
    messages.append(user_message)
    return messages


def _selected_tools(prepared: Dict[str, Any]) -> List[Dict[str, Any]]:
    tools = [item for item in (prepared.get("tools") or []) if isinstance(item, dict)]
    choice = prepared.get("tool_choice")
    if choice == "none":
        return []
    if isinstance(choice, dict):
        function = choice.get("function") or {}
        selected_name = function.get("name")
        if selected_name:
            selected = [item for item in tools if (item.get("function") or {}).get("name") == selected_name]
            return selected or tools
    return tools


def ollama_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    def seconds(name: str) -> Optional[float]:
        value = data.get(name)
        return round(float(value) / 1_000_000_000, 6) if value is not None else None

    eval_count = _safe_int(data.get("eval_count"))
    prompt_count = _safe_int(data.get("prompt_eval_count"))
    load_seconds = seconds("load_duration")
    eval_seconds = seconds("eval_duration")
    prompt_seconds = seconds("prompt_eval_duration")
    # Time to first token: Ollama's non-streaming /api/chat response has no
    # explicit first-token timestamp, but load_duration + prompt_eval_duration
    # IS the wall-clock time before the first generated token - the model
    # finishes loading (if cold), processes the prompt, then starts emitting.
    # Present whenever at least one of the two components is known.
    ttft_ms = None
    if load_seconds is not None or prompt_seconds is not None:
        ttft_ms = round(((load_seconds or 0.0) + (prompt_seconds or 0.0)) * 1000, 1)
    return {
        "total_duration_seconds": seconds("total_duration"),
        "load_duration_seconds": load_seconds,
        "prompt_eval_count": prompt_count,
        "prompt_eval_duration_seconds": prompt_seconds,
        "prompt_tokens_per_second": round(prompt_count / prompt_seconds, 3) if prompt_count and prompt_seconds else None,
        "eval_count": eval_count,
        "eval_duration_seconds": eval_seconds,
        "tokens_per_second": round(eval_count / eval_seconds, 3) if eval_count and eval_seconds else None,
        "ttft_ms": ttft_ms,
    }


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def ollama_error_message(response: requests.Response, model_id: str) -> str:
    body = response.text.strip()
    try:
        parsed = response.json()
        body = parsed.get("error") or body
    except Exception:
        pass
    lowered = body.lower()
    if response.status_code == 404 or "not found" in lowered or "pull" in lowered:
        return f"Ollama model '{model_id}' is not available. Run: ollama pull {model_id}. Details: {body}"
    if "out of memory" in lowered or "cuda" in lowered or "memory" in lowered:
        return f"Ollama reported a memory-related failure. Lower context/output limits or use a smaller profile. Details: {body}"
    return f"Ollama request failed with HTTP {response.status_code}. Details: {body}"


def call_ollama_detailed(prepared: Dict[str, Any]) -> Dict[str, Any]:
    if require_gpu_only():
        raise BackendCallError("GPU-only mode is enabled; refusing to call Ollama.")
    profile = prepared["profile"]
    base_url = get_backend_base_url(profile, "ollama")
    payload = {
        "model": prepared["model"],
        "messages": _ollama_messages(prepared),
        "stream": False,
        "options": options_payload(prepared),
    }
    if prepared.get("keep_alive"):
        payload["keep_alive"] = prepared["keep_alive"]
    tools = _selected_tools(prepared)
    if tools:
        payload["tools"] = tools
    native_format = ollama_format(prepared.get("response_format"))
    if native_format is not None:
        payload["format"] = native_format
    # Thinking models (Qwen3, DeepSeek-R1) put reasoning into a separate `thinking`
    # field; `content` is empty until the model finishes. `think: false` disables
    # the reasoning pass entirely for cases where we only want the final answer.
    if profile.get("think") is not None:
        payload["think"] = bool(profile["think"])
    try:
        response = requests.post(f"{base_url}/api/chat", json=payload, timeout=prepared["timeout_seconds"])
    except requests.Timeout as exc:
        raise BackendCallError(
            f"Ollama request timed out after {prepared['timeout_seconds']} seconds. Lower context/output limits or try a smaller profile."
        ) from exc
    except requests.ConnectionError as exc:
        raise BackendCallError(f"Ollama is not running or is unreachable at {base_url}. Start Ollama and retry.") from exc

    if not response.ok:
        raise BackendCallError(ollama_error_message(response, prepared["model"]))

    try:
        data = response.json()
    except Exception as exc:
        raise BackendCallError(f"Ollama returned invalid JSON: {response.text[:500]}") from exc

    message = data.get("message") if isinstance(data.get("message"), dict) else {}
    content = str(message.get("content", data.get("response", "")))
    return {
        "content": content,
        "tool_calls": message.get("tool_calls") or [],
        "metrics": ollama_metrics(data),
        "done_reason": data.get("done_reason"),
    }


def call_ollama(prepared: Dict[str, Any]) -> str:
    return str(call_ollama_detailed(prepared).get("content") or "")


def stream_ollama_events(prepared: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Yield native content/tool/metrics events from an Ollama chat call."""
    if require_gpu_only():
        raise BackendCallError("GPU-only mode is enabled; refusing to call Ollama.")
    profile = prepared["profile"]
    base_url = get_backend_base_url(profile, "ollama")
    payload = {
        "model": prepared["model"],
        "messages": _ollama_messages(prepared),
        "stream": True,
        "options": options_payload(prepared),
    }
    if prepared.get("keep_alive"):
        payload["keep_alive"] = prepared["keep_alive"]
    tools = _selected_tools(prepared)
    if tools:
        payload["tools"] = tools
    native_format = ollama_format(prepared.get("response_format"))
    if native_format is not None:
        payload["format"] = native_format
    if profile.get("think") is not None:
        payload["think"] = bool(profile["think"])
    try:
        response = requests.post(
            f"{base_url}/api/chat",
            json=payload,
            timeout=prepared["timeout_seconds"],
            stream=True,
        )
    except requests.Timeout as exc:
        raise BackendCallError(
            f"Ollama request timed out after {prepared['timeout_seconds']} seconds. Lower context/output limits or try a smaller profile."
        ) from exc
    except requests.ConnectionError as exc:
        raise BackendCallError(f"Ollama is not running or is unreachable at {base_url}. Start Ollama and retry.") from exc

    # Use the response as a context manager so the underlying connection is
    # released even if the SSE consumer disconnects mid-stream (GeneratorExit).
    with response:
        if not response.ok:
            raise BackendCallError(ollama_error_message(response, prepared["model"]))

        try:
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                message = data.get("message") if isinstance(data.get("message"), dict) else {}
                event = {
                    "content": str(message.get("content", data.get("response", ""))),
                    "tool_calls": message.get("tool_calls") or [],
                    "done": bool(data.get("done")),
                    "done_reason": data.get("done_reason"),
                    "metrics": ollama_metrics(data) if data.get("done") else None,
                }
                if event["content"] or event["tool_calls"] or event["done"]:
                    yield event
                if data.get("done"):
                    break
        except requests.exceptions.RequestException as exc:
            raise BackendCallError(f"Ollama connection was lost mid-response: {exc}") from exc


def stream_ollama(prepared: Dict[str, Any]) -> Iterator[str]:
    """Backward-compatible content-only stream."""
    for event in stream_ollama_events(prepared):
        if event.get("content"):
            yield str(event["content"])


def embed_ollama(base_url: str, model: str, inputs: List[str], timeout: int = 180) -> Dict[str, Any]:
    if require_gpu_only():
        raise BackendCallError("GPU-only mode is enabled; refusing to call Ollama.")
    if not is_loopback_url(base_url):
        raise BackendCallError(f"Refusing non-local Ollama URL: {base_url}")
    base_url = strip_trailing_slash(base_url)
    try:
        response = requests.post(
            f"{base_url}/api/embed",
            json={"model": model, "input": inputs, "truncate": True},
            timeout=timeout,
        )
    except requests.Timeout as exc:
        raise BackendCallError(f"Ollama embedding request timed out after {timeout} seconds.") from exc
    except requests.ConnectionError as exc:
        raise BackendCallError(f"Ollama is not running or is unreachable at {base_url}.") from exc
    if response.status_code == 404:
        embeddings = []
        counts = 0
        for value in inputs:
            try:
                fallback = requests.post(
                    f"{base_url}/api/embeddings",
                    json={"model": model, "prompt": value},
                    timeout=timeout,
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                raise BackendCallError(f"Ollama legacy embedding request failed at {base_url}.") from exc
            if not fallback.ok:
                raise BackendCallError(ollama_error_message(fallback, model))
            try:
                body = fallback.json()
            except ValueError as exc:
                raise BackendCallError("Ollama returned invalid JSON for a legacy embedding request.") from exc
            embeddings.append(body.get("embedding") or [])
            counts += max(1, len(value) // 4)
        return {"embeddings": embeddings, "prompt_eval_count": counts, "metrics": {}}
    if not response.ok:
        raise BackendCallError(ollama_error_message(response, model))
    try:
        data = response.json()
    except ValueError as exc:
        raise BackendCallError("Ollama returned invalid JSON for an embedding request.") from exc
    return {
        "embeddings": data.get("embeddings") or [],
        "prompt_eval_count": _safe_int(data.get("prompt_eval_count")),
        "metrics": ollama_metrics(data),
    }


def encode_embedding_base64(values: List[float]) -> str:
    packed = struct.pack(f"<{len(values)}f", *(float(value) for value in values))
    return base64.b64encode(packed).decode("ascii")


#: How long a look at Ollama is trusted by /health. Callers poll /health (the console on every page, `ybm status`, launchers), and an
#: Ollama that is not running used to cost every one of those calls its timeout.
OLLAMA_PROBE_TTL_SECONDS = 10.0
_probe_lock = threading.Lock()
_probe_cache: Dict[str, Tuple[float, Tuple[List[str], Optional[str]]]] = {}


def ollama_models_cached(base_url: str) -> Tuple[List[str], Optional[str]]:
    """`ollama_models`, remembered for a few seconds, so that a health check does not make a request (or wait for a refusal) each time."""
    now = time.monotonic()
    with _probe_lock:
        cached = _probe_cache.get(base_url)
        if cached is not None and now - cached[0] < OLLAMA_PROBE_TTL_SECONDS:
            return cached[1]
    result = ollama_models(base_url)
    with _probe_lock:
        _probe_cache[base_url] = (time.monotonic(), result)
    return result


def ollama_models(base_url: str) -> Tuple[List[str], Optional[str]]:
    if not is_loopback_url(base_url):
        return [], f"Refusing non-local Ollama URL: {base_url}"
    try:
        # A connection to a loopback port nobody listens on is refused at once on Linux and macOS but takes about two seconds on Windows,
        # so the connect step gets half a second: a local Ollama that is running accepts immediately.
        response = requests.get(f"{strip_trailing_slash(base_url)}/api/tags", timeout=(0.5, 3))
        response.raise_for_status()
        data = response.json()
        return [item.get("name", "") for item in data.get("models", []) if item.get("name")], None
    except requests.ConnectionError:
        return [], f"Ollama is not reachable at {base_url}."
    except Exception as exc:
        return [], str(exc)
