"""Decision models ("System One"): answer typed questions about a piece of text.

A decision model does not write text. It reads one *state* and a set of named
*questions* and returns, for each question, a typed answer with a probability per
option, in a single forward pass. LocalDeploy serves them at ``POST /v1/systemone``
so an application can ask "is this a task or chat, and which route?" in about a
tenth of a second on a GPU and fall back to a language model only when the answer
is not confident.

Three adapters share one request and response shape:

* ``d1``             LiquidAI d1 models (``AutoModel`` with ``system_one``)
* ``laya``           Laya checkpoints (``laya.Router``)
* ``systemone-http`` any other server that already speaks ``/v1/systemone``
                     (``laya-serve``, ollaya, ...), loopback only like every backend

The heavy packages (torch, transformers, laya) are optional and imported only
when a decision profile is first used: ``pip install -r requirements-decision.txt``.
"""

from __future__ import annotations

import os
import re
import threading
import time
from typing import Any, Dict, List, Tuple

import requests

from ..utils import BackendCallError, is_loopback_url, offline_mode, strip_trailing_slash

DECISION_BACKEND = "decision"
ADAPTER_NAMES = ("d1", "laya", "systemone-http")
QUESTION_TYPES = ("choice", "score", "noul")

MAX_QUESTIONS = 32
MAX_CHOICE_OPTIONS = 100
MAX_SCORE_LEVELS = 10
MAX_STATE_CHARS = 30000
_NAME = re.compile(r"^[\w.\-]{1,64}$")

INSTALL_HINT = (
    "Decision models need optional packages. Install them into LocalDeploy's environment with: "
    "python -m pip install -r requirements-decision.txt"
)


class DecisionRequestError(ValueError):
    """The request itself is wrong (HTTP 400)."""


class DecisionUnavailable(BackendCallError):
    """The profile cannot be used yet: packages, license or model files missing (HTTP 503)."""


class DecisionError(BackendCallError):
    """The model or the remote server failed while answering (HTTP 502)."""


# --- request validation -----------------------------------------------------


def validate_request(state: Any, questions: Any, max_state_chars: int = MAX_STATE_CHARS) -> None:
    """Reject a malformed request before any model is loaded."""
    if isinstance(state, str):
        size = len(state)
    else:
        try:
            import json

            size = len(json.dumps(state, ensure_ascii=False, default=str))
        except (TypeError, ValueError) as exc:
            raise DecisionRequestError(f"state must be text or JSON: {exc}") from exc
    if size > max_state_chars:
        raise DecisionRequestError(f"state is too large: {size} characters, limit {max_state_chars}.")
    if not isinstance(questions, dict) or not questions:
        raise DecisionRequestError("questions must be a non-empty object of named questions.")
    if len(questions) > MAX_QUESTIONS:
        raise DecisionRequestError(f"too many questions: {len(questions)}, limit {MAX_QUESTIONS}.")
    for name, question in questions.items():
        _validate_question(name, question)


def _validate_question(name: Any, question: Any) -> None:
    if not isinstance(name, str) or not _NAME.match(name):
        raise DecisionRequestError(f"invalid question name {name!r}: use 1-64 letters, digits, '_', '.' or '-'.")
    if not isinstance(question, dict):
        raise DecisionRequestError(f"question '{name}' must be an object.")
    kind = question.get("type")
    if kind not in QUESTION_TYPES:
        raise DecisionRequestError(f"question '{name}': type must be one of {', '.join(QUESTION_TYPES)}.")
    instructions = question.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip() or len(instructions) > 2000:
        raise DecisionRequestError(f"question '{name}': instructions must be 1-2000 characters of text.")
    criteria = question.get("criteria")
    if kind == "choice":
        if not isinstance(criteria, dict) or not 2 <= len(criteria) <= MAX_CHOICE_OPTIONS:
            raise DecisionRequestError(
                f"question '{name}': a choice needs criteria with 2-{MAX_CHOICE_OPTIONS} options (name: description)."
            )
        for option, description in criteria.items():
            if not isinstance(option, str) or not option.strip() or len(option) > 64:
                raise DecisionRequestError(f"question '{name}': option names must be 1-64 characters of text.")
            if not isinstance(description, str) or len(description) > 500:
                raise DecisionRequestError(f"question '{name}': option '{option}' needs a description of up to 500 characters.")
    elif kind == "score":
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= MAX_SCORE_LEVELS:
            raise DecisionRequestError(
                f"question '{name}': a score needs criteria listing 2-{MAX_SCORE_LEVELS} levels, lowest first."
            )
        if not all(isinstance(level, str) and len(level) <= 300 for level in criteria):
            raise DecisionRequestError(f"question '{name}': each level must be text of up to 300 characters.")
    elif criteria is not None:  # noul
        if not isinstance(criteria, dict) or not set(criteria) <= {"true", "false"} or not all(
            isinstance(value, str) for value in criteria.values()
        ):
            raise DecisionRequestError(f"question '{name}': criteria for a yes/no question may only define 'true' and 'false'.")


# --- adapters ---------------------------------------------------------------


class _Adapter:
    """One loaded decision model. ``decide`` returns ``{"answers": ..., "usage": ...}``."""

    def __init__(self, profile_name: str, profile: Dict[str, Any]) -> None:
        self.profile_name = profile_name
        self.profile = profile
        self.lock = threading.Lock()
        self.loaded = False

    def load(self) -> None:
        raise NotImplementedError

    def decide(self, state: Any, questions: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        return {"profile": self.profile_name, "loaded": self.loaded}


def _require(module: str) -> Any:
    try:
        return __import__(module)
    except ModuleNotFoundError as exc:
        raise DecisionUnavailable(f"{INSTALL_HINT} (missing: {exc.name})") from exc


def _pick_device(torch: Any, wanted: str) -> str:
    wanted = (wanted or "auto").lower()
    if wanted == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if wanted == "cuda" and not torch.cuda.is_available():
        raise DecisionUnavailable("device is 'cuda' but no CUDA device is available. Use device 'cpu' or 'auto'.")
    if wanted not in {"cpu", "cuda"}:
        raise DecisionUnavailable("device must be 'auto', 'cpu' or 'cuda'.")
    return wanted


class D1Adapter(_Adapter):
    """LiquidAI d1: ``AutoModel`` with remote code exposing ``system_one``."""

    def load(self) -> None:
        profile = self.profile
        if profile.get("trust_remote_code") is not True:
            raise DecisionUnavailable(
                f"Profile '{self.profile_name}' runs code shipped with the model. Set \"trust_remote_code\": true "
                "in the profile to allow it, and keep \"revision\" pinned to a commit you have reviewed."
            )
        if not profile.get("revision"):
            raise DecisionUnavailable(
                f"Profile '{self.profile_name}' needs a pinned \"revision\" (a Hugging Face commit hash) because it "
                "runs remote code."
            )
        torch = _require("torch")
        transformers = _require("transformers")
        device = _pick_device(torch, str(profile.get("device") or "auto"))
        dtype_name = str(profile.get("dtype") or ("float32" if device == "cpu" else "float16"))
        dtype = getattr(torch, dtype_name, None)
        if not isinstance(dtype, torch.dtype):
            raise DecisionUnavailable(f"dtype '{dtype_name}' is not a torch dtype.")
        threads = profile.get("threads")
        if threads:
            torch.set_num_threads(int(threads))
        try:
            model = transformers.AutoModel.from_pretrained(
                str(profile["model_id"]),
                revision=str(profile["revision"]),
                trust_remote_code=True,
                dtype=dtype,
                local_files_only=offline_mode(),
            )
        except OSError as exc:
            raise DecisionUnavailable(_files_hint(self.profile_name, exc)) from exc
        except ImportError as exc:  # the model's own code imports a package that is not installed
            raise DecisionUnavailable(f"{INSTALL_HINT} ({exc})") from exc
        self.torch = torch
        self.device = device
        self.model = model.to(device).eval()
        self.one_question_at_a_time = False
        self.loaded = True

    def decide(self, state: Any, questions: Dict[str, Any]) -> Dict[str, Any]:
        with self.torch.inference_mode():
            if self.one_question_at_a_time and len(questions) > 1:
                return self._one_at_a_time(state, questions)
            try:
                return self.model.system_one(state, questions)
            except RuntimeError as exc:
                # d1 answers several questions over one shared pass with a flash-attention kernel
                # that some PyTorch builds (notably the Windows CUDA wheels) are compiled without.
                # A single question never needs it, so ask them one by one: same answers, no
                # shared-prefix speedup.
                if len(questions) > 1 and "flash_attention" in str(exc).lower():
                    self.one_question_at_a_time = True
                    return self._one_at_a_time(state, questions)
                raise

    def _one_at_a_time(self, state: Any, questions: Dict[str, Any]) -> Dict[str, Any]:
        answers: Dict[str, Any] = {}
        usage = {"input_tokens": 0, "output_tokens": 0}
        for name, question in questions.items():
            part = self.model.system_one(state, {name: question})
            answers.update(part["answers"])
            usage["input_tokens"] += int((part.get("usage") or {}).get("input_tokens", 0))
        return {"answers": answers, "usage": usage}

    def describe(self) -> Dict[str, Any]:
        info = super().describe()
        if self.loaded:
            info["device"] = self.device
            if self.one_question_at_a_time:
                info["one_question_at_a_time"] = True
        return info


class LayaAdapter(_Adapter):
    """Laya: ``laya.Router`` picks and lazily loads a checkpoint."""

    def load(self) -> None:
        if offline_mode():
            os.environ["HF_HUB_OFFLINE"] = "1"  # laya fetches through huggingface_hub and has no switch of its own
        _require("torch")
        laya = _require("laya")
        profile = self.profile
        device = str(profile.get("device") or "auto").lower()
        kwargs: Dict[str, Any] = {"device": None if device == "auto" else device}
        if profile.get("revision"):
            kwargs["revision"] = str(profile["revision"])
        try:
            self.router = laya.Router(**kwargs)
            self.checkpoint = str(profile.get("checkpoint") or "english")
            self.router.load(self.checkpoint)
        except (OSError, ValueError) as exc:
            raise DecisionUnavailable(_files_hint(self.profile_name, exc)) from exc
        self.loaded = True

    def decide(self, state: Any, questions: Dict[str, Any]) -> Dict[str, Any]:
        result = self.router.predict(state, questions, model=self.checkpoint)
        usage = result.get("usage") or {}
        return {"answers": result["answers"], "usage": usage}

    def describe(self) -> Dict[str, Any]:
        info = super().describe()
        if self.loaded:
            info["checkpoint"] = self.checkpoint
        return info


class HttpAdapter(_Adapter):
    """Forward to another ``/v1/systemone`` server on this machine."""

    def load(self) -> None:
        base_url = strip_trailing_slash(str(self.profile.get("base_url") or ""))
        if not base_url or not is_loopback_url(base_url):
            raise DecisionUnavailable(
                f"Profile '{self.profile_name}' needs a loopback base_url (http://127.0.0.1:PORT) for the systemone-http adapter."
            )
        self.url = f"{base_url}/v1/systemone"
        self.loaded = True

    def decide(self, state: Any, questions: Dict[str, Any]) -> Dict[str, Any]:
        headers = {}
        token = str(self.profile.get("api_key") or "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        body: Dict[str, Any] = {"state": state, "questions": questions}
        if self.profile.get("model_id"):
            body["model"] = self.profile["model_id"]
        timeout = float(self.profile.get("timeout_seconds") or 30)
        try:
            response = requests.post(self.url, json=body, headers=headers, timeout=timeout)
        except requests.RequestException as exc:
            raise DecisionError(f"decision server at {self.url} is not reachable: {exc}") from exc
        if response.status_code != 200:
            raise DecisionError(f"decision server at {self.url} answered HTTP {response.status_code}: {response.text[:300]}")
        try:
            data = response.json()
        except ValueError as exc:
            raise DecisionError(f"decision server at {self.url} did not return JSON.") from exc
        if not isinstance(data, dict) or "answers" not in data:
            raise DecisionError(f"decision server at {self.url} returned no 'answers'.")
        return {"answers": data["answers"], "usage": data.get("usage") or {}}


_ADAPTER_CLASSES = {"d1": D1Adapter, "laya": LayaAdapter, "systemone-http": HttpAdapter}


def _files_hint(profile_name: str, exc: Exception) -> str:
    if offline_mode():
        return (
            f"Profile '{profile_name}' could not find its model files and OFFLINE=true forbids downloading them. "
            f"Run once with OFFLINE=false to fetch them. ({exc})"
        )
    return f"Profile '{profile_name}' could not load its model files: {exc}"


# --- runtime ----------------------------------------------------------------

_registry_lock = threading.Lock()
_adapters: Dict[Tuple[str, str], _Adapter] = {}


def _fingerprint(profile: Dict[str, Any]) -> str:
    keys = ("adapter", "model_id", "revision", "device", "dtype", "checkpoint", "base_url", "threads")
    return "|".join(str(profile.get(key)) for key in keys)


def check_profile(profile_name: str, profile: Dict[str, Any]) -> None:
    """Fail early, and clearly, for anything knowable without loading the model."""
    adapter = str(profile.get("adapter") or "").lower()
    if adapter not in _ADAPTER_CLASSES:
        raise DecisionUnavailable(
            f"Profile '{profile_name}': adapter must be one of {', '.join(ADAPTER_NAMES)} (got {adapter or 'nothing'})."
        )
    needed = profile.get("license_acceptance_required")
    if needed and profile.get("license_accepted") is not True:
        raise DecisionUnavailable(
            f"Profile '{profile_name}' uses a model under the {needed}. Read it ({profile.get('license_url') or 'see the model page'}), "
            "then set \"license_accepted\": true in this profile to use it."
        )


def get_adapter(profile_name: str, profile: Dict[str, Any]) -> _Adapter:
    check_profile(profile_name, profile)
    key = (profile_name, _fingerprint(profile))
    with _registry_lock:
        adapter = _adapters.get(key)
        if adapter is None:
            adapter = _ADAPTER_CLASSES[str(profile["adapter"]).lower()](profile_name, profile)
            _adapters[key] = adapter
    if not adapter.loaded:
        with adapter.lock:
            if not adapter.loaded:
                adapter.load()
    return adapter


def decide(profile_name: str, profile: Dict[str, Any], state: Any, questions: Dict[str, Any]) -> Dict[str, Any]:
    """Answer ``questions`` about ``state`` with the profile's model."""
    validate_request(state, questions, int(profile.get("max_prompt_chars") or MAX_STATE_CHARS))
    started = time.perf_counter()
    was_loaded = _is_loaded(profile_name, profile)
    try:
        adapter = get_adapter(profile_name, profile)
    except (DecisionError, DecisionUnavailable, DecisionRequestError):
        raise
    except Exception as exc:  # loading a model can fail in ways nobody listed; never a raw 500
        raise DecisionError(f"could not load '{profile_name}': {type(exc).__name__}: {exc}") from exc
    loaded_at = time.perf_counter()
    with adapter.lock:
        try:
            result = adapter.decide(state, questions)
        except (DecisionError, DecisionUnavailable, DecisionRequestError):
            raise
        except Exception as exc:  # a model failure must not become a raw 500
            raise DecisionError(f"{type(exc).__name__}: {exc}") from exc
    done = time.perf_counter()
    result = dict(result)
    result["latency_ms"] = round((done - loaded_at) * 1000, 2)
    if not was_loaded:
        result["load_ms"] = round((loaded_at - started) * 1000, 2)
    return result


def _is_loaded(profile_name: str, profile: Dict[str, Any]) -> bool:
    adapter = _adapters.get((profile_name, _fingerprint(profile)))
    return bool(adapter and adapter.loaded)


def status() -> List[Dict[str, Any]]:
    """What is currently loaded, for /health. Never loads anything."""
    with _registry_lock:
        return [adapter.describe() for adapter in _adapters.values() if adapter.loaded]


def reset() -> None:
    """Forget every loaded model (tests, and config changes)."""
    with _registry_lock:
        _adapters.clear()


def warm(profile_name: str, profile: Dict[str, Any]) -> None:
    """Load a profile's model now so the first request does not pay for it."""
    get_adapter(profile_name, profile)
