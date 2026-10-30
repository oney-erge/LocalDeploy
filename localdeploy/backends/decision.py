"""Decision models ("System One"): answer typed questions about a piece of text.

A decision model does not write text. It reads one *state* and a set of named
*questions* and returns, for each question, a typed answer with a probability per
option, in a single forward pass. LocalDeploy serves them at ``POST /v1/systemone``
so an application can ask "is this a task or chat, and which route?" in well under a
second on a CPU and fall back to a language model only when the answer
is not confident.

Four adapters share one request and response shape:

* ``llamacpp``       a GGUF file in a ``llama-server`` that LocalDeploy starts and manages (no Python
                     packages, no model code; the fastest option on a CPU)
* ``d1``             LiquidAI d1 models through PyTorch (``AutoModel`` with ``system_one``)
* ``laya``           Laya checkpoints through PyTorch (``laya.Router``)
* ``systemone-http`` any other server that already speaks ``/v1/systemone``
                     (``laya-serve``, ollaya, ...), loopback only like every backend

For the PyTorch adapters the heavy packages (torch, transformers, laya) are optional and imported only
when a decision profile is first used: ``pip install -r requirements-decision.txt``.
"""

from __future__ import annotations

import atexit
import os
import re
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from ..utils import BackendCallError, app_home, is_loopback_url, offline_mode, strip_trailing_slash

DECISION_BACKEND = "decision"
ADAPTER_NAMES = ("d1", "laya", "systemone-http", "llamacpp")
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

    def stop(self) -> None:
        """Release anything the adapter started (a server process). Most adapters hold nothing."""

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

    send_model = True  # a server hosting several models picks one by name; llama-server hosts exactly one

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
        if self.send_model and self.profile.get("model_id"):
            body["model"] = self.profile["model_id"]
        timeout = float(self.profile.get("timeout_seconds") or 30)
        try:
            response = requests.post(self.url, json=body, headers=headers, timeout=timeout)
        except requests.RequestException as exc:
            raise DecisionError(f"decision server at {self.url} is not reachable: {exc}") from exc
        if response.status_code != 200:
            raise self._http_error(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise DecisionError(f"decision server at {self.url} did not return JSON.") from exc
        if not isinstance(data, dict) or "answers" not in data:
            raise DecisionError(f"decision server at {self.url} returned no 'answers'.")
        return {"answers": data["answers"], "usage": data.get("usage") or {}}

    def _http_error(self, response: requests.Response) -> BackendCallError:
        return DecisionError(f"decision server at {self.url} answered HTTP {response.status_code}: {response.text[:300]}")


LLAMACPP_TESTED_BUILD = 11976  # the llama.cpp release the comparison in docs/MODELS.md used
LLAMACPP_GET = (
    "On Windows: winget install ggml.llamacpp (or winget upgrade ggml.llamacpp). On macOS: brew install llama.cpp. "
    "Or download a release from https://github.com/ggml-org/llama.cpp/releases and set LLAMACPP_SERVER_PATH "
    "(or the profile's \"server_path\") to its llama-server."
)


def _winget_llamacpp(name: str) -> Optional[str]:
    """The copy ``winget install ggml.llamacpp`` keeps in the WinGet package folder. WinGet also puts it on PATH,
    but only for programs started after the install, so a LocalDeploy that was already running would miss it."""
    local = os.getenv("LOCALAPPDATA")
    if not local:
        return None
    packages = Path(local) / "Microsoft" / "WinGet" / "Packages"
    for candidate in sorted(packages.glob(f"ggml.llamacpp*/**/{name}.exe")):
        return str(candidate)
    return None


def find_llama_server(configured: str = "") -> Optional[str]:
    """llama-server, looked for the way scripts/start_llamacpp.ps1 looks: the configured path or command, else
    PATH, else a WinGet install. A configured value that does not resolve is not replaced by another copy."""
    if configured:
        return configured if Path(configured).is_file() else shutil.which(configured)
    return shutil.which("llama-server") or _winget_llamacpp("llama-server")


def find_llama_quantize(server: Optional[str] = None) -> Optional[str]:
    """llama-quantize: next to llama-server (every release and the WinGet package ship both), else PATH or WinGet."""
    if server:
        sibling = Path(server).with_name("llama-quantize" + Path(server).suffix)
        if sibling.is_file():
            return str(sibling)
    return shutil.which("llama-quantize") or _winget_llamacpp("llama-quantize")


def llama_server_build(server: str) -> Optional[int]:
    """The build number ``llama-server --version`` reports, or None. Older builds print ``version: 9297 (b0df4c0cf)``,
    newer ones ``version: 0.8.0-dev (build 11976, commit 2e0d86beb)``."""
    try:
        result = subprocess.run(  # noqa: S603 - the path was found or configured above
            [server, "--version"], capture_output=True, text=True, timeout=20, check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = (result.stdout or "") + (result.stderr or "")
    match = re.search(r"\(build (\d+)", output) or re.search(r"version:\s*(\d+)\s*\(", output)
    return int(match.group(1)) if match else None


def _repo_stem(repo: str) -> str:
    name = repo.split(":", 1)[0].rstrip("/").split("/")[-1]
    return re.sub(r"[-_]gguf$", "", name, flags=re.IGNORECASE)


def quantized_gguf(
    repo: str, quant: str, out_dir: Path, *, quantizer: str, source_file: Optional[str] = None, threads: int = 0, log=None
) -> Path:
    """A ``quant`` GGUF of a Hugging Face GGUF repository, made once with llama-quantize from its BF16 file.

    The published d3 files are Q8_0 and BF16; on a CPU a Q4_K_M file answered about 1.8 times faster with the
    same routing accuracy in docs/MODELS.md. The result is kept in ``out_dir`` and reused, so only the first call
    downloads (the BF16 file of d3-lite is about 1.7 GB) and quantizes.
    """
    log = log or (lambda message: None)
    if not re.fullmatch(r"[A-Za-z0-9_]+", quant):
        raise DecisionUnavailable(f"quantize must be a llama.cpp type such as Q4_K_M, not {quant!r}.")
    repo = repo.split(":", 1)[0]
    target = Path(out_dir) / f"{_repo_stem(repo)}-{quant}.gguf"
    if target.is_file():
        return target
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise DecisionUnavailable("Making a quantized file needs huggingface_hub (it is in requirements.txt).") from exc
    if source_file is None:
        if offline_mode():
            raise DecisionUnavailable(
                f"{target.name} has not been made yet and OFFLINE=true forbids downloading {repo}. "
                "Run once with OFFLINE=false."
            )
        ggufs = [
            name for name in HfApi().list_repo_files(repo)
            if name.lower().endswith(".gguf") and not Path(name).name.lower().startswith("mmproj")
        ]
        source_file = next((name for needle in ("bf16", "f16", "f32") for name in ggufs if needle in name.lower()), None)
        if source_file is None:
            raise DecisionUnavailable(f"{repo} has no BF16/F16 GGUF file to quantize from. Found: {', '.join(ggufs) or 'none'}.")
    log(f"downloading {repo}/{source_file}")
    source = hf_hub_download(repo, source_file, local_files_only=offline_mode())
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".partial")
    command = [quantizer, source, str(partial), quant] + ([str(threads)] if threads else [])
    log("running " + " ".join(command))
    result = subprocess.run(  # noqa: S603 - the quantizer was found next to llama-server or on PATH
        command, capture_output=True, text=True, check=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode != 0 or not partial.is_file():
        partial.unlink(missing_ok=True)
        tail = " | ".join(((result.stderr or "") + (result.stdout or "")).strip().splitlines()[-4:])
        raise DecisionUnavailable(f"llama-quantize failed (exit {result.returncode}) making {target.name}: {tail}")
    partial.replace(target)
    return target


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class LlamaCppAdapter(HttpAdapter):
    """Run a GGUF decision model in a ``llama-server`` that LocalDeploy starts, and forward to it.

    llama.cpp serves ``/v1/systemone`` itself for GGUF decision models. Compared with loading the same model
    through PyTorch it needs no Python model code (so no ``trust_remote_code``), runs the quantized file, and was
    2.4 times faster on a CPU in LocalDeploy's own comparison (docs/MODELS.md). The server is started on first
    use on a free loopback port, restarted if it dies, and stopped when LocalDeploy exits.

    ``model_id`` is a path to a ``.gguf`` file, or ``repo:quant`` for llama.cpp to fetch from Hugging Face
    (for example ``ggml-org/vllm-d3-lite-GGUF:Q8_0``). With ``"quantize": "Q4_K_M"`` and a repository as
    ``model_id``, LocalDeploy makes that smaller file once from the repository's BF16 file and uses it.
    """

    send_model = False

    def __init__(self, profile_name: str, profile: Dict[str, Any]) -> None:
        super().__init__(profile_name, profile)
        self.process: Optional[subprocess.Popen] = None
        self.log_path: Optional[Path] = None
        self.server_path: Optional[str] = None
        atexit.register(self.stop)

    def _server(self) -> str:
        configured = str(self.profile.get("server_path") or os.getenv("LLAMACPP_SERVER_PATH") or "").strip()
        found = find_llama_server(configured)
        if not found:
            where = f"{configured} was not found. " if configured else ""
            raise DecisionUnavailable(
                f"Profile '{self.profile_name}' needs llama.cpp's llama-server (b{LLAMACPP_TESTED_BUILD} or newer: it "
                f"serves /v1/systemone). {where}{LLAMACPP_GET}"
            )
        self.server_path = found
        return found

    def _model(self, server: str) -> str:
        model = str(self.profile.get("model_id") or "").strip()
        quant = str(self.profile.get("quantize") or "").strip()
        if not model or not quant or Path(model).suffix.lower() == ".gguf":
            return model
        quantizer = find_llama_quantize(server)
        if not quantizer:
            raise DecisionUnavailable(f"Profile '{self.profile_name}' asks for {quant} but llama-quantize was not found. {LLAMACPP_GET}")
        return str(quantized_gguf(model, quant, app_home() / "models" / "decision", quantizer=quantizer))

    def _command(self, server: str, port: int) -> List[str]:
        profile = self.profile
        model = self._model(server)
        if not model:
            raise DecisionUnavailable(f"Profile '{self.profile_name}' needs a model_id (a .gguf path or repo:quant).")
        batch = str(int(profile.get("batch") or 4096))  # a question is read in one batch, so it must hold the whole prompt
        command = [
            server, "--host", "127.0.0.1", "--port", str(port),
            "-c", str(int(profile.get("context") or 4096)), "-b", batch, "-ub", batch, "-np", "1",
        ]
        if Path(model).suffix.lower() == ".gguf":
            if not Path(model).is_file():
                raise DecisionUnavailable(f"Profile '{self.profile_name}': the model file {model} does not exist.")
            command += ["-m", model]
        else:
            command += ["-hf", model]
            if offline_mode():
                command += ["--offline"]
        if profile.get("threads"):
            command += ["-t", str(int(profile["threads"]))]
        device = str(profile.get("device") or "auto").lower()
        if device == "cpu":
            command += ["-ngl", "0"]
        elif device == "cuda":
            command += ["-ngl", "99"]
        elif device != "auto":
            raise DecisionUnavailable("device must be 'auto', 'cpu' or 'cuda'.")
        return command

    def _pid_path(self) -> Path:
        return app_home() / "logs" / f"decision-{re.sub(r'[^A-Za-z0-9_.-]', '_', self.profile_name)}.pid"

    def _stop_left_over(self) -> None:
        """Stop the llama-server a LocalDeploy that was killed outright left running.

        A crash or a forced stop skips the exit handler, so its llama-server keeps the model in memory (about
        1 GB for d3-lite) until something ends it. The pid file names that exact process by id and start time, so
        no other llama-server (a chat model's, a user's own) is touched, even if the id has been reused.
        """
        path = self._pid_path()
        try:
            pid_text, created_text = path.read_text(encoding="utf-8").split()
            pid, created = int(pid_text), float(created_text)
        except (OSError, ValueError):
            return
        try:
            import psutil

            process = psutil.Process(pid)
            if abs(process.create_time() - created) < 1 and "llama-server" in process.name().lower():
                process.terminate()
                process.wait(timeout=10)
        except Exception:  # noqa: BLE001 - already gone, or not the process that was recorded
            pass
        path.unlink(missing_ok=True)

    def _record_pid(self) -> None:
        pid = getattr(self.process, "pid", None)
        if not isinstance(pid, int):
            return
        try:
            import psutil

            self._pid_path().write_text(f"{pid} {psutil.Process(pid).create_time()}", encoding="utf-8")
        except Exception:  # noqa: BLE001 - only a convenience for the next start
            pass

    def load(self) -> None:
        command = self._command(self._server(), int(self.profile.get("port") or _free_port()))
        port = int(command[command.index("--port") + 1])
        logs = app_home() / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        self._stop_left_over()
        self.log_path = logs / f"decision-{re.sub(r'[^A-Za-z0-9_.-]', '_', self.profile_name)}.log"
        with open(self.log_path, "ab") as log:
            self.process = subprocess.Popen(  # noqa: S603 - the command is built here from validated profile fields
                command, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        self._record_pid()
        base = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + float(self.profile.get("startup_timeout_seconds") or 180)
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise DecisionUnavailable(
                    f"llama-server for '{self.profile_name}' exited with code {self.process.returncode} while starting. "
                    f"Last log lines: {self._log_tail()}"
                )
            try:
                reply = requests.get(f"{base}/health", timeout=2)
                if reply.status_code == 200:
                    self.url = f"{base}/v1/systemone"
                    self.loaded = True
                    return
            except requests.RequestException:
                pass  # still starting
            time.sleep(0.5)
        self.stop()
        raise DecisionUnavailable(
            f"llama-server for '{self.profile_name}' did not become ready in time. Last log lines: {self._log_tail()}"
        )

    def decide(self, state: Any, questions: Dict[str, Any]) -> Dict[str, Any]:
        if self.process is None or self.process.poll() is not None:
            self.loaded = False
            self.load()  # it died since the last request (or was stopped): start it again
        return super().decide(state, questions)

    def _http_error(self, response: requests.Response) -> BackendCallError:
        if response.status_code == 404:
            # An older llama.cpp starts and loads the model fine but has no /v1/systemone.
            build = llama_server_build(self.server_path) if self.server_path else None
            return DecisionUnavailable(
                f"This llama-server{f' (build {build})' if build else ''} does not serve /v1/systemone. Decision models "
                f"need a newer llama.cpp (b{LLAMACPP_TESTED_BUILD} was tested). {LLAMACPP_GET}"
            )
        return super()._http_error(response)

    def _log_tail(self, lines: int = 6) -> str:
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace") if self.log_path else ""
        except OSError:
            return "(no log)"
        return " | ".join(text.strip().splitlines()[-lines:]) or "(empty log)"

    def stop(self) -> None:
        process, self.process, self.loaded = self.process, None, False
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        if process is not None:
            try:
                self._pid_path().unlink(missing_ok=True)
            except OSError:
                pass

    def describe(self) -> Dict[str, Any]:
        info = super().describe()
        if self.loaded:
            info["server"] = self.url
        return info


_ADAPTER_CLASSES = {"d1": D1Adapter, "laya": LayaAdapter, "systemone-http": HttpAdapter, "llamacpp": LlamaCppAdapter}


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
    keys = (
        "adapter", "model_id", "revision", "device", "dtype", "checkpoint", "base_url", "threads", "server_path", "context",
        "batch", "port", "quantize",
    )
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
    """Forget every loaded model and stop any server LocalDeploy started (tests, and config changes)."""
    with _registry_lock:
        for adapter in _adapters.values():
            adapter.stop()
        _adapters.clear()


def warm(profile_name: str, profile: Dict[str, Any]) -> None:
    """Load a profile's model now so the first request does not pay for it."""
    get_adapter(profile_name, profile)
