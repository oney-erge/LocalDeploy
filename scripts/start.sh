#!/usr/bin/env bash
# Local (non-Docker) launcher for macOS/Linux: sets up a venv, starts Ollama
# when it is installed but not running, then starts the API and UI.
set -euo pipefail

cd "$(dirname "$0")/.."
source ./scripts/install-utils.sh
install_init "$PWD" "LocalDeploy"
install_enable_traps
install_lock
install_require_space "$PWD" 3

# --- guided install helpers ---------------------------------------------------
# Mirrors what start.ps1 offers via winget on Windows: ask once, use whatever
# package manager is actually present, and never block a non-interactive run
# (piped input, CI, no attached terminal) - those always fall through to
# printing the manual instructions instead of prompting or acting.

confirm() {
  if [ ! -t 0 ]; then
    return 1
  fi
  local reply
  read -r -p "$1 [y/N] " reply || return 1
  case "$reply" in
    [Yy] | [Yy][Ee][Ss]) return 0 ;;
    *) return 1 ;;
  esac
}

install_python_guided() {
  case "$(uname -s)" in
    Darwin)
      if command -v brew >/dev/null 2>&1; then
        if confirm "Python 3 was not found. Install it now with 'brew install python'?"; then
          if install_retry "Python Homebrew install" brew install python; then return 0; fi
        fi
      fi
      ;;
    Linux)
      if command -v apt-get >/dev/null 2>&1; then
        if confirm "Python 3 was not found. Install it now with 'sudo apt install python3 python3-venv python3-pip'?"; then
          if install_retry "APT metadata refresh" sudo apt-get update -y &&
             install_retry "Python APT install" sudo apt-get install -y python3 python3-venv python3-pip; then return 0; fi
        fi
      elif command -v dnf >/dev/null 2>&1; then
        if confirm "Python 3 was not found. Install it now with 'sudo dnf install python3 python3-pip'?"; then
          if install_retry "Python DNF install" sudo dnf install -y python3 python3-pip; then return 0; fi
        fi
      elif command -v pacman >/dev/null 2>&1; then
        if confirm "Python 3 was not found. Install it now with 'sudo pacman -S python'?"; then
          if install_retry "Python pacman install" sudo pacman -Sy --noconfirm python; then return 0; fi
        fi
      elif command -v zypper >/dev/null 2>&1; then
        if confirm "Python 3 was not found. Install it now with 'sudo zypper install python3 python3-pip'?"; then
          if install_retry "Python zypper install" sudo zypper install -y python3 python3-pip; then return 0; fi
        fi
      fi
      ;;
  esac
  return 1
}

install_ollama_guided() {
  case "$(uname -s)" in
    Darwin)
      if command -v brew >/dev/null 2>&1; then
        if confirm "Ollama was not found. Install it now with 'brew install ollama'?"; then
          if install_retry "Ollama Homebrew install" brew install ollama; then return 0; fi
        fi
      fi
      ;;
    Linux)
      if confirm "Ollama was not found. Install it now using the official installer (curl -fsSL https://ollama.com/install.sh | sh)?"; then
        local installer
        installer="$(mktemp)"
        if install_download "https://ollama.com/install.sh" "$installer" "Ollama installer" &&
           sh "$installer"; then
          rm -f -- "$installer"
          return 0
        fi
        rm -f -- "$installer"
      fi
      ;;
  esac
  return 1
}

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  if install_python_guided && command -v "$PYTHON" >/dev/null 2>&1; then
    echo "[start] Python installed."
  else
    echo "ERROR: Python 3 was not found." >&2
    echo "  macOS:          brew install python  (or https://www.python.org/downloads/)" >&2
    echo "  Debian/Ubuntu:  sudo apt install python3 python3-venv" >&2
    echo "  Fedora:         sudo dnf install python3" >&2
    echo "  Arch:           sudo pacman -S python" >&2
    echo "Then re-run this script." >&2
    exit 1
  fi
fi
if ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "ERROR: Python 3.10+ is required; found $("$PYTHON" --version 2>&1)." >&2
  exit 1
fi

if [ ! -d .venv ]; then
  echo "[start] creating virtualenv (.venv) ..."
  "$PYTHON" -m venv .venv || {
    echo "ERROR: could not create a virtualenv. On Debian/Ubuntu: sudo apt install python3-venv" >&2
    exit 1
  }
fi
# shellcheck disable=SC1091
. .venv/bin/activate

# Reinstall only when requirements.txt changed since the last install, so a
# git pull that adds a dependency can't leave the venv broken.
REQ_HASH="$( (sha256sum requirements.txt 2>/dev/null || shasum -a 256 requirements.txt) | cut -d' ' -f1 )"
if [ "$(cat .venv/requirements.sha256 2>/dev/null)" != "$REQ_HASH" ]; then
  echo "[start] installing dependencies (first run can take a minute) ..."
  install_retry "pip upgrade" pip install --quiet --upgrade pip
  install_retry "dependency installation" pip install --quiet -r requirements.txt || {
    echo "ERROR: dependency install failed. Check your internet connection and re-run." >&2
    exit 1
  }
  echo "$REQ_HASH" > .venv/requirements.sha256
fi

# Seed local environment defaults if absent. The live config starts empty and
# is created when the first model profile is saved.
[ -f .env ] || { [ -f .env.example ] && cp .env.example .env; } || true

# Match the Python server's dotenv behavior without sourcing .env as shell code.
# Explicit shell values still win over the file, just as they do in start.ps1.
IFS=$'\t' read -r HOST PORT OLLAMA_URL START_OLLAMA_VALUE OLLAMA_NO_CLOUD_VALUE OLLAMA_HOST_VALUE < <(
  "$PYTHON" - "${API_HOST:-}" "${API_PORT:-}" "${OLLAMA_BASE_URL:-}" \
    "${START_OLLAMA:-}" "${OLLAMA_NO_CLOUD:-}" <<'PY'
import sys
from ipaddress import ip_address
from urllib.parse import urlsplit

from dotenv import dotenv_values

values = dotenv_values(".env")
requested = (
    (sys.argv[1], "API_HOST", "127.0.0.1"),
    (sys.argv[2], "API_PORT", "8000"),
    (sys.argv[3], "OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
    (sys.argv[4], "START_OLLAMA", "true"),
    (sys.argv[5], "OLLAMA_NO_CLOUD", "true"),
)
resolved = []
for explicit, name, default in requested:
    value = explicit or values.get(name) or default
    if any(character in value for character in "\t\r\n"):
        raise SystemExit(f"Invalid control character in {name}")
    resolved.append(value)
resolved[2] = resolved[2].rstrip("/")
for index, name in ((3, "START_OLLAMA"), (4, "OLLAMA_NO_CLOUD")):
    normalized = resolved[index].lower()
    if normalized in {"1", "true", "yes", "on"}:
        resolved[index] = "true"
    elif normalized in {"0", "false", "no", "off"}:
        resolved[index] = "false"
    else:
        raise SystemExit(f"{name} must be true or false")
parsed = urlsplit(resolved[2])
if parsed.scheme not in {"http", "https"} or not parsed.netloc:
    raise SystemExit("OLLAMA_BASE_URL must be an HTTP URL")
if parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
    raise SystemExit("OLLAMA_BASE_URL must contain only a local host and optional port")
hostname = (parsed.hostname or "").lower()
try:
    is_loopback = hostname == "localhost" or ip_address(hostname).is_loopback
    parsed.port  # validates the optional port
except ValueError:
    is_loopback = False
if not is_loopback:
    raise SystemExit("OLLAMA_BASE_URL must use localhost or a loopback IP address")
print("\t".join([*resolved, parsed.netloc]))
PY
)

ollama_ready() {
  curl -fsS --max-time 2 "${OLLAMA_URL}/api/tags" >/dev/null 2>&1
}

if ! ollama_ready; then
  if ! command -v ollama >/dev/null 2>&1; then
    install_ollama_guided || true
  fi
  if ollama_ready; then
    # The guided installer above can already leave Ollama running (e.g. the
    # official Linux script starts a systemd service) - nothing left to do,
    # and starting a second `ollama serve` here would just fail to bind the
    # port that one is already using.
    echo "[start] Ollama is ready at $OLLAMA_URL"
  elif ! command -v ollama >/dev/null 2>&1; then
    echo "[start] NOTE: Ollama was not found. Install it from https://ollama.com/download."
    echo "        The UI will start anyway and shows Ollama's status on the Setup tab."
  elif [ "$START_OLLAMA_VALUE" = "false" ]; then
    echo "[start] Ollama is not reachable and automatic startup is disabled (START_OLLAMA=false)."
  else
    mkdir -p logs
    managed_pid=""
    if [ -f logs/ollama.pid ]; then
      managed_pid="$(cat logs/ollama.pid 2>/dev/null || true)"
      case "$managed_pid" in
        ''|*[!0-9]*) managed_pid="" ;;
        *) kill -0 "$managed_pid" 2>/dev/null || managed_pid="" ;;
      esac
    fi
    if [ -z "$managed_pid" ]; then
      rm -f logs/ollama.pid
      echo "[start] starting Ollama ..."
      OLLAMA_NO_CLOUD="$OLLAMA_NO_CLOUD_VALUE" OLLAMA_HOST="$OLLAMA_HOST_VALUE" \
        nohup ollama serve >logs/ollama.out.log 2>logs/ollama.err.log &
      managed_pid=$!
      printf '%s\n' "$managed_pid" > logs/ollama.pid
    else
      echo "[start] waiting for the Ollama process started earlier (PID $managed_pid) ..."
    fi
    for ((attempt = 0; attempt < 20; attempt++)); do
      ollama_ready && break
      kill -0 "$managed_pid" 2>/dev/null || break
      sleep 0.75
    done
    if ollama_ready; then
      echo "[start] Ollama is ready at $OLLAMA_URL"
    elif ! kill -0 "$managed_pid" 2>/dev/null; then
      rm -f logs/ollama.pid
      echo "[start] WARNING: Ollama exited during startup. Check logs/ollama.err.log." >&2
    else
      echo "[start] WARNING: Ollama has not answered yet. Check logs/ollama.err.log." >&2
    fi
  fi
fi
install_complete

BROWSE_HOST="$HOST"
case "$BROWSE_HOST" in
  0.0.0.0|::) BROWSE_HOST="127.0.0.1" ;;
  ::1) BROWSE_HOST="[::1]" ;;
esac
UI_URL="http://${BROWSE_HOST}:${PORT}/ui"
echo "[start] LocalDeploy UI:  $UI_URL"
echo "[start] Stop: ./scripts/stop.sh  (add --ollama to stop a managed Ollama process)"

# Open the UI once the server answers (set NO_BROWSER=1 to skip).
if [ -z "${NO_BROWSER:-}" ]; then
  opener=""
  command -v xdg-open >/dev/null 2>&1 && opener="xdg-open"
  command -v open >/dev/null 2>&1 && opener="open"
  if [ -n "$opener" ]; then
    (
      for ((attempt = 0; attempt < 60; attempt++)); do
        if curl -fsS "http://${BROWSE_HOST}:${PORT}/health" >/dev/null 2>&1; then
          "$opener" "$UI_URL" >/dev/null 2>&1 || true
          break
        fi
        sleep 1
      done
    ) &
  fi
fi

mkdir -p logs
printf '%s\n' "$$" > logs/api_server.pid
exec uvicorn api_server:app --host "$HOST" --port "$PORT"
