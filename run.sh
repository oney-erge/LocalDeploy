#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./scripts/install-utils.sh
install_init "$PWD" "LocalDeploy"
install_enable_traps
action=run
case "${1:-}" in run|doctor|repair|docker|stop|logs) action=$1; shift ;; esac
no_browser=0
for arg in "$@"; do [ "$arg" = --no-browser ] && no_browser=1 || { echo "unknown option: $arg" >&2; exit 2; }; done
port=${API_PORT:-8000}
url="http://127.0.0.1:$port"
health_check() { if command -v curl >/dev/null 2>&1; then curl -fsS --max-time 2 "$url/health" >/dev/null; else wget -qO- --timeout=2 "$url/health" >/dev/null; fi; }
wait_ready() { for _ in $(seq 1 120); do health_check 2>/dev/null && return; sleep 0.5; done; return 1; }
open_url() { [ "$no_browser" -eq 1 ] && return; command -v open >/dev/null 2>&1 && open "$url/ui" || command -v xdg-open >/dev/null 2>&1 && xdg-open "$url/ui" || true; }
docker_running() { command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1 && [ -n "$(docker compose ps --quiet 2>/dev/null)" ]; }

case "$action" in
  docker)
    command -v docker >/dev/null 2>&1 || { echo "Docker is not installed." >&2; exit 1; }
    docker info >/dev/null 2>&1 || { echo "Docker is installed but its engine is not running." >&2; exit 1; }
    install_lock
    install_require_space "$PWD" 3
    docker compose up --detach --build
    wait_ready || { docker compose logs; echo "LocalDeploy did not become ready at $url." >&2; exit 1; }
    install_complete
    echo "LocalDeploy is ready at $url/ui"; open_url; exit 0 ;;
  logs)
    docker_running && exec docker compose logs --follow
    if compgen -G 'logs/*.log' >/dev/null; then exec tail -n 100 -F logs/*.log; fi
    echo "No LocalDeploy logs exist yet. Start it with ./run.sh."; exit 0 ;;
  stop)
    if [ -f logs/api_server.pid ]; then exec ./scripts/stop.sh; fi
    if docker_running; then exec docker compose down; fi
    exec ./scripts/stop.sh ;;
  doctor)
    [ -x .venv/bin/python ] || { echo "Environment: missing, run ./run.sh once" >&2; exit 1; }
    .venv/bin/python -c "import api_server, localdeploy; print('Imports: ready')"
    health_check 2>/dev/null && echo "API: $url" || echo "API: not running"
    exit 0 ;;
  repair) rm -f .venv/requirements.sha256 ;;
esac
if [ "$no_browser" -eq 1 ]; then
  export NO_BROWSER=1
fi
exec ./scripts/start.sh
