"""Console entry point: ``localdeploy`` serves the API + web UI.

For pip/pipx installs there is no repo checkout to anchor to, so runtime state
(.env, config.json, logs/, reports/) lives in the app home - ~/.localdeploy by
default, or wherever LOCALDEPLOY_HOME points (see ``localdeploy.utils.app_home``).
Running from a source checkout keeps using the repo root, same as the start
scripts.
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import webbrowser

from . import __version__
from .utils import api_client_base_url, app_home


def port_in_use(host: str, port: int) -> bool:
    """True when something already accepts connections on host:port."""
    probe_host = "127.0.0.1" if host in {"", "0.0.0.0"} else "::1" if host == "::" else host
    try:
        with socket.create_connection((probe_host, port), timeout=0.3):
            return True
    except OSError:
        return False


def choose_port(host: str, port: int, *, explicit: bool, attempts: int = 10) -> int | None:
    """Return the port to serve on, or None (after saying why) when none can be used.

    Port 8000 is a popular default, so another local tool often holds it. Without this check
    LocalDeploy printed that port's URL, opened the browser on the other program, and ended with
    a single bind error and exit code 0. An explicitly requested port is never changed silently.
    """
    if not port_in_use(host, port):
        return port
    if explicit:
        print(
            f"Port {port} is already in use by another program. Pick a different one, "
            f"for example: localdeploy --port {port + 1}",
            file=sys.stderr,
        )
        return None
    for candidate in range(port + 1, port + attempts + 1):
        if not port_in_use(host, candidate):
            print(f"Port {port} is already in use; using {candidate} instead.")
            return candidate
    print(f"Ports {port}-{port + attempts} are all in use. Pick one with --port.", file=sys.stderr)
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="localdeploy",
        description="Pick, deploy & benchmark the best local AI model for your machine.",
    )
    parser.add_argument("--host", help="bind address (default: API_HOST or 127.0.0.1)")
    parser.add_argument("--port", type=int, help="port (default: API_PORT or 8000)")
    parser.add_argument("--no-browser", action="store_true", help="don't open the UI in a browser")
    parser.add_argument("--version", action="version", version=f"localdeploy {__version__}")
    args = parser.parse_args(argv)

    home = app_home()
    home.mkdir(parents=True, exist_ok=True)
    # CLI flags beat .env: they land in the environment before api_server's
    # load_dotenv runs, and load_dotenv never overrides existing variables.
    if args.host:
        os.environ["API_HOST"] = args.host
    if args.port:
        os.environ["API_PORT"] = str(args.port)

    import uvicorn

    from localdeploy.server import app  # noqa: PLC0415 - import after env is settled

    host = os.getenv("API_HOST", "127.0.0.1")
    chosen = choose_port(host, int(os.getenv("API_PORT", "8000")), explicit=bool(args.port) or "API_PORT" in os.environ)
    if chosen is None:
        return 2
    port = chosen
    os.environ["API_PORT"] = str(port)  # client scripts started from here read the port from the environment
    client_base = api_client_base_url(host, port)
    url = f"{client_base}/ui"
    print(f"LocalDeploy {__version__}")
    print(f"  home:  {home}")
    print(f"  UI:    {url}")
    print(f"  docs:  {client_base}/docs")
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
