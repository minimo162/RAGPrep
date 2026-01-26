from __future__ import annotations

import argparse
import sys
import threading
import time
from typing import Final
from urllib.error import URLError
from urllib.request import Request, urlopen

import uvicorn

from ragprep.web.app import app

DEFAULT_HOST: Final[str] = "127.0.0.1"
DEFAULT_PORT: Final[int] = 8000
DEFAULT_READY_TIMEOUT_S: Final[float] = 20.0


class _ThreadedUvicornServer(uvicorn.Server):
    def install_signal_handlers(self) -> None:  # pragma: no cover
        return


def _client_host_for(bind_host: str) -> str:
    if bind_host in {"0.0.0.0", "::"}:
        return "127.0.0.1"
    return bind_host


def _wait_for_health(*, health_url: str, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    request = Request(health_url, headers={"User-Agent": "ragprep-desktop"})

    while time.monotonic() < deadline:
        try:
            with urlopen(request, timeout=1.0) as response:
                body = response.read(32).decode("utf-8", errors="ignore").strip()
                if response.status == 200 and body == "ok":
                    return True
        except (OSError, URLError):
            pass
        time.sleep(0.2)

    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAGPrep desktop launcher (Uvicorn + pywebview)")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Bind port (default: 8000)")
    parser.add_argument(
        "--ready-timeout",
        type=float,
        default=DEFAULT_READY_TIMEOUT_S,
        help="Seconds to wait for server readiness (default: 20)",
    )
    args = parser.parse_args(argv)

    config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )
    server = _ThreadedUvicornServer(config=config)
    server_thread = threading.Thread(target=server.run, name="ragprep-uvicorn", daemon=True)
    server_thread.start()

    try:
        client_host = _client_host_for(args.host)
        health_url = f"http://{client_host}:{args.port}/health"
        if not _wait_for_health(health_url=health_url, timeout_s=args.ready_timeout):
            print(
                f"[ERROR] Server did not become ready: {health_url}",
                file=sys.stderr,
            )
            return 1

        try:
            import webview
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] Failed to import pywebview: {exc}", file=sys.stderr)
            return 1

        url = f"http://{client_host}:{args.port}/"
        webview.create_window("RAGPrep", url)
        webview.start()
        return 0
    finally:
        server.should_exit = True
        server_thread.join(timeout=5.0)


if __name__ == "__main__":
    raise SystemExit(main())
