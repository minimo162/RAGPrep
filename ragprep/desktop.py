from __future__ import annotations

import argparse
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Final
from urllib.error import URLError
from urllib.parse import unquote
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


_FILENAME_STAR_RE: Final[re.Pattern[str]] = re.compile(
    r"filename\\*=UTF-8''(?P<name>[^;]+)",
    re.IGNORECASE,
)
_FILENAME_RE: Final[re.Pattern[str]] = re.compile(
    r'filename="?([^";]+)"?',
    re.IGNORECASE,
)


def _filename_from_content_disposition(content_disposition: str) -> str | None:
    match = _FILENAME_STAR_RE.search(content_disposition)
    if match is not None:
        name = unquote(match.group("name"))
        return Path(name.replace("\r", "").replace("\n", "")).name

    match = _FILENAME_RE.search(content_disposition)
    if match is not None:
        name = match.group(1)
        return Path(name.replace("\r", "").replace("\n", "")).name

    return None


class _DesktopApi:
    def __init__(self, *, base_url: str, webview: Any) -> None:
        self._base_url = base_url.rstrip("/")
        self._webview = webview

    def save_json(self, job_id: str, download_url: str | None = None) -> dict[str, str]:
        url = download_url or f"{self._base_url}/download/{job_id}.json"
        request = Request(url, headers={"User-Agent": "ragprep-desktop"})

        try:
            with urlopen(request, timeout=30.0) as response:
                if response.status != 200:
                    return {"status": "error", "message": f"download failed: {response.status}"}
                json_bytes = response.read()
                content_disposition = response.headers.get("Content-Disposition", "")
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "message": f"download failed: {exc}"}

        filename = _filename_from_content_disposition(content_disposition) or f"{job_id}.json"

        try:
            selection = self._webview.create_file_dialog(
                self._webview.SAVE_DIALOG,
                save_filename=filename,
                file_types=[("JSON (*.json)", "*.json"), ("All files (*.*)", "*.*")],
            )
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "message": f"save dialog failed: {exc}"}

        if not selection:
            return {"status": "cancelled"}

        selected_path = selection[0] if isinstance(selection, (list, tuple)) else selection
        try:
            with open(str(selected_path), "wb") as handle:
                handle.write(json_bytes)
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "message": f"write failed: {exc}"}

        return {"status": "ok", "path": str(selected_path), "filename": filename}

    def save_markdown(self, job_id: str, download_url: str | None = None) -> dict[str, str]:
        url = download_url or f"{self._base_url}/download/{job_id}.md"
        request = Request(url, headers={"User-Agent": "ragprep-desktop"})

        try:
            with urlopen(request, timeout=30.0) as response:
                if response.status != 200:
                    return {"status": "error", "message": f"download failed: {response.status}"}
                markdown_bytes = response.read()
                content_disposition = response.headers.get("Content-Disposition", "")
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "message": f"download failed: {exc}"}

        filename = _filename_from_content_disposition(content_disposition) or f"{job_id}.md"

        try:
            selection = self._webview.create_file_dialog(
                self._webview.SAVE_DIALOG,
                save_filename=filename,
                file_types=[("Markdown (*.md)", "*.md"), ("All files (*.*)", "*.*")],
            )
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "message": f"save dialog failed: {exc}"}

        if not selection:
            return {"status": "cancelled"}

        selected_path = selection[0] if isinstance(selection, (list, tuple)) else selection
        try:
            with open(str(selected_path), "wb") as handle:
                handle.write(markdown_bytes)
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "message": f"write failed: {exc}"}

        return {"status": "ok", "path": str(selected_path), "filename": filename}


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

        base_url = f"http://{client_host}:{args.port}"
        api = _DesktopApi(base_url=base_url, webview=webview)
        web_url = f"{base_url}/"
        webview.create_window("RAGPrep", web_url, js_api=api)
        webview.start()
        return 0
    finally:
        server.should_exit = True
        server_thread.join(timeout=5.0)


if __name__ == "__main__":
    raise SystemExit(main())
