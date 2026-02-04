from __future__ import annotations

import argparse
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Final
from urllib.parse import unquote
from urllib.request import Request, urlopen

import httpx
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
    headers = {"User-Agent": "ragprep-desktop"}
    timeout = httpx.Timeout(1.0)

    # Desktop environments often have proxy env vars configured. We must not send loopback
    # readiness checks through a proxy.
    with httpx.Client(timeout=timeout, follow_redirects=False, trust_env=False) as client:
        while time.monotonic() < deadline:
            try:
                response = client.get(health_url, headers=headers)
                if response.status_code == 200 and (response.text or "").strip() == "ok":
                    return True
            except httpx.HTTPError:
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


def _windows_known_folder_downloads() -> Path | None:
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class GUID(ctypes.Structure):
            _fields_ = [
                ("Data1", wintypes.DWORD),
                ("Data2", wintypes.WORD),
                ("Data3", wintypes.WORD),
                ("Data4", ctypes.c_ubyte * 8),
            ]

        folder_id_downloads = GUID(
            0x374DE290,
            0x123F,
            0x4565,
            (ctypes.c_ubyte * 8)(0x91, 0x64, 0x39, 0xC4, 0x92, 0x5E, 0x46, 0x7B),
        )
        path_ptr = ctypes.c_wchar_p()
        result = ctypes.windll.shell32.SHGetKnownFolderPath(
            ctypes.byref(folder_id_downloads),
            0,
            0,
            ctypes.byref(path_ptr),
        )
        if result != 0:
            return None
        try:
            value = path_ptr.value
            if not value:
                return None
            return Path(value)
        finally:
            ctypes.windll.ole32.CoTaskMemFree(path_ptr)
    except Exception:  # noqa: BLE001
        return None


def _resolve_downloads_dir() -> Path | None:
    candidates: list[Path] = []
    known = _windows_known_folder_downloads()
    if known is not None:
        candidates.append(known)
    candidates.append(Path.home() / "Downloads")

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
        except Exception:  # noqa: BLE001
            continue
        if candidate.is_dir():
            return candidate
    return None


def _unique_download_path(directory: Path, filename: str) -> Path:
    safe_name = Path(filename).name
    base = directory / safe_name
    if not base.exists():
        return base
    stem = base.stem
    suffix = base.suffix
    for index in range(1, 1000):
        candidate = directory / f"{stem} ({index}){suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("too many existing files in downloads directory")


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

    def save_html(self, job_id: str, download_url: str | None = None) -> dict[str, str]:
        url = download_url or f"{self._base_url}/download/{job_id}.html"
        request = Request(url, headers={"User-Agent": "ragprep-desktop"})

        try:
            with urlopen(request, timeout=30.0) as response:
                if response.status != 200:
                    return {"status": "error", "message": f"download failed: {response.status}"}
                html_bytes = response.read()
                content_disposition = response.headers.get("Content-Disposition", "")
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "message": f"download failed: {exc}"}

        filename = _filename_from_content_disposition(content_disposition) or f"{job_id}.html"

        downloads_error: Exception | None = None
        downloads_dir = _resolve_downloads_dir()
        if downloads_dir is not None:
            try:
                target_path = _unique_download_path(downloads_dir, filename)
                target_path.write_bytes(html_bytes)
                return {
                    "status": "ok",
                    "path": str(target_path),
                    "filename": target_path.name,
                }
            except Exception as exc:  # noqa: BLE001
                downloads_error = exc

        try:
            selection = self._webview.create_file_dialog(
                self._webview.SAVE_DIALOG,
                save_filename=filename,
                file_types=[("HTML (*.html)", "*.html"), ("All files (*.*)", "*.*")],
            )
        except Exception as exc:  # noqa: BLE001
            if downloads_error is not None:
                return {
                    "status": "error",
                    "message": (
                        "downloads save failed: "
                        f"{downloads_error}; save dialog failed: {exc}"
                    ),
                }
            return {"status": "error", "message": f"save dialog failed: {exc}"}

        if not selection:
            return {"status": "cancelled"}

        selected_path = selection[0] if isinstance(selection, (list, tuple)) else selection
        try:
            with open(str(selected_path), "wb") as handle:
                handle.write(html_bytes)
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

        downloads_error: Exception | None = None
        downloads_dir = _resolve_downloads_dir()
        if downloads_dir is not None:
            try:
                target_path = _unique_download_path(downloads_dir, filename)
                target_path.write_bytes(markdown_bytes)
                return {
                    "status": "ok",
                    "path": str(target_path),
                    "filename": target_path.name,
                }
            except Exception as exc:  # noqa: BLE001
                downloads_error = exc

        try:
            selection = self._webview.create_file_dialog(
                self._webview.SAVE_DIALOG,
                save_filename=filename,
                file_types=[("Markdown (*.md)", "*.md"), ("All files (*.*)", "*.*")],
            )
        except Exception as exc:  # noqa: BLE001
            if downloads_error is not None:
                return {
                    "status": "error",
                    "message": (
                        "downloads save failed: "
                        f"{downloads_error}; save dialog failed: {exc}"
                    ),
                }
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
