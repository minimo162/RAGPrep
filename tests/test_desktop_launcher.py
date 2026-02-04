from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, TracebackType

import pytest
import uvicorn

from ragprep import desktop
from ragprep.web.app import app


class _FakeResponse:
    def __init__(self, *, status: int, body: bytes, headers: dict[str, str] | None = None) -> None:
        self.status = status
        self._body = body
        self.headers = headers or {}

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        _ = exc_type
        _ = exc
        _ = tb
        return None

    def read(self) -> bytes:
        return self._body


def test_health_endpoint_returns_ok() -> None:
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.text == "ok"


def test_client_host_for_binds_all_interfaces_to_loopback() -> None:
    assert desktop._client_host_for("0.0.0.0") == "127.0.0.1"
    assert desktop._client_host_for("::") == "127.0.0.1"
    assert desktop._client_host_for("127.0.0.1") == "127.0.0.1"


def test_desktop_launcher_calls_webview_and_stops_server(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubServer:
        last_instance: StubServer | None = None

        def __init__(self, config: uvicorn.Config) -> None:
            self.config = config
            self.should_exit = False
            self.run_called = False
            StubServer.last_instance = self

        def run(self) -> None:
            self.run_called = True

    webview_stub = ModuleType("webview")
    calls: dict[str, object] = {}

    def create_window(title: str, url: str, *, js_api: object | None = None) -> None:
        calls["title"] = title
        calls["url"] = url
        calls["js_api"] = js_api

    def start() -> None:
        calls["started"] = True

    webview_stub.create_window = create_window  # type: ignore[attr-defined]
    webview_stub.start = start  # type: ignore[attr-defined]

    monkeypatch.setattr(desktop, "_ThreadedUvicornServer", StubServer)
    monkeypatch.setattr(desktop, "_wait_for_health", lambda *, health_url, timeout_s: True)
    monkeypatch.setitem(sys.modules, "webview", webview_stub)

    result = desktop.main([])
    assert result == 0

    assert calls["title"] == "RAGPrep"
    assert calls["url"] == "http://127.0.0.1:8000/"
    assert calls["started"] is True
    assert calls["js_api"] is not None

    server = StubServer.last_instance
    assert server is not None
    assert server.run_called is True
    assert server.should_exit is True


def test_desktop_launcher_exits_when_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubServer:
        last_instance: StubServer | None = None

        def __init__(self, config: uvicorn.Config) -> None:
            self.config = config
            self.should_exit = False
            StubServer.last_instance = self

        def run(self) -> None:
            return

    monkeypatch.setattr(desktop, "_ThreadedUvicornServer", StubServer)
    monkeypatch.setattr(desktop, "_wait_for_health", lambda *, health_url, timeout_s: False)

    result = desktop.main([])
    assert result == 1

    server = StubServer.last_instance
    assert server is not None
    assert server.should_exit is True


def test_save_html_writes_to_downloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    downloads_dir = tmp_path / "Downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)

    def _fake_urlopen(_request: object, timeout: float = 0.0) -> _FakeResponse:
        _ = timeout
        return _FakeResponse(
            status=200,
            body=b"hello",
            headers={"Content-Disposition": 'attachment; filename="report.html"'},
        )

    dialog_called = {"value": False}

    class WebviewStub:
        SAVE_DIALOG = object()

        def create_file_dialog(self, *args: object, **kwargs: object) -> None:
            dialog_called["value"] = True
            raise AssertionError("save dialog should not be called when downloads is available")

    monkeypatch.setattr(desktop, "urlopen", _fake_urlopen)
    monkeypatch.setattr(desktop, "_resolve_downloads_dir", lambda: downloads_dir)

    api = desktop._DesktopApi(base_url="http://127.0.0.1:8000", webview=WebviewStub())
    result = api.save_html("job123", "http://127.0.0.1:8000/download/job123.html")

    saved_path = downloads_dir / "report.html"
    assert result["status"] == "ok"
    assert result["path"] == str(saved_path)
    assert saved_path.exists()
    assert saved_path.read_bytes() == b"hello"
    assert dialog_called["value"] is False


def test_save_html_renames_on_collision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    downloads_dir = tmp_path / "Downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    existing_path = downloads_dir / "report.html"
    existing_path.write_bytes(b"old")

    def _fake_urlopen(_request: object, timeout: float = 0.0) -> _FakeResponse:
        _ = timeout
        return _FakeResponse(
            status=200,
            body=b"new",
            headers={"Content-Disposition": 'attachment; filename="report.html"'},
        )

    class WebviewStub:
        SAVE_DIALOG = object()

        def create_file_dialog(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("save dialog should not be called when downloads is available")

    monkeypatch.setattr(desktop, "urlopen", _fake_urlopen)
    monkeypatch.setattr(desktop, "_resolve_downloads_dir", lambda: downloads_dir)

    api = desktop._DesktopApi(base_url="http://127.0.0.1:8000", webview=WebviewStub())
    result = api.save_html("job123", "http://127.0.0.1:8000/download/job123.html")

    renamed_path = downloads_dir / "report (1).html"
    assert result["status"] == "ok"
    assert result["path"] == str(renamed_path)
    assert existing_path.read_bytes() == b"old"
    assert renamed_path.exists()
    assert renamed_path.read_bytes() == b"new"

