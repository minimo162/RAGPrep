from __future__ import annotations

import sys
from types import ModuleType

import pytest
import uvicorn

from ragprep import desktop
from ragprep.web.app import app


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

