from __future__ import annotations

import io
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from ragprep.config import get_settings
from ragprep.ocr import lighton_server
from ragprep.ocr.lighton_assets import LightOnAssetPaths


class _DummyProcess:
    def __init__(self, *, exit_code: int | None = None, stderr_text: str = "") -> None:
        self._exit_code = exit_code
        self.stderr = io.BytesIO(stderr_text.encode("utf-8"))

    def poll(self) -> int | None:
        return self._exit_code

    def terminate(self) -> None:
        self._exit_code = 0

    def wait(self, timeout: float | None = None) -> int:
        _ = timeout
        self._exit_code = 0
        return 0

    def kill(self) -> None:
        self._exit_code = 1


class _StubResponse:
    def __init__(self, *, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> object:
        return self._payload


class _StubClient:
    def __init__(self, *, responses: dict[str, _StubResponse]) -> None:
        self._responses = responses

    def __enter__(self) -> _StubClient:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        _ = exc_type, exc, tb
        return None

    def get(self, url: str) -> _StubResponse:
        response = self._responses.get(url)
        if response is None:
            raise AssertionError(f"unexpected URL: {url}")
        return response


def _asset_paths(tmp_path: Path) -> LightOnAssetPaths:
    model = tmp_path / "model.gguf"
    mmproj = tmp_path / "mmproj.gguf"
    model.write_bytes(b"m")
    mmproj.write_bytes(b"p")
    return LightOnAssetPaths(model_path=model, mmproj_path=mmproj)


def test_resolve_llama_server_executable_requires_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RAGPREP_LLAMA_SERVER_PATH", raising=False)
    monkeypatch.setattr(lighton_server, "_auto_discover_llama_server_executable", lambda _s: None)
    monkeypatch.setattr(lighton_server, "_auto_install_llama_server_executable", lambda _s: None)
    settings = get_settings()

    with pytest.raises(RuntimeError, match="llama-server not found"):
        _ = lighton_server._resolve_llama_server_executable(settings)


def test_resolve_llama_server_executable_auto_detects_and_sets_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    discovered = tmp_path / "llama-server.exe"
    discovered.write_bytes(b"")
    monkeypatch.delenv("RAGPREP_LLAMA_SERVER_PATH", raising=False)
    monkeypatch.setattr(
        lighton_server,
        "_auto_discover_llama_server_executable",
        lambda _s: str(discovered),
    )
    settings = get_settings()

    resolved = lighton_server._resolve_llama_server_executable(settings)
    assert resolved == str(discovered)
    assert str(os.getenv("RAGPREP_LLAMA_SERVER_PATH")) == str(discovered)


def test_resolve_llama_server_executable_auto_installs_when_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    discovered = tmp_path / "auto" / "llama-server.exe"
    discovered.parent.mkdir(parents=True, exist_ok=True)
    discovered.write_bytes(b"")
    monkeypatch.delenv("RAGPREP_LLAMA_SERVER_PATH", raising=False)
    monkeypatch.setattr(lighton_server, "_auto_discover_llama_server_executable", lambda _s: None)
    monkeypatch.setattr(
        lighton_server,
        "_auto_install_llama_server_executable",
        lambda _s: str(discovered),
    )
    settings = get_settings()

    resolved = lighton_server._resolve_llama_server_executable(settings)
    assert resolved == str(discovered)
    assert str(os.getenv("RAGPREP_LLAMA_SERVER_PATH")) == str(discovered)


def test_select_prebuilt_asset_prefers_windows_avx2() -> None:
    assets: list[dict[str, object]] = [
        {
            "name": "llama-b9999-bin-win-cuda-12.4-x64.zip",
            "browser_download_url": "https://example.com/cuda.zip",
        },
        {
            "name": "llama-b9999-bin-win-avx2-x64.zip",
            "browser_download_url": "https://example.com/avx2.zip",
        },
        {
            "name": "llama-b9999-bin-win-openblas-x64.zip",
            "browser_download_url": "https://example.com/openblas.zip",
        },
    ]

    picked = lighton_server._select_prebuilt_asset(assets, platform_name="win32")
    assert picked is not None
    assert picked["name"] == "llama-b9999-bin-win-avx2-x64.zip"


def test_start_command_contains_required_flags(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = get_settings()
    assets = _asset_paths(tmp_path)
    captured: dict[str, Any] = {}

    def _fake_popen(command: list[str], **kwargs: object) -> _DummyProcess:
        captured["command"] = command
        captured["kwargs"] = kwargs
        return _DummyProcess(exit_code=None)

    monkeypatch.setattr(lighton_server.subprocess, "Popen", _fake_popen)

    attempt = lighton_server._StartAttempt(
        name="gpu-aggressive",
        ngl=-1,
        parallel=4,
        flash_attn=True,
    )
    _ = lighton_server._start_llama_server_process(
        executable="llama-server",
        assets=assets,
        settings=settings,
        attempt=attempt,
    )

    command = captured["command"]
    assert "-m" in command
    assert str(assets.model_path) in command
    assert "--mmproj" in command
    assert str(assets.mmproj_path) in command
    assert "--jinja" in command
    assert "--flash-attn" in command
    assert "-ngl" in command
    assert "-np" in command
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs.get("stderr") is subprocess.DEVNULL


def test_is_server_healthy_requires_models_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    base_url = "http://127.0.0.1:8080"
    monkeypatch.setattr(
        lighton_server.httpx,
        "Client",
        lambda **_kwargs: _StubClient(
            responses={
                f"{base_url}/health": _StubResponse(status_code=200, payload={"ok": True}),
                f"{base_url}/v1/models": _StubResponse(status_code=404, payload={}),
            }
        ),
    )

    assert lighton_server._is_server_healthy(base_url) is False


def test_is_server_healthy_accepts_expected_openai_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_url = "http://127.0.0.1:8080"
    monkeypatch.setattr(
        lighton_server.httpx,
        "Client",
        lambda **_kwargs: _StubClient(
            responses={
                f"{base_url}/health": _StubResponse(status_code=200, payload={"ok": True}),
                f"{base_url}/v1/models": _StubResponse(
                    status_code=200,
                    payload={"data": [{"id": "lighton-ocr"}]},
                ),
            }
        ),
    )

    assert lighton_server._is_server_healthy(base_url) is True


def test_ensure_server_retries_and_falls_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = get_settings()
    attempted: list[str] = []
    outcomes = iter([(False, "gpu1"), (False, "gpu2"), (True, None)])

    monkeypatch.setattr(lighton_server, "_ACTIVE_PROCESS", None)
    monkeypatch.setattr(lighton_server, "_ACTIVE_BASE_URL", None)
    monkeypatch.setattr(lighton_server, "_is_server_healthy", lambda _base: False)
    monkeypatch.setattr(
        lighton_server,
        "ensure_lighton_assets",
        lambda _s, **_kwargs: _asset_paths(tmp_path),
    )
    monkeypatch.setattr(
        lighton_server,
        "_resolve_llama_server_executable",
        lambda _s: "llama-server",
    )

    def _fake_start_llama_server_process(
        *,
        executable: str,
        assets: LightOnAssetPaths,
        settings: object,
        attempt: lighton_server._StartAttempt,
    ) -> _DummyProcess:
        _ = executable, assets, settings
        attempted.append(attempt.name)
        return _DummyProcess(exit_code=None)

    monkeypatch.setattr(
        lighton_server,
        "_start_llama_server_process",
        _fake_start_llama_server_process,
    )
    monkeypatch.setattr(lighton_server, "_wait_for_server_ready", lambda **_kwargs: next(outcomes))

    base = lighton_server.ensure_server_base_url(settings)
    assert base == f"http://{settings.lighton_server_host}:{settings.lighton_server_port}"
    assert attempted == ["gpu-aggressive", "gpu-conservative", "cpu-fallback"]


def test_ensure_server_raises_when_all_attempts_fail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = get_settings()

    monkeypatch.setattr(lighton_server, "_ACTIVE_PROCESS", None)
    monkeypatch.setattr(lighton_server, "_ACTIVE_BASE_URL", None)
    monkeypatch.setattr(lighton_server, "_is_server_healthy", lambda _base: False)
    monkeypatch.setattr(
        lighton_server,
        "ensure_lighton_assets",
        lambda _s, **_kwargs: _asset_paths(tmp_path),
    )
    monkeypatch.setattr(
        lighton_server,
        "_resolve_llama_server_executable",
        lambda _s: "llama-server",
    )
    monkeypatch.setattr(
        lighton_server,
        "_start_llama_server_process",
        lambda **_kwargs: _DummyProcess(exit_code=None),
    )
    monkeypatch.setattr(
        lighton_server,
        "_wait_for_server_ready",
        lambda **_kwargs: (False, "health check timed out"),
    )

    with pytest.raises(RuntimeError, match="Failed to start llama-server"):
        _ = lighton_server.ensure_server_base_url(settings)


def test_build_mmproj_candidates_includes_fallback_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAGPREP_LIGHTON_MMPROJ_FILE", "mmproj-BF16.gguf")
    settings = get_settings()

    candidates = lighton_server._build_mmproj_candidates(settings)
    assert candidates == ["mmproj-BF16.gguf", "mmproj-F32.gguf"]


def test_ensure_server_tries_mmproj_fallback_when_primary_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("RAGPREP_LIGHTON_MMPROJ_FILE", "mmproj-BF16.gguf")
    settings = get_settings()
    tried_mmproj: list[str] = []

    monkeypatch.setattr(lighton_server, "_ACTIVE_PROCESS", None)
    monkeypatch.setattr(lighton_server, "_ACTIVE_BASE_URL", None)
    monkeypatch.setattr(lighton_server, "_is_server_healthy", lambda _base: False)
    monkeypatch.setattr(
        lighton_server,
        "_resolve_llama_server_executable",
        lambda _s: "llama-server",
    )
    monkeypatch.setattr(
        lighton_server,
        "_build_attempts",
        lambda _s: [
            lighton_server._StartAttempt(
                name="gpu-aggressive",
                ngl=0,
                parallel=1,
                flash_attn=False,
            )
        ],
    )

    def _fake_assets(
        _settings: object,
        *,
        mmproj_file: str | None = None,
    ) -> LightOnAssetPaths:
        assert mmproj_file is not None
        model = tmp_path / "model.gguf"
        mmproj_path = tmp_path / mmproj_file
        model.write_bytes(b"m")
        mmproj_path.write_bytes(b"p")
        return LightOnAssetPaths(model_path=model, mmproj_path=mmproj_path)

    def _fake_start_llama_server_process(
        *,
        executable: str,
        assets: LightOnAssetPaths,
        settings: object,
        attempt: lighton_server._StartAttempt,
    ) -> _DummyProcess:
        _ = executable, settings, attempt
        tried_mmproj.append(assets.mmproj_path.name)
        return _DummyProcess(exit_code=None)

    outcomes = iter([(False, "bf16 failed"), (True, None)])
    monkeypatch.setattr(lighton_server, "ensure_lighton_assets", _fake_assets)
    monkeypatch.setattr(
        lighton_server,
        "_start_llama_server_process",
        _fake_start_llama_server_process,
    )
    monkeypatch.setattr(lighton_server, "_wait_for_server_ready", lambda **_kwargs: next(outcomes))

    base = lighton_server.ensure_server_base_url(settings)
    assert base == f"http://{settings.lighton_server_host}:{settings.lighton_server_port}"
    assert tried_mmproj == ["mmproj-BF16.gguf", "mmproj-F32.gguf"]


def test_ensure_server_error_includes_mmproj_names_when_all_fail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("RAGPREP_LIGHTON_MMPROJ_FILE", "mmproj-BF16.gguf")
    settings = get_settings()

    monkeypatch.setattr(lighton_server, "_ACTIVE_PROCESS", None)
    monkeypatch.setattr(lighton_server, "_ACTIVE_BASE_URL", None)
    monkeypatch.setattr(lighton_server, "_is_server_healthy", lambda _base: False)
    monkeypatch.setattr(
        lighton_server,
        "_resolve_llama_server_executable",
        lambda _s: "llama-server",
    )
    monkeypatch.setattr(
        lighton_server,
        "_build_attempts",
        lambda _s: [
            lighton_server._StartAttempt(
                name="gpu-aggressive",
                ngl=0,
                parallel=1,
                flash_attn=False,
            )
        ],
    )

    def _fake_assets(
        _settings: object,
        *,
        mmproj_file: str | None = None,
    ) -> LightOnAssetPaths:
        assert mmproj_file is not None
        model = tmp_path / "model.gguf"
        mmproj_path = tmp_path / mmproj_file
        model.write_bytes(b"m")
        mmproj_path.write_bytes(b"p")
        return LightOnAssetPaths(model_path=model, mmproj_path=mmproj_path)

    monkeypatch.setattr(lighton_server, "ensure_lighton_assets", _fake_assets)
    monkeypatch.setattr(
        lighton_server,
        "_start_llama_server_process",
        lambda **_kwargs: _DummyProcess(exit_code=None),
    )
    monkeypatch.setattr(
        lighton_server,
        "_wait_for_server_ready",
        lambda **_kwargs: (False, "health check timed out"),
    )

    with pytest.raises(RuntimeError, match="mmproj=mmproj-BF16.gguf"):
        _ = lighton_server.ensure_server_base_url(settings)
