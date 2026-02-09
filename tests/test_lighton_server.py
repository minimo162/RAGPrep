from __future__ import annotations

import io
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
    monkeypatch.setattr(lighton_server.shutil, "which", lambda _name: None)
    settings = get_settings()

    with pytest.raises(RuntimeError, match="llama-server not found"):
        _ = lighton_server._resolve_llama_server_executable(settings)


def test_start_command_contains_required_flags(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = get_settings()
    assets = _asset_paths(tmp_path)
    captured: dict[str, Any] = {}

    def _fake_popen(command: list[str], **_kwargs: object) -> _DummyProcess:
        captured["command"] = command
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
        lambda _s: _asset_paths(tmp_path),
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
        lambda _s: _asset_paths(tmp_path),
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
