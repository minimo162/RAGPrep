from __future__ import annotations

from pathlib import Path

import pytest

from ragprep.config import get_settings
from ragprep.ocr import lighton_assets


def test_ensure_lighton_assets_uses_existing_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("RAGPREP_LIGHTON_MODEL_DIR", str(tmp_path))
    settings = get_settings()

    model_path = tmp_path / settings.lighton_model_file
    mmproj_path = tmp_path / settings.lighton_mmproj_file
    model_path.write_bytes(b"model")
    mmproj_path.write_bytes(b"mmproj")

    monkeypatch.setattr(
        lighton_assets,
        "_download_hf_file",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("download must not run")),
    )

    result = lighton_assets.ensure_lighton_assets(settings)
    assert result.model_path == model_path
    assert result.mmproj_path == mmproj_path


def test_ensure_lighton_assets_downloads_when_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("RAGPREP_LIGHTON_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("RAGPREP_LIGHTON_AUTO_DOWNLOAD", "1")
    settings = get_settings()

    downloaded: list[str] = []

    def _fake_download_hf_file(
        *,
        repo_id: str,
        filename: str,
        target_path: Path,
        token: str | None,
        timeout_seconds: float,
    ) -> None:
        _ = repo_id, token, timeout_seconds
        downloaded.append(filename)
        target_path.write_bytes(b"ok")

    monkeypatch.setattr(lighton_assets, "_download_hf_file", _fake_download_hf_file)

    result = lighton_assets.ensure_lighton_assets(settings)
    assert result.model_path.exists()
    assert result.mmproj_path.exists()
    assert set(downloaded) == {settings.lighton_model_file, settings.lighton_mmproj_file}


def test_ensure_lighton_assets_rejects_missing_when_auto_download_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("RAGPREP_LIGHTON_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("RAGPREP_LIGHTON_AUTO_DOWNLOAD", "0")
    settings = get_settings()

    with pytest.raises(RuntimeError, match="auto-download is disabled"):
        _ = lighton_assets.ensure_lighton_assets(settings)


def test_ensure_lighton_assets_accepts_mmproj_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("RAGPREP_LIGHTON_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("RAGPREP_LIGHTON_AUTO_DOWNLOAD", "1")
    settings = get_settings()
    (tmp_path / settings.lighton_model_file).write_bytes(b"model")

    downloaded: list[str] = []

    def _fake_download_hf_file(
        *,
        repo_id: str,
        filename: str,
        target_path: Path,
        token: str | None,
        timeout_seconds: float,
    ) -> None:
        _ = repo_id, token, timeout_seconds
        downloaded.append(filename)
        target_path.write_bytes(b"ok")

    monkeypatch.setattr(lighton_assets, "_download_hf_file", _fake_download_hf_file)

    result = lighton_assets.ensure_lighton_assets(settings, mmproj_file="mmproj-F32.gguf")
    assert result.mmproj_path.name == "mmproj-F32.gguf"
    assert "mmproj-F32.gguf" in downloaded
