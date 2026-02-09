from __future__ import annotations

from dataclasses import replace

import pytest

from ragprep.config import get_settings
from ragprep.ocr import glm_ocr


def test_glm_ocr_uses_transformers_mode_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAGPREP_GLM_OCR_MODE", raising=False)
    settings = get_settings()

    monkeypatch.setattr(
        "ragprep.ocr.glm_ocr._ocr_image_base64_via_transformers",
        lambda _image_b64, *, settings: f"mode={settings.glm_ocr_mode or 'transformers'}",
    )

    result = glm_ocr.ocr_image_base64("aGVsbG8=", settings=settings)
    assert result == "mode=transformers"


def test_glm_ocr_uses_transformers_mode_when_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_GLM_OCR_MODE", "transformers")
    settings = get_settings()

    monkeypatch.setattr(
        "ragprep.ocr.glm_ocr._ocr_image_base64_via_transformers",
        lambda _image_b64, *, settings: f"mode={settings.glm_ocr_mode}",
    )

    result = glm_ocr.ocr_image_base64("aGVsbG8=", settings=settings)
    assert result == "mode=transformers"


def test_glm_ocr_rejects_unsupported_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = replace(get_settings(), glm_ocr_mode="invalid")
    with pytest.raises(RuntimeError, match="Unsupported GLM-OCR mode"):
        glm_ocr.ocr_image_base64("aGVsbG8=", settings=settings)


def test_glm_ocr_rejects_invalid_base64() -> None:
    settings = get_settings()
    with pytest.raises(ValueError, match="not valid base64"):
        glm_ocr.ocr_image_base64("not base64 !!!", settings=settings)
