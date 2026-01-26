from __future__ import annotations

import pytest


def test_lightonocr_module_is_importable() -> None:
    from ragprep.ocr import lightonocr

    assert callable(lightonocr.ocr_image)


def test_lightonocr_dry_run_returns_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    from PIL import Image

    from ragprep.ocr import lightonocr

    monkeypatch.setenv("LIGHTONOCR_DRY_RUN", "1")
    image = Image.new("RGB", (1, 1), color=(255, 255, 255))
    assert lightonocr.ocr_image(image) == lightonocr.DRY_RUN_OUTPUT
