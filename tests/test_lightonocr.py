from __future__ import annotations

import importlib

import pytest


def test_ragprep_ocr_package_is_removed() -> None:
    with pytest.raises(ModuleNotFoundError, match=r"ragprep\.ocr"):
        importlib.import_module("ragprep.ocr")


def test_lightonocr_module_is_removed() -> None:
    with pytest.raises(ModuleNotFoundError, match=r"ragprep\.ocr"):
        importlib.import_module("ragprep.ocr.lightonocr")
