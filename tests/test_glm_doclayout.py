from __future__ import annotations

import builtins
import os
import subprocess
import sys
import warnings
from contextlib import contextmanager
from dataclasses import replace
from types import ModuleType
from typing import Any, cast

import pytest

from ragprep.config import get_settings
from ragprep.layout import glm_doclayout


def test_glm_doclayout_local_mode_requires_optional_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_LAYOUT_MODE", "local-paddle")
    settings = get_settings()

    # Force ImportError even if paddleocr is installed on the machine running tests.
    orig_import = cast(Any, builtins.__import__)

    def _fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "paddleocr" or name.startswith("paddleocr."):
            raise ImportError("blocked for test")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    glm_doclayout._get_paddleocr_engine.cache_clear()

    image_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/"
        "mGQAAAAASUVORK5CYII="
    )
    with pytest.raises(RuntimeError, match="Install PaddleOCR"):
        glm_doclayout.analyze_layout_image_base64(image_b64, settings=settings)


def test_analyze_layout_rejects_local_fast_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_LAYOUT_MODE", "local-fast")
    settings = get_settings()
    with pytest.raises(RuntimeError, match="does not use image layout API"):
        glm_doclayout.analyze_layout_image_base64("aGVsbG8=", settings=settings)


def test_analyze_layout_rejects_unknown_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    bad_settings = replace(settings, layout_mode="unknown")
    with pytest.raises(RuntimeError, match="RAGPREP_LAYOUT_MODE=local-paddle"):
        glm_doclayout.analyze_layout_image_base64("aGVsbG8=", settings=bad_settings)


def test_load_paddleocr_ppstructure_falls_back_to_v3(monkeypatch: pytest.MonkeyPatch) -> None:
    class _StubPPStructureV3:
        def __init__(self, **_kwargs: object) -> None:
            return

    stub = ModuleType("paddleocr")
    stub.PPStructureV3 = _StubPPStructureV3  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "paddleocr", stub)
    glm_doclayout._get_paddleocr_engine.cache_clear()

    loaded = glm_doclayout._load_paddleocr_ppstructure()
    assert loaded is _StubPPStructureV3


def test_suppress_paddle_ccache_probe_noise_silences_ccache_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_check_output(*popenargs: object, **kwargs: object) -> object:
        captured["kwargs"] = dict(kwargs)
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd="where ccache",
        )

    monkeypatch.setattr(subprocess, "check_output", _fake_check_output)

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        with glm_doclayout._suppress_paddle_ccache_probe_noise():
            with pytest.raises(subprocess.CalledProcessError):
                subprocess.check_output(["where", "ccache"])
            warnings.warn(
                "No ccache found. Please be aware that recompiling all source files may be "
                "required. You can download and install ccache from: x",
                UserWarning,
                stacklevel=2,
            )
            warnings.warn("other warning", UserWarning, stacklevel=2)

    assert captured
    kwargs = cast(dict[str, object], captured["kwargs"])
    assert kwargs.get("stderr") is subprocess.DEVNULL
    assert not any("No ccache found" in str(entry.message) for entry in records)
    assert any("other warning" in str(entry.message) for entry in records)


def test_load_paddleocr_ppstructure_uses_ccache_noise_suppression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StubPPStructureV3:
        def __init__(self, **_kwargs: object) -> None:
            return

    stub = ModuleType("paddleocr")
    stub.PPStructureV3 = _StubPPStructureV3  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "paddleocr", stub)

    calls: list[str] = []

    @contextmanager
    def _fake_suppressor() -> Any:
        calls.append("enter")
        try:
            yield
        finally:
            calls.append("exit")

    monkeypatch.setattr(glm_doclayout, "_suppress_paddle_ccache_probe_noise", _fake_suppressor)
    glm_doclayout._get_paddleocr_engine.cache_clear()

    loaded = glm_doclayout._load_paddleocr_ppstructure()
    assert loaded is _StubPPStructureV3
    assert calls == ["enter", "exit"]


def test_filter_supported_constructor_kwargs_drops_unsupported_names() -> None:
    class StubEngine:
        def __init__(self, *, device: str, layout: bool, ocr: bool) -> None:
            _ = device, layout, ocr

    kwargs = {
        "device": "cpu",
        "layout": True,
        "ocr": False,
        "show_log": False,
        "table": False,
    }
    filtered = glm_doclayout._filter_supported_constructor_kwargs(StubEngine, kwargs)
    assert filtered == {"device": "cpu", "layout": True, "ocr": False}


def test_filter_supported_constructor_kwargs_keeps_kwargs_for_var_keyword() -> None:
    class StubEngine:
        def __init__(self, **kwargs: object) -> None:
            _ = kwargs

    kwargs = {"device": "cpu", "layout": True, "show_log": False}
    filtered = glm_doclayout._filter_supported_constructor_kwargs(StubEngine, kwargs)
    assert filtered == kwargs


def test_get_paddleocr_engine_retries_on_unknown_show_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StubPPStructureV3:
        last_kwargs: dict[str, object] | None = None

        def __init__(self, **kwargs: object) -> None:
            _StubPPStructureV3.last_kwargs = dict(kwargs)
            if "show_log" in kwargs:
                raise ValueError("Unknown argument: show_log")

    stub = ModuleType("paddleocr")
    stub.PPStructureV3 = _StubPPStructureV3  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "paddleocr", stub)
    glm_doclayout._get_paddleocr_engine.cache_clear()

    engine = glm_doclayout._get_paddleocr_engine()
    assert isinstance(engine, _StubPPStructureV3)
    assert _StubPPStructureV3.last_kwargs is not None
    assert "show_log" not in _StubPPStructureV3.last_kwargs


def test_get_paddleocr_engine_defaults_to_cpu_and_disables_hpi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StubPPStructureV3:
        last_kwargs: dict[str, object] | None = None

        def __init__(self, **kwargs: object) -> None:
            if "show_log" in kwargs:
                raise ValueError("Unknown argument: show_log")
            _StubPPStructureV3.last_kwargs = dict(kwargs)
            if kwargs.get("device") != "cpu":
                raise AssertionError("expected device=cpu")
            if kwargs.get("enable_hpi") is not False:
                raise AssertionError("expected enable_hpi=False")

    stub = ModuleType("paddleocr")
    stub.PPStructureV3 = _StubPPStructureV3  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "paddleocr", stub)
    glm_doclayout._get_paddleocr_engine.cache_clear()

    engine = glm_doclayout._get_paddleocr_engine()
    assert isinstance(engine, _StubPPStructureV3)
    assert _StubPPStructureV3.last_kwargs is not None
    assert _StubPPStructureV3.last_kwargs.get("device") == "cpu"
    assert _StubPPStructureV3.last_kwargs.get("enable_hpi") is False


def test_get_paddleocr_engine_instructs_paddlex_ocr_extras(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _DependencyError(Exception):
        __module__ = "paddlex.utils.deps"

    class _StubPPStructureV3:
        def __init__(self, **_kwargs: object) -> None:
            err = _DependencyError(
                "PP-StructureV3 requires additional dependencies. Install `paddlex[ocr]`."
            )
            raise RuntimeError(
                "A dependency error occurred during pipeline creation. Please refer to the "
                "installation documentation to ensure all required dependencies are installed."
            ) from err

    stub = ModuleType("paddleocr")
    stub.PPStructureV3 = _StubPPStructureV3  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "paddleocr", stub)
    glm_doclayout._get_paddleocr_engine.cache_clear()

    with pytest.raises(RuntimeError, match=r"paddlex\[ocr\]"):
        glm_doclayout._get_paddleocr_engine()


def test_invoke_paddle_engine_uses_predict_when_not_callable() -> None:
    class Engine:
        def predict(self, _image: object) -> object:
            return [{"bbox": [0, 0, 1, 1], "type": "text"}]

    out = glm_doclayout._invoke_paddle_engine(Engine(), image=object())
    assert isinstance(out, list)


def test_invoke_paddle_engine_falls_back_to_list_argument() -> None:
    class Engine:
        last_arg: object | None = None

        def predict(self, image: object) -> object:
            Engine.last_arg = image
            if not isinstance(image, list):
                raise TypeError("expected list")
            return [{"bbox": [0, 0, 1, 1], "type": "text"}]

    out = glm_doclayout._invoke_paddle_engine(Engine(), image=object())
    assert isinstance(out, list)
    assert isinstance(Engine.last_arg, list)


def test_normalize_paddle_layout_output_flattens_singleton_list() -> None:
    items, raw = glm_doclayout._normalize_paddle_layout_output(
        [[{"bbox": [0, 0, 1, 1], "type": "text"}]]
    )
    assert isinstance(items, list)
    assert isinstance(raw, list)
    assert isinstance(items[0], dict)


def test_normalize_paddle_layout_output_extracts_boxes_from_paddlex_detresult_json() -> None:
    class StubDetResult:
        json = {
            "res": {
                "boxes": [
                    {
                        "label": "text",
                        "score": 0.5,
                        "coordinate": [0.0, 1.0, 2.0, 3.0],
                    }
                ]
            }
        }

    items, raw = glm_doclayout._normalize_paddle_layout_output(
        [{"layout_det_res": StubDetResult()}]
    )
    assert isinstance(raw, list)
    assert items == [
        {"bbox": [0.0, 1.0, 2.0, 3.0], "type": "text", "score": 0.5},
    ]


def test_invoke_paddle_engine_for_layout_instructs_on_pir_onednn_error() -> None:
    class Engine:
        def predict(self, _image: object) -> object:
            raise NotImplementedError(
                "(Unimplemented) ConvertPirAttribute2RuntimeAttribute not support "
                "[pir::ArrayAttribute<pir::DoubleAttribute>] (at onednn_instruction.cc:118)"
            )

    with pytest.raises(RuntimeError, match=r"FLAGS_use_mkldnn"):
        glm_doclayout._invoke_paddle_engine_for_layout(Engine(), image=object())


def test_apply_paddle_safe_mode_env_sets_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_LAYOUT_PADDLE_SAFE_MODE", "1")
    monkeypatch.delenv("FLAGS_use_mkldnn", raising=False)
    monkeypatch.delenv("FLAGS_enable_pir_api", raising=False)

    glm_doclayout._apply_paddle_safe_mode_env()

    assert os.environ["FLAGS_use_mkldnn"] == "0"
    assert os.environ["FLAGS_enable_pir_api"] == "0"


def test_apply_paddle_safe_mode_env_does_not_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_LAYOUT_PADDLE_SAFE_MODE", "true")
    monkeypatch.setenv("FLAGS_use_mkldnn", "1")
    monkeypatch.setenv("FLAGS_enable_pir_api", "1")

    glm_doclayout._apply_paddle_safe_mode_env()

    assert os.environ["FLAGS_use_mkldnn"] == "1"
    assert os.environ["FLAGS_enable_pir_api"] == "1"


def test_prewarm_layout_backend_initializes_local_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAGPREP_LAYOUT_MODE", "local-paddle")
    settings = get_settings()
    calls: list[str] = []

    monkeypatch.setattr(
        glm_doclayout,
        "configure_model_cache",
        lambda _settings: calls.append("cache"),
    )
    monkeypatch.setattr(glm_doclayout, "_apply_paddle_safe_mode_env", lambda: calls.append("env"))
    monkeypatch.setattr(glm_doclayout, "_get_paddleocr_engine", lambda: calls.append("engine"))

    glm_doclayout.prewarm_layout_backend(settings=settings)
    assert calls == ["cache", "env", "engine"]


def test_prewarm_layout_backend_rejects_local_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAGPREP_LAYOUT_MODE", "local-fast")
    settings = get_settings()
    with pytest.raises(RuntimeError, match="RAGPREP_LAYOUT_MODE=local-paddle"):
        glm_doclayout.prewarm_layout_backend(settings=settings)
