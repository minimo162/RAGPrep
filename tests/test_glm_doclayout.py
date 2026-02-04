from __future__ import annotations

import base64
import builtins
import json
import os
import sys
from types import ModuleType
from typing import Any, cast

import httpx
import pytest

from ragprep.config import get_settings
from ragprep.layout import glm_doclayout


def test_glm_doclayout_sends_openai_chat_completions_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAGPREP_LAYOUT_MODE", "server")
    monkeypatch.setenv("RAGPREP_LAYOUT_BASE_URL", "http://localhost:8080/")
    monkeypatch.setenv("RAGPREP_LAYOUT_MODEL", "zai-org/GLM-OCR")
    monkeypatch.setenv("RAGPREP_LAYOUT_API_KEY", "secret")
    monkeypatch.setenv("RAGPREP_LAYOUT_MAX_TOKENS", "123")
    monkeypatch.setenv("RAGPREP_LAYOUT_TIMEOUT_SECONDS", "9")
    settings = get_settings()

    captured: dict[str, object] = {}

    def _fake_post(
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout_seconds: int,
    ) -> httpx.Response:
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = payload
        captured["timeout_seconds"] = timeout_seconds
        content = {
            "schema_version": "v1",
            "elements": [
                {"page_index": 0, "bbox": [1, 2, 3, 4], "label": "text", "score": 0.9},
            ],
        }
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(content)}}]},
        )

    monkeypatch.setattr("ragprep.layout.glm_doclayout._post_chat_completions", _fake_post)

    image_b64 = base64.b64encode(b"not-a-real-png").decode("ascii")
    result = glm_doclayout.analyze_layout_image_base64(image_b64, settings=settings)

    assert result["schema_version"] == "v1"
    assert isinstance(result["elements"], list)
    assert result["elements"][0]["label"] == "text"

    assert captured["url"] == "http://localhost:8080/v1/chat/completions"
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers["Authorization"] == "Bearer secret"

    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "zai-org/GLM-OCR"
    assert payload["max_tokens"] == 123

    messages = payload["messages"]
    assert isinstance(messages, list)
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1]["type"] == "text"
    assert content[1]["text"] == glm_doclayout.DEFAULT_LAYOUT_ANALYSIS_PROMPT

    assert captured["timeout_seconds"] == 9


def test_glm_doclayout_timeout_maps_to_stable_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAGPREP_LAYOUT_MODE", "server")
    monkeypatch.setenv("RAGPREP_LAYOUT_BASE_URL", "http://localhost:8080/")
    monkeypatch.setenv("RAGPREP_LAYOUT_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("RAGPREP_LAYOUT_RETRY_COUNT", "0")
    settings = get_settings()

    def _fake_post(
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout_seconds: int,
    ) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=httpx.Request("POST", url))

    monkeypatch.setattr("ragprep.layout.glm_doclayout._post_chat_completions", _fake_post)

    image_b64 = base64.b64encode(b"not-a-real-png").decode("ascii")
    with pytest.raises(RuntimeError, match=r"Layout analysis request timed out"):
        glm_doclayout.analyze_layout_image_base64(image_b64, settings=settings)


def test_glm_doclayout_retries_timeout_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAGPREP_LAYOUT_MODE", "server")
    monkeypatch.setenv("RAGPREP_LAYOUT_BASE_URL", "http://localhost:8080/")
    monkeypatch.setenv("RAGPREP_LAYOUT_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("RAGPREP_LAYOUT_RETRY_COUNT", "1")
    monkeypatch.setenv("RAGPREP_LAYOUT_RETRY_BACKOFF_SECONDS", "0")
    settings = get_settings()

    calls = 0

    def _fake_post(
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout_seconds: int,
    ) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("timed out", request=httpx.Request("POST", url))
        content = {
            "schema_version": "v1",
            "elements": [
                {"page_index": 0, "bbox": [0, 0, 1, 1], "label": "text", "score": 0.9},
            ],
        }
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(content)}}]},
        )

    monkeypatch.setattr("ragprep.layout.glm_doclayout._post_chat_completions", _fake_post)

    image_b64 = base64.b64encode(b"not-a-real-png").decode("ascii")
    result = glm_doclayout.analyze_layout_image_base64(image_b64, settings=settings)
    assert result["schema_version"] == "v1"
    assert calls == 2


def test_glm_doclayout_retry_exhaustion_raises_timeout_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAGPREP_LAYOUT_MODE", "server")
    monkeypatch.setenv("RAGPREP_LAYOUT_BASE_URL", "http://localhost:8080/")
    monkeypatch.setenv("RAGPREP_LAYOUT_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("RAGPREP_LAYOUT_RETRY_COUNT", "2")
    monkeypatch.setenv("RAGPREP_LAYOUT_RETRY_BACKOFF_SECONDS", "0")
    settings = get_settings()

    calls = 0

    def _fake_post(
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout_seconds: int,
    ) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("timed out", request=httpx.Request("POST", url))

    monkeypatch.setattr("ragprep.layout.glm_doclayout._post_chat_completions", _fake_post)

    image_b64 = base64.b64encode(b"not-a-real-png").decode("ascii")
    with pytest.raises(RuntimeError, match=r"Layout analysis request timed out"):
        glm_doclayout.analyze_layout_image_base64(image_b64, settings=settings)
    assert calls == 3


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
                'PP-StructureV3 requires additional dependencies. Install `paddlex[ocr]`.'
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


def test_glm_doclayout_raises_on_non_200(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_LAYOUT_MODE", "server")
    settings = get_settings()

    def _fake_post(**_kwargs: object) -> httpx.Response:
        return httpx.Response(500, content=b"oops")

    monkeypatch.setattr("ragprep.layout.glm_doclayout._post_chat_completions", _fake_post)

    image_b64 = base64.b64encode(b"x").decode("ascii")
    with pytest.raises(RuntimeError, match="returned 500"):
        glm_doclayout.analyze_layout_image_base64(image_b64, settings=settings)


def test_glm_doclayout_rejects_invalid_base64(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_LAYOUT_MODE", "server")
    settings = get_settings()
    with pytest.raises(ValueError, match="not valid base64"):
        glm_doclayout.analyze_layout_image_base64("not base64 !!!", settings=settings)


def test_glm_doclayout_rejects_non_json_content(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_LAYOUT_MODE", "server")
    settings = get_settings()

    def _fake_post(**_kwargs: object) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "not-json"}}]})

    monkeypatch.setattr("ragprep.layout.glm_doclayout._post_chat_completions", _fake_post)

    image_b64 = base64.b64encode(b"x").decode("ascii")
    with pytest.raises(RuntimeError, match="does not contain a JSON object"):
        glm_doclayout.analyze_layout_image_base64(image_b64, settings=settings)


def test_glm_doclayout_parses_fenced_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_LAYOUT_MODE", "server")
    settings = get_settings()

    fenced = """```json
{"schema_version":"v1","elements":[{"page_index":0,"bbox":[1,2,3,4],"label":"text","score":0.9}]}
```"""

    def _fake_post(**_kwargs: object) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": fenced}}]})

    monkeypatch.setattr("ragprep.layout.glm_doclayout._post_chat_completions", _fake_post)

    image_b64 = base64.b64encode(b"x").decode("ascii")
    result = glm_doclayout.analyze_layout_image_base64(image_b64, settings=settings)
    assert result["schema_version"] == "v1"
    elements = result["elements"]
    assert isinstance(elements, list)
    assert isinstance(elements[0], dict)
    assert elements[0]["label"] == "text"


def test_glm_doclayout_extracts_json_from_surrounding_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_LAYOUT_MODE", "server")
    settings = get_settings()

    content = (
        "Here is the result:\n"
        '{"schema_version":"v1","elements":[{"page_index":0,"bbox":[1,2,3,4],"label":"text"}]}\n'
        "done"
    )

    def _fake_post(**_kwargs: object) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    monkeypatch.setattr("ragprep.layout.glm_doclayout._post_chat_completions", _fake_post)

    image_b64 = base64.b64encode(b"x").decode("ascii")
    result = glm_doclayout.analyze_layout_image_base64(image_b64, settings=settings)
    assert result["schema_version"] == "v1"
    elements = result["elements"]
    assert isinstance(elements, list)
    assert isinstance(elements[0], dict)
    assert elements[0]["page_index"] == 0
