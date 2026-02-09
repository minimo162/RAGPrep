from __future__ import annotations

import base64
import binascii
import inspect
import io
import json
import os
import subprocess
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, cast

from ragprep.config import Settings
from ragprep.model_cache import configure_model_cache

ENV_LAYOUT_PADDLE_SAFE_MODE = "RAGPREP_LAYOUT_PADDLE_SAFE_MODE"

def _strip_data_url_prefix(value: str) -> str:
    raw = value.strip()
    if not raw:
        return ""
    if raw.startswith("data:"):
        comma_index = raw.find(",")
        if comma_index != -1:
            return raw[comma_index + 1 :].strip()
    return raw


def _validate_base64_payload(payload: str) -> None:
    try:
        decoded = base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image_base64 is not valid base64") from exc
    if not decoded:
        raise ValueError("image_base64 is empty")


class _GlmOcrMode:
    transformers = "transformers"
    local_fast = "local-fast"
    local_paddle = "local-paddle"


def _is_ccache_probe_command(command: object) -> bool:
    if isinstance(command, (list, tuple)) and len(command) >= 2:
        executable = str(command[0]).strip().lower()
        target = str(command[1]).strip().lower()
        return executable in {"where", "which"} and target == "ccache"

    if isinstance(command, str):
        normalized = " ".join(command.strip().lower().split())
        return normalized in {"where ccache", "which ccache"}

    return False


@contextmanager
def _suppress_paddle_ccache_probe_noise() -> Any:
    """
    Suppress known non-fatal ccache probe noise emitted during Paddle import.

    On Windows, `where ccache` writes a localized "not found" message to stderr.
    Paddle also emits a `UserWarning` when ccache is absent. Both are informational;
    local layout inference works without ccache.
    """

    original_check_output = cast(Callable[..., Any], subprocess.check_output)

    def _quiet_check_output(*popenargs: Any, **kwargs: Any) -> Any:
        command = kwargs.get("args")
        if command is None and popenargs:
            command = popenargs[0]
        if _is_ccache_probe_command(command):
            patched_kwargs = dict(kwargs)
            patched_kwargs.setdefault("stderr", subprocess.DEVNULL)
            return original_check_output(*popenargs, **patched_kwargs)
        return original_check_output(*popenargs, **kwargs)

    subprocess.check_output = cast(Any, _quiet_check_output)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"No ccache found\..*",
                category=UserWarning,
            )
            yield
    finally:
        subprocess.check_output = cast(Any, original_check_output)


def _load_paddleocr_ppstructure() -> Any:
    with _suppress_paddle_ccache_probe_noise():
        try:
            from paddleocr import PPStructure

            return PPStructure
        except ImportError:
            # Some PaddleOCR versions expose PPStructureV3 instead of PPStructure.
            try:
                from paddleocr import PPStructureV3

                return PPStructureV3
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "Local layout analysis requires PaddleOCR with PPStructure/PPStructureV3 "
                    "available. Install PaddleOCR (CPU) and try again. Suggested install: "
                    "uv pip install paddlepaddle paddleocr"
                ) from exc
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Local layout analysis requires optional dependencies. "
                "Install PaddleOCR (CPU) and try again. "
                "Suggested install: uv pip install paddlepaddle paddleocr"
            ) from exc


def _try_get_dist_version(dist_name: str) -> str | None:
    try:
        from importlib.metadata import version
    except Exception:  # noqa: BLE001
        return None

    try:
        return str(version(dist_name))
    except Exception:  # noqa: BLE001
        return None


def _is_paddlex_dependency_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    if "dependency error occurred during pipeline creation" in message:
        return True

    cause = getattr(exc, "__cause__", None)
    if cause is None:
        return False

    cause_mod = getattr(cause.__class__, "__module__", "") or ""
    cause_name = getattr(cause.__class__, "__name__", "") or ""
    if cause_mod.startswith("paddlex.") and cause_name == "DependencyError":
        return True

    cause_message = str(cause).lower()
    if "paddlex" in cause_message and "requires additional dependencies" in cause_message:
        return True
    return False


def _paddlex_ocr_install_hint() -> str:
    paddlex_version = _try_get_dist_version("paddlex")
    if paddlex_version:
        return f'uv pip install "paddlex[ocr]=={paddlex_version}"'
    return 'uv pip install "paddlex[ocr]"'


def _is_truthy_env(name: str) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return False
    value = raw.strip().lower()
    return value not in {"", "0", "false", "no", "off"}


def _apply_paddle_safe_mode_env() -> None:
    if not _is_truthy_env(ENV_LAYOUT_PADDLE_SAFE_MODE):
        return

    # These should be set before importing/initializing paddle/paddleocr.
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("FLAGS_enable_pir_api", "0")


def _is_paddle_pir_onednn_error(exc: BaseException) -> bool:
    message = str(exc)
    if "ConvertPirAttribute2RuntimeAttribute" in message:
        return True
    if "onednn_instruction.cc" in message:
        return True
    return False


def _paddle_pir_onednn_workaround_hint() -> str:
    return "\n".join(
        [
            "This appears to be a PaddlePaddle runtime limitation (PIR/OneDNN). Try disabling",
            "OneDNN/MKLDNN and PIR API flags, then restart the app:",
            "",
            "PowerShell:",
            "  $env:FLAGS_use_mkldnn = \"0\"",
            "  $env:FLAGS_enable_pir_api = \"0\"",
            "",
            "bash:",
            "  export FLAGS_use_mkldnn=0",
            "  export FLAGS_enable_pir_api=0",
            "",
            "Or set a single switch and restart:",
            f"  {ENV_LAYOUT_PADDLE_SAFE_MODE}=1",
        ]
    )


def _invoke_paddle_engine_for_layout(engine: object, image: object) -> object:
    try:
        return _invoke_paddle_engine(engine, image)
    except (NotImplementedError, RuntimeError) as exc:
        if _is_paddle_pir_onednn_error(exc):
            raise RuntimeError(_paddle_pir_onednn_workaround_hint()) from exc
        raise


def _invoke_paddle_engine(engine: object, image: object) -> object:
    if callable(engine):
        return engine(image)

    tried: list[str] = []
    for method in ("predict", "run", "infer", "process"):
        fn = getattr(engine, method, None)
        if not callable(fn):
            continue
        tried.append(method)

        try:
            return fn(image)
        except TypeError as exc:
            # Some versions expect a list of images.
            try:
                return fn([image])
            except TypeError:
                raise exc from None

    engine_type = type(engine).__name__
    tried_str = ", ".join(tried) if tried else "(none found)"
    raise RuntimeError(
        f"Unsupported PaddleOCR engine {engine_type!r}: not callable and no supported methods "
        f"found (tried: {tried_str})."
    )


def _coerce_to_list(value: object) -> list[object] | None:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict) or isinstance(value, (str, bytes)):
        return None
    if hasattr(value, "__iter__"):
        try:
            return list(value)
        except TypeError:
            return None
    return None


def _extract_paddlex_det_boxes(value: object) -> list[dict[str, object]] | None:
    """
    Extract a list of detection box dicts from PaddleX object detection result objects.

    Some PP-Structure pipelines return a container object (e.g. `DetResult`) that stores its
    payload under a `.json` attribute (dict). We support this shape so local-paddle can
    produce normalized `elements`.
    """

    if value is None:
        return None

    # PaddleX result objects often store a JSON-ish dict as an attribute named `json`.
    json_attr = getattr(value, "json", None)
    if isinstance(json_attr, dict):
        payload = json_attr
    elif isinstance(value, dict):
        payload = value
    else:
        return None

    res = payload.get("res") if isinstance(payload.get("res"), dict) else payload
    if not isinstance(res, dict):
        return None
    boxes = res.get("boxes")
    if not isinstance(boxes, list) or not all(isinstance(x, dict) for x in boxes):
        return None

    out: list[dict[str, object]] = []
    for box in boxes:
        coordinate = box.get("coordinate")
        if coordinate is None:
            coordinate = box.get("bbox")
        if coordinate is None:
            coordinate = box.get("box")
        out.append(
            {
                "bbox": coordinate,
                "type": box.get("label") or box.get("type") or box.get("cls_name") or "text",
                "score": box.get("score"),
            }
        )
    return out


def _normalize_paddle_layout_output(raw: object) -> tuple[list[dict[str, object]], object]:
    """
    Normalize PaddleOCR pipeline output to a list of item dicts.

    We keep this permissive because PaddleOCR output shapes vary across versions.
    """

    if isinstance(raw, list):
        raw_list = raw
        raw_for_json: object = raw
    else:
        coerced = _coerce_to_list(raw)
        if coerced is not None:
            raw_list = coerced
            raw_for_json = coerced
        else:
            raw_list = []
            raw_for_json = raw

    # PP-StructureV3 often returns a singleton container dict with a `layout_det_res` object
    # (e.g. PaddleX `DetResult`). Extract the underlying `boxes` list when available.
    if raw_list and len(raw_list) == 1 and isinstance(raw_list[0], dict):
        container = cast(dict[str, object], raw_list[0])
        for key in ("layout_det_res", "region_det_res"):
            extracted = _extract_paddlex_det_boxes(container.get(key))
            if extracted is not None:
                return extracted, raw_for_json

    if raw_list and len(raw_list) == 1 and isinstance(raw_list[0], list):
        inner = raw_list[0]
        if isinstance(inner, list) and all(isinstance(x, dict) for x in inner):
            return cast(list[dict[str, object]], inner), raw_for_json

    if raw_list and all(isinstance(x, dict) for x in raw_list):
        return cast(list[dict[str, object]], raw_list), raw_for_json

    if isinstance(raw, dict):
        for key in ("layout_res", "layout", "result", "elements", "data", "outputs"):
            value = raw.get(key)
            if isinstance(value, list) and all(isinstance(x, dict) for x in value):
                return cast(list[dict[str, object]], value), raw

    raise RuntimeError(
        "Local layout analysis returned an invalid result "
        f"(expected list[dict], got: {type(raw).__name__})."
    )


def _filter_supported_constructor_kwargs(
    constructor: Any,
    kwargs: dict[str, object],
) -> dict[str, object]:
    """
    Best-effort filter of kwargs using constructor signature.

    Some PaddleOCR versions reject unknown kwargs only at runtime, causing repeated expensive
    retries. Filtering up-front reduces prewarm/init latency.
    """

    try:
        signature = inspect.signature(constructor)
    except (TypeError, ValueError):
        return dict(kwargs)

    parameters = list(signature.parameters.values())
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters):
        return dict(kwargs)

    allowed_names = {
        p.name
        for p in parameters
        if p.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    }
    return {k: v for k, v in kwargs.items() if k in allowed_names}


@lru_cache(maxsize=1)


def _get_paddleocr_engine() -> Any:
    PPStructure = _load_paddleocr_ppstructure()

    desired_kwargs: dict[str, object] = {
        "show_log": False,
        "use_gpu": False,
        "device": "cpu",
        "layout": True,
        "ocr": False,
        "table": False,
        "enable_hpi": False,
        "enable_mkldnn": False,
    }

    # PaddleOCR pipelines vary across versions and sometimes validate kwargs strictly.
    # We try a preferred set of arguments and retry by removing only the reported unknown ones.
    kwargs = _filter_supported_constructor_kwargs(PPStructure, desired_kwargs)
    for _ in range(len(kwargs) + 1):
        try:
            return PPStructure(**kwargs)
        except ValueError as exc:
            message = str(exc).strip()
            prefix = "Unknown argument:"
            if not message.startswith(prefix):
                raise
            unknown = message[len(prefix) :].strip()
            if not unknown or unknown not in kwargs:
                raise
            kwargs.pop(unknown, None)
        except RuntimeError as exc:
            if _is_paddlex_dependency_error(exc):
                hint = _paddlex_ocr_install_hint()
                raise RuntimeError(
                    "Local layout analysis requires PaddleX OCR extras for PP-StructureV3. "
                    f"Install them and try again: {hint}"
                ) from exc
            raise

    raise RuntimeError("Failed to initialize PaddleOCR PPStructure backend.")


def _paddle_bbox_to_xyxy(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)):
        return None

    # Common format: [x0, y0, x1, y1]
    if len(value) == 4 and all(isinstance(v, (int, float)) for v in value):
        x0, y0, x1, y1 = (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
        if x0 < x1 and y0 < y1:
            return x0, y0, x1, y1
        return None

    # Polygon format: [[x,y], [x,y], [x,y], [x,y]]
    if len(value) == 4 and all(isinstance(v, (list, tuple)) and len(v) == 2 for v in value):
        xs: list[float] = []
        ys: list[float] = []
        for pt in value:
            x, y = pt[0], pt[1]
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                return None
            xs.append(float(x))
            ys.append(float(y))
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        if x0 < x1 and y0 < y1:
            return x0, y0, x1, y1
        return None

    return None


_PADDLE_LABEL_MAP: dict[str, str] = {
    "title": "title",
    "header": "heading",
    "heading": "heading",
    "text": "text",
    "paragraph": "text",
    "list": "text",
    "footer": "text",
    "table": "table",
    "figure": "figure",
    "image": "figure",
    "equation": "text",
}


def _normalize_paddle_label(value: str) -> str:
    key = (value or "").strip().lower()
    if not key:
        return "text"
    return _PADDLE_LABEL_MAP.get(key, key)


def _analyze_layout_local_paddle(image_base64: str, *, settings: Settings) -> dict[str, object]:
    if not image_base64:
        raise ValueError("image_base64 is empty")

    payload = _strip_data_url_prefix(image_base64)
    if not payload:
        raise ValueError("image_base64 is empty")
    _validate_base64_payload(payload)

    configure_model_cache(settings)

    # Fail-fast on missing optional dependencies.
    _apply_paddle_safe_mode_env()
    engine = _get_paddleocr_engine()

    image_bytes = base64.b64decode(payload, validate=False)

    try:
        from PIL import Image
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Pillow is required for local layout analysis.") from exc

    try:
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("numpy is required for local layout analysis.") from exc

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            rgb = img.convert("RGB")
            arr = np.asarray(rgb)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Failed to decode PNG for local layout analysis.") from exc

    # Many CV stacks expect BGR.
    if getattr(arr, "ndim", 0) == 3 and getattr(arr, "shape", (0, 0, 0))[2] == 3:
        arr = arr[:, :, ::-1]

    raw_result = _invoke_paddle_engine_for_layout(engine, arr)
    result, raw_for_json = _normalize_paddle_layout_output(raw_result)

    elements: list[dict[str, object]] = []
    for item in result:
        bbox_obj = item.get("bbox") if "bbox" in item else item.get("box")
        bbox = _paddle_bbox_to_xyxy(bbox_obj)
        if bbox is None:
            continue

        label_obj = item.get("type") if "type" in item else item.get("label")
        label = _normalize_paddle_label(label_obj) if isinstance(label_obj, str) else "text"

        score_obj = item.get("score", None)
        score = float(score_obj) if isinstance(score_obj, (int, float)) else None
        if score is not None and not (0.0 <= score <= 1.0):
            score = None

        elements.append(
            {
                "page_index": 0,
                "bbox": bbox,
                "label": label,
                "score": score,
            }
        )

    def _sort_key(e: dict[str, object]) -> tuple[float, float, str]:
        bbox = cast(tuple[float, float, float, float], e["bbox"])
        label = cast(str, e["label"])
        return (bbox[1], bbox[0], label)

    elements.sort(key=_sort_key)

    raw = json.dumps(
        {
            "backend": _GlmOcrMode.local_paddle,
            "model": settings.layout_model,
            "result": raw_for_json,
        },
        ensure_ascii=False,
        default=str,
    )
    return {
        "schema_version": "pp-doclayout-v3",
        "elements": elements,
        "raw": raw,
    }


def prewarm_layout_backend(*, settings: Settings) -> None:
    """
    Eagerly initialize layout backend resources to reduce first-request cold start.

    - local-paddle / transformers: initialize and cache PaddleOCR PPStructure engine.
    """

    mode = (settings.layout_mode or "").strip().lower() or _GlmOcrMode.transformers
    if mode in {_GlmOcrMode.transformers, _GlmOcrMode.local_paddle}:
        configure_model_cache(settings)
        _apply_paddle_safe_mode_env()
        _ = _get_paddleocr_engine()
        return
    raise RuntimeError(
        "Layout analysis requires RAGPREP_LAYOUT_MODE=local-paddle."
    )


def analyze_layout_image_base64(image_base64: str, *, settings: Settings) -> dict[str, object]:
    """
    Request layout analysis for a single page image.

    Supported backends:
    - `local-paddle`: run PP-DocLayout-V3 locally via PaddleOCR (no Docker).

    Note:
    - `local-fast` mode is a text-layer-based path handled in `pdf_to_html` and does not
      use this image API.

    Return shape is stable: `schema_version`, `elements[{page_index,bbox,label,score}]`, `raw`.
    """

    mode = (settings.layout_mode or "").strip().lower() or _GlmOcrMode.local_fast
    if mode == _GlmOcrMode.transformers:
        mode = _GlmOcrMode.local_fast

    if mode == _GlmOcrMode.local_fast:
        raise RuntimeError(
            "local-fast layout mode does not use image layout API. "
            "Use pdf_to_html() fast layout path or set RAGPREP_LAYOUT_MODE=local-paddle."
        )

    if mode == _GlmOcrMode.local_paddle:
        return _analyze_layout_local_paddle(image_base64, settings=settings)
    raise RuntimeError("Layout analysis requires RAGPREP_LAYOUT_MODE=local-paddle.")
