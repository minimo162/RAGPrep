from __future__ import annotations

import base64
import binascii
import io
import json
import os
import time
from functools import lru_cache
from typing import Any, cast

import httpx

from ragprep.config import Settings

DEFAULT_LAYOUT_ANALYSIS_PROMPT = (
    "Document Layout Analysis (PP-DocLayout-V3): "
    "Return JSON only with keys: schema_version, elements[]. "
    "Each element must include: page_index, bbox[x0,y0,x1,y1], label, score."
)

ENV_LAYOUT_PADDLE_SAFE_MODE = "RAGPREP_LAYOUT_PADDLE_SAFE_MODE"


def _normalize_base_url(base_url: str) -> str:
    return base_url.strip().rstrip("/")


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


def _post_chat_completions(
    *,
    url: str,
    headers: dict[str, str],
    payload: dict[str, object],
    timeout_seconds: int,
) -> httpx.Response:
    timeout = httpx.Timeout(timeout_seconds)
    # Desktop environments often have proxy env vars configured; avoid routing loopback
    # layout requests through proxies.
    with httpx.Client(timeout=timeout, follow_redirects=False, trust_env=False) as client:
        return client.post(url, headers=headers, json=payload)


def _parse_chat_completions_response_content(response: httpx.Response) -> str:
    if response.status_code != 200:
        body = (response.text or "").strip()
        if len(body) > 800:
            body = body[:800] + "…"
        raise RuntimeError(f"GLM server returned {response.status_code}: {body}")

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError("GLM server returned invalid JSON.") from exc

    if not isinstance(data, dict):
        raise RuntimeError("GLM server returned invalid JSON shape (expected object).")

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("GLM response missing choices.")

    first = choices[0]
    if not isinstance(first, dict):
        raise RuntimeError("GLM response choices[0] is invalid.")

    message = first.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("GLM response missing message.")

    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError("GLM response message.content is missing or not a string.")

    return content


class _GlmOcrMode:
    transformers = "transformers"
    server = "server"
    local_paddle = "local-paddle"

_CODE_FENCE = "```"


def _sleep_backoff(*, base_seconds: float, attempt_index: int) -> None:
    if base_seconds <= 0:
        return
    if attempt_index < 0:
        attempt_index = 0
    time.sleep(base_seconds * (2**attempt_index))


def _as_float(value: object, *, name: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"{name} must be a number, got: {value!r}")


def _as_int(value: object, *, name: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise ValueError(f"{name} must be an int, got: {value!r}")


def _as_str(value: object, *, name: str) -> str:
    if isinstance(value, str):
        return value
    raise ValueError(f"{name} must be a string, got: {value!r}")


def _parse_bbox(value: object) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"bbox must be a list[4], got: {value!r}")
    x0 = _as_float(value[0], name="bbox[0]")
    y0 = _as_float(value[1], name="bbox[1]")
    x1 = _as_float(value[2], name="bbox[2]")
    y1 = _as_float(value[3], name="bbox[3]")
    if not (x0 < x1 and y0 < y1):
        raise ValueError(f"bbox must satisfy x0<x1 and y0<y1, got: {value!r}")
    return x0, y0, x1, y1


def _strip_code_fence(content: str) -> str:
    text = content.strip()
    if not text.startswith(_CODE_FENCE):
        return content

    # Format we expect: ```json\n{...}\n```
    first_nl = text.find("\n")
    if first_nl == -1:
        return content
    closing = text.find(_CODE_FENCE, first_nl + 1)
    if closing == -1:
        return content
    inner = text[first_nl + 1 : closing]
    return inner.strip()


def _extract_first_json_object(content: str) -> str:
    text = content.strip()
    if not text:
        raise RuntimeError("Layout analysis returned empty content.")

    start = text.find("{")
    if start == -1:
        raise RuntimeError("Layout analysis content does not contain a JSON object.")

    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
            if depth < 0:
                break

    raise RuntimeError("Layout analysis content contains an unterminated JSON object.")


def _sanitize_layout_content(content: str) -> str:
    # Prefer fenced JSON if present, but fall back to brace extraction.
    unfenced = _strip_code_fence(content)
    return _extract_first_json_object(unfenced)


def _parse_layout_result(content: str) -> tuple[str, tuple[dict[str, object], ...]]:
    sanitized = _sanitize_layout_content(content)
    try:
        payload = json.loads(sanitized)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Layout analysis did not return valid JSON.") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("Layout analysis JSON must be an object.")

    schema_version = payload.get("schema_version")
    elements = payload.get("elements")
    if not isinstance(schema_version, str) or not schema_version.strip():
        raise RuntimeError("Layout analysis JSON missing schema_version.")
    if not isinstance(elements, list):
        raise RuntimeError("Layout analysis JSON missing elements list.")

    cleaned: list[dict[str, object]] = []
    for elt in elements:
        if not isinstance(elt, dict):
            raise RuntimeError("Layout analysis elements must be objects.")
        cleaned.append(elt)
    return schema_version, tuple(cleaned)


def _load_paddleocr_ppstructure() -> Any:
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
                "available. Install PaddleOCR (CPU) and try again, or use "
                "RAGPREP_LAYOUT_MODE=server. Suggested install: uv pip install paddlepaddle "
                "paddleocr"
            ) from exc
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Local layout analysis requires optional dependencies. "
            "Install PaddleOCR (CPU) and try again, or use RAGPREP_LAYOUT_MODE=server. "
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
            "",
            "If it still fails, try upgrading/downgrading Paddle packages, or use server mode:",
            "  RAGPREP_LAYOUT_MODE=server",
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
    kwargs = dict(desired_kwargs)
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


def analyze_layout_image_base64(image_base64: str, *, settings: Settings) -> dict[str, object]:
    """
    Request layout analysis for a single page image.

    Supported backends:
    - `server`: call an OpenAI-compatible `/v1/chat/completions` endpoint (GLM-OCR server).
    - `local-paddle`: run PP-DocLayout-V3 locally via PaddleOCR (no Docker).

    Return shape is stable: `schema_version`, `elements[{page_index,bbox,label,score}]`, `raw`.
    """

    mode = (settings.layout_mode or "").strip().lower() or _GlmOcrMode.transformers
    if mode in {_GlmOcrMode.transformers, _GlmOcrMode.local_paddle}:
        return _analyze_layout_local_paddle(image_base64, settings=settings)
    if mode != _GlmOcrMode.server:
        raise RuntimeError(
            "Layout analysis requires RAGPREP_LAYOUT_MODE=server or "
            "RAGPREP_LAYOUT_MODE=local-paddle."
        )

    if not image_base64:
        raise ValueError("image_base64 is empty")

    payload = _strip_data_url_prefix(image_base64)
    if not payload:
        raise ValueError("image_base64 is empty")
    _validate_base64_payload(payload)

    base_url = _normalize_base_url(settings.layout_base_url)
    url = f"{base_url}/v1/chat/completions"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if settings.layout_api_key is not None:
        headers["Authorization"] = f"Bearer {settings.layout_api_key}"

    data_url = f"data:image/png;base64,{payload}"
    request_payload: dict[str, object] = {
        "model": settings.layout_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": DEFAULT_LAYOUT_ANALYSIS_PROMPT},
                ],
            }
        ],
        "max_tokens": settings.layout_max_tokens,
    }

    retry_count = max(0, int(settings.layout_retry_count))
    backoff_seconds = float(settings.layout_retry_backoff_seconds)

    for attempt_index in range(retry_count + 1):
        try:
            response = _post_chat_completions(
                url=url,
                headers=headers,
                payload=request_payload,
                timeout_seconds=settings.layout_timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            if attempt_index < retry_count:
                _sleep_backoff(base_seconds=backoff_seconds, attempt_index=attempt_index)
                continue
            raise RuntimeError(
                "Layout analysis request timed out. "
                f"base_url={base_url!r}. "
                "Ensure the layout server is running and reachable."
            ) from exc
        except httpx.RequestError as exc:
            if attempt_index < retry_count:
                _sleep_backoff(base_seconds=backoff_seconds, attempt_index=attempt_index)
                continue
            raise RuntimeError(
                "Failed to reach layout server. "
                f"base_url={base_url!r}. "
                f"error={exc}. "
                "Ensure the layout server is running and reachable."
            ) from exc

        # Retry on transient gateway/service errors from the server.
        if response.status_code in {502, 503, 504} and attempt_index < retry_count:
            _sleep_backoff(base_seconds=backoff_seconds, attempt_index=attempt_index)
            continue
        break
    else:  # pragma: no cover
        raise RuntimeError("unreachable")

    content = _parse_chat_completions_response_content(response)
    schema_version, raw_elements = _parse_layout_result(content)

    elements: list[dict[str, object]] = []
    for raw in raw_elements:
        page_index = _as_int(raw.get("page_index"), name="page_index")
        bbox = _parse_bbox(raw.get("bbox"))
        label = _as_str(raw.get("label"), name="label").strip()
        if not label:
            raise ValueError("label must be non-empty")
        score_obj = raw.get("score", None)
        score = _as_float(score_obj, name="score") if score_obj is not None else None
        if score is not None and not (0.0 <= score <= 1.0):
            raise ValueError(f"score must be in [0,1], got: {score!r}")

        elements.append(
            {
                "page_index": page_index,
                "bbox": bbox,
                "label": label,
                "score": score,
            }
        )

    return {
        "schema_version": schema_version,
        "elements": elements,
        "raw": content,
    }

