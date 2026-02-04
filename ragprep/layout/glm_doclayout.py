from __future__ import annotations

import base64
import binascii
import json

import httpx

from ragprep.config import Settings

DEFAULT_LAYOUT_ANALYSIS_PROMPT = (
    "Document Layout Analysis (PP-DocLayout-V3): "
    "Return JSON only with keys: schema_version, elements[]. "
    "Each element must include: page_index, bbox[x0,y0,x1,y1], label, score."
)


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
    with httpx.Client(timeout=timeout) as client:
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

_CODE_FENCE = "```"


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


def analyze_layout_image_base64(image_base64: str, *, settings: Settings) -> dict[str, object]:
    """
    Request layout analysis for a single page image.

    Current implementation supports `server` mode only. If you need local PP-DocLayout-V3
    inference, implement it as a separate backend and keep this function's return shape stable.
    """

    mode = (settings.glm_ocr_mode or "").strip().lower() or _GlmOcrMode.transformers
    if mode != _GlmOcrMode.server:
        raise RuntimeError("Layout analysis currently requires RAGPREP_GLM_OCR_MODE=server.")

    if not image_base64:
        raise ValueError("image_base64 is empty")

    payload = _strip_data_url_prefix(image_base64)
    if not payload:
        raise ValueError("image_base64 is empty")
    _validate_base64_payload(payload)

    url = f"{_normalize_base_url(settings.glm_ocr_base_url)}/v1/chat/completions"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if settings.glm_ocr_api_key is not None:
        headers["Authorization"] = f"Bearer {settings.glm_ocr_api_key}"

    data_url = f"data:image/png;base64,{payload}"
    request_payload: dict[str, object] = {
        "model": settings.glm_ocr_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": DEFAULT_LAYOUT_ANALYSIS_PROMPT},
                ],
            }
        ],
        "max_tokens": settings.glm_ocr_max_tokens,
    }

    response = _post_chat_completions(
        url=url,
        headers=headers,
        payload=request_payload,
        timeout_seconds=settings.glm_ocr_timeout_seconds,
    )

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

