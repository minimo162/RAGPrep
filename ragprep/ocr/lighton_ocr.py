from __future__ import annotations

import base64
import binascii
import json

import httpx

from ragprep.config import Settings

DEFAULT_LIGHTON_PROMPT = (
    "OCR and layout analysis. Return JSON only with keys: "
    "schema_version, elements[], lines[]. "
    "elements[]: {page_index:int,bbox:[x0,y0,x1,y1],label:string,score?:number,order?:int}. "
    "lines[]: {bbox:[x0,y0,x1,y1],text:string,confidence?:number}."
)

_CODE_FENCE = "```"


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
    with httpx.Client(timeout=timeout, follow_redirects=False, trust_env=False) as client:
        return client.post(url, headers=headers, json=payload)


def _parse_chat_completions_response_content(response: httpx.Response) -> str:
    if response.status_code != 200:
        body = (response.text or "").strip()
        if len(body) > 800:
            body = body[:800] + "..."
        raise RuntimeError(f"LightOn OCR server returned {response.status_code}: {body}")

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError("LightOn OCR server returned invalid JSON.") from exc

    if not isinstance(data, dict):
        raise RuntimeError("LightOn OCR server returned invalid JSON shape (expected object).")

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("LightOn OCR response missing choices.")

    first = choices[0]
    if not isinstance(first, dict):
        raise RuntimeError("LightOn OCR response choices[0] is invalid.")

    message = first.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("LightOn OCR response missing message.")

    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Some OpenAI-compatible servers return content as content parts.
        text_parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str):
                text_parts.append(text)
        merged = "".join(text_parts).strip()
        if merged:
            return merged
    raise RuntimeError("LightOn OCR response message.content is missing or not text.")


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


def _parse_bbox(value: object, *, name: str = "bbox") -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{name} must be a list[4], got: {value!r}")
    x0 = _as_float(value[0], name=f"{name}[0]")
    y0 = _as_float(value[1], name=f"{name}[1]")
    x1 = _as_float(value[2], name=f"{name}[2]")
    y1 = _as_float(value[3], name=f"{name}[3]")
    if not (x0 < x1 and y0 < y1):
        raise ValueError(f"{name} must satisfy x0<x1 and y0<y1, got: {value!r}")
    return x0, y0, x1, y1


def _strip_code_fence(content: str) -> str:
    text = content.strip()
    if not text.startswith(_CODE_FENCE):
        return text

    first_nl = text.find("\n")
    if first_nl == -1:
        return text
    closing = text.find(_CODE_FENCE, first_nl + 1)
    if closing == -1:
        return text
    return text[first_nl + 1 : closing].strip()


def _extract_first_json_object(content: str) -> str:
    text = content.strip()
    if not text:
        raise RuntimeError("LightOn OCR returned empty content.")

    start = text.find("{")
    if start == -1:
        raise RuntimeError("LightOn OCR content does not contain a JSON object.")

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
    raise RuntimeError("LightOn OCR content contains an unterminated JSON object.")


def _sanitize_content(content: str) -> str:
    return _extract_first_json_object(_strip_code_fence(content))


def _parse_schema(content: str) -> tuple[str, list[dict[str, object]], list[dict[str, object]]]:
    sanitized = _sanitize_content(content)
    try:
        payload = json.loads(sanitized)
    except json.JSONDecodeError as exc:
        raise RuntimeError("LightOn OCR did not return valid JSON.") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("LightOn OCR JSON must be an object.")

    schema_version_obj = payload.get("schema_version")
    elements_obj = payload.get("elements")
    lines_obj = payload.get("lines")

    schema_version = _as_str(schema_version_obj, name="schema_version").strip()
    if not schema_version:
        raise RuntimeError("LightOn OCR JSON missing schema_version.")
    if not isinstance(elements_obj, list):
        raise RuntimeError("LightOn OCR JSON missing elements list.")
    if not isinstance(lines_obj, list):
        raise RuntimeError("LightOn OCR JSON missing lines list.")

    elements: list[dict[str, object]] = []
    for raw in elements_obj:
        if not isinstance(raw, dict):
            raise RuntimeError("LightOn OCR elements must be objects.")
        page_index = _as_int(raw.get("page_index"), name="page_index")
        bbox = _parse_bbox(raw.get("bbox"), name="bbox")
        label = _as_str(raw.get("label"), name="label").strip()
        if not label:
            raise ValueError("label must be non-empty")
        score_obj = raw.get("score")
        score: float | None = None
        if score_obj is not None:
            score = _as_float(score_obj, name="score")
            if not (0.0 <= score <= 1.0):
                raise ValueError(f"score must be in [0,1], got: {score!r}")
        order_obj = raw.get("order")
        order: int | None = None
        if order_obj is not None:
            order = _as_int(order_obj, name="order")
            if order < 0:
                raise ValueError("order must be >= 0")

        item: dict[str, object] = {
            "page_index": page_index,
            "bbox": bbox,
            "label": label,
            "score": score,
        }
        if order is not None:
            item["order"] = order
        elements.append(item)

    lines: list[dict[str, object]] = []
    for raw in lines_obj:
        if not isinstance(raw, dict):
            raise RuntimeError("LightOn OCR lines must be objects.")
        bbox = _parse_bbox(raw.get("bbox"), name="lines[].bbox")
        text = _as_str(raw.get("text"), name="lines[].text")
        confidence_obj = raw.get("confidence")
        confidence: float | None = None
        if confidence_obj is not None:
            confidence = _as_float(confidence_obj, name="confidence")
            if not (0.0 <= confidence <= 1.0):
                raise ValueError(f"confidence must be in [0,1], got: {confidence!r}")
        item = {"bbox": bbox, "text": text}
        if confidence is not None:
            item["confidence"] = confidence
        lines.append(item)

    return schema_version, elements, lines


def analyze_ocr_layout_image_base64(image_base64: str, *, settings: Settings) -> dict[str, object]:
    if not image_base64:
        raise ValueError("image_base64 is empty")

    payload = _strip_data_url_prefix(image_base64)
    if not payload:
        raise ValueError("image_base64 is empty")
    _validate_base64_payload(payload)

    base_url = _normalize_base_url(settings.lighton_base_url)
    url = f"{base_url}/v1/chat/completions"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if settings.lighton_api_key is not None:
        headers["Authorization"] = f"Bearer {settings.lighton_api_key}"

    data_url = f"data:image/png;base64,{payload}"
    request_payload: dict[str, object] = {
        "model": settings.lighton_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": DEFAULT_LIGHTON_PROMPT},
                ],
            }
        ],
        "max_tokens": settings.lighton_max_tokens,
    }

    try:
        response = _post_chat_completions(
            url=url,
            headers=headers,
            payload=request_payload,
            timeout_seconds=settings.lighton_timeout_seconds,
        )
    except httpx.TimeoutException as exc:
        raise RuntimeError(
            "LightOn OCR request timed out. "
            f"base_url={base_url!r}. "
            "Ensure the LightOn OCR server is running and reachable."
        ) from exc
    except httpx.RequestError as exc:
        raise RuntimeError(
            "Failed to reach LightOn OCR server. "
            f"base_url={base_url!r}. "
            f"error={exc}. "
            "Ensure the LightOn OCR server is running and reachable."
        ) from exc

    content = _parse_chat_completions_response_content(response)
    schema_version, elements, lines = _parse_schema(content)
    return {
        "schema_version": schema_version,
        "elements": elements,
        "lines": lines,
        "raw": content,
    }
