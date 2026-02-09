from __future__ import annotations

import base64
import binascii

import httpx

from ragprep.config import Settings
from ragprep.ocr.lighton_server import ensure_server_base_url


def ocr_image_base64(image_base64: str, *, settings: Settings) -> str:
    payload = _strip_data_url_prefix(image_base64)
    if not payload:
        raise ValueError("image_base64 is empty")
    _validate_base64(payload)

    base_url = ensure_server_base_url(settings)
    request_url = f"{base_url}/v1/chat/completions"

    body = {
        "model": "lighton-ocr",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{payload}"},
                    }
                ],
            }
        ],
        "max_tokens": int(settings.lighton_max_tokens),
        "temperature": float(settings.lighton_temperature),
        "top_p": float(settings.lighton_top_p),
    }

    timeout = float(settings.lighton_request_timeout_seconds)
    try:
        with httpx.Client(timeout=timeout, trust_env=False) as client:
            response = client.post(request_url, json=body)
    except httpx.HTTPError as exc:
        raise RuntimeError(f"LightOn OCR request failed: {exc}") from exc

    if response.status_code != 200:
        text = response.text.strip()
        if len(text) > 300:
            text = text[:300].strip() + "..."
        raise RuntimeError(
            f"LightOn OCR request failed: HTTP {response.status_code}, body={text!r}"
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError("LightOn OCR response is not valid JSON.") from exc
    return _extract_response_text(data).strip()


def _extract_response_text(payload: object) -> str:
    if not isinstance(payload, dict):
        raise RuntimeError("LightOn OCR response has unexpected shape.")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("LightOn OCR response has no choices.")

    first = choices[0]
    if not isinstance(first, dict):
        raise RuntimeError("LightOn OCR response choice has unexpected shape.")
    message = first.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("LightOn OCR response message has unexpected shape.")
    content = message.get("content")

    if isinstance(content, str):
        return _normalize_newlines(content)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text_obj = item.get("text")
            if isinstance(text_obj, str) and text_obj:
                parts.append(text_obj)
        if parts:
            return _normalize_newlines("".join(parts))
    raise RuntimeError("LightOn OCR response content is empty.")


def _strip_data_url_prefix(value: str) -> str:
    raw = value.strip()
    if not raw:
        return ""
    if raw.startswith("data:"):
        comma = raw.find(",")
        if comma >= 0:
            return raw[comma + 1 :].strip()
    return raw


def _validate_base64(value: str) -> None:
    try:
        decoded = base64.b64decode(value, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image_base64 is not valid base64") from exc
    if not decoded:
        raise ValueError("image_base64 is empty")


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")
