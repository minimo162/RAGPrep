from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping
from typing import Any

_DEFAULT_IMAGE_MIME = "image/png"


def _require_non_empty(name: str, value: str) -> str:
    if value is None:
        raise ValueError(f"{name} is required")
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(f"{name} is required")
    return trimmed


def _normalize_data_url_payload(payload: str) -> str:
    normalized = "".join(payload.split())
    if not normalized:
        raise ValueError("image_base64 is empty")
    try:
        decoded = base64.b64decode(normalized, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image_base64 is not valid base64") from exc
    if not decoded:
        raise ValueError("image_base64 is empty")
    return normalized


def normalize_image_base64(image_base64: str, *, mime_type: str = _DEFAULT_IMAGE_MIME) -> str:
    if image_base64 is None or not image_base64.strip():
        raise ValueError("image_base64 is empty")
    raw = image_base64.strip()
    mime = _require_non_empty("mime_type", mime_type)
    payload = raw

    if raw.lower().startswith("data:"):
        comma_index = raw.find(",")
        if comma_index == -1:
            raise ValueError("image_base64 data URL is missing a comma")
        header = raw[5:comma_index]
        payload = raw[comma_index + 1 :]
        if header:
            header_parts = header.split(";")
            if header_parts[0].strip():
                mime = header_parts[0].strip()

    normalized_payload = _normalize_data_url_payload(payload)
    return f"data:{mime};base64,{normalized_payload}"


def build_ocr_chat_payload(
    *,
    prompt: str,
    image_base64: str,
    model: str,
    max_tokens: int | None = None,
    temperature: float | None = None,
    image_mime_type: str = _DEFAULT_IMAGE_MIME,
) -> dict[str, Any]:
    normalized_prompt = _require_non_empty("prompt", prompt)
    normalized_model = _require_non_empty("model", model)

    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")

    image_url = normalize_image_base64(image_base64, mime_type=image_mime_type)
    payload: dict[str, Any] = {
        "model": normalized_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": normalized_prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
    }

    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature

    return payload


def _format_error_detail(error: object) -> str:
    if isinstance(error, Mapping):
        message = error.get("message")
        error_type = error.get("type")
        code = error.get("code")
        parts = [str(item) for item in (message, error_type, code) if item]
        if parts:
            return " / ".join(parts)
        return str(error)
    return str(error)


def _content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if not isinstance(item, Mapping):
                continue
            text_value = item.get("text")
            if text_value is None:
                continue
            texts.append(str(text_value))
        if texts:
            return "\n".join(texts)
    raise RuntimeError("response message content is not text")


def extract_ocr_text(response: Mapping[str, Any]) -> str:
    if response is None:
        raise RuntimeError("response is empty")

    error = response.get("error")
    if error:
        detail = _format_error_detail(error)
        raise RuntimeError(f"llama-server error: {detail}")

    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("response does not contain choices")

    first_choice = choices[0]
    if not isinstance(first_choice, Mapping):
        raise RuntimeError("response choice is invalid")

    message = first_choice.get("message")
    if not isinstance(message, Mapping):
        raise RuntimeError("response choice has no message")

    content = message.get("content")
    if content is None:
        raise RuntimeError("response message has no content")

    text = _content_to_text(content)
    if not text:
        raise RuntimeError("response message content is empty")

    return text
