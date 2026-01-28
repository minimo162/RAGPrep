from __future__ import annotations

import base64

import pytest

from ragprep.ocr.llama_server_payload import (
    build_ocr_chat_payload,
    extract_ocr_text,
    normalize_image_base64,
)


def _encode(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def test_normalize_image_base64_adds_data_url() -> None:
    payload = _encode(b"hello")
    result = normalize_image_base64(payload)
    assert result.startswith("data:image/png;base64,")
    assert result.endswith(payload)


def test_normalize_image_base64_preserves_mime() -> None:
    payload = _encode(b"hello")
    result = normalize_image_base64(f"data:image/jpeg;base64,{payload}")
    assert result.startswith("data:image/jpeg;base64,")
    assert result.endswith(payload)


def test_normalize_image_base64_rejects_empty() -> None:
    with pytest.raises(ValueError, match="image_base64 is empty"):
        normalize_image_base64("  ")


def test_normalize_image_base64_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="not valid base64"):
        normalize_image_base64("not-base64@@@")


def test_build_ocr_chat_payload_structure() -> None:
    payload = _encode(b"payload")
    result = build_ocr_chat_payload(
        prompt="Extract text.",
        image_base64=payload,
        model="llama-vision",
        max_tokens=256,
        temperature=0.1,
    )
    assert result["model"] == "llama-vision"
    assert result["max_tokens"] == 256
    assert result["temperature"] == 0.1

    messages = result["messages"]
    assert isinstance(messages, list)
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "Extract text."
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_build_ocr_chat_payload_omits_optional_fields() -> None:
    payload = _encode(b"payload")
    result = build_ocr_chat_payload(
        prompt="Extract text.",
        image_base64=payload,
        model="llama-vision",
    )
    assert "max_tokens" not in result
    assert "temperature" not in result


def test_extract_ocr_text_from_string_content() -> None:
    response = {"choices": [{"message": {"content": "HELLO"}}]}
    assert extract_ocr_text(response) == "HELLO"


def test_extract_ocr_text_from_list_content() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "text", "text": "A"},
                        {"type": "text", "text": "B"},
                    ]
                }
            }
        ]
    }
    assert extract_ocr_text(response) == "A\nB"


def test_extract_ocr_text_raises_on_error_response() -> None:
    response = {"error": {"message": "bad request", "type": "invalid_request_error"}}
    with pytest.raises(RuntimeError, match="bad request"):
        extract_ocr_text(response)


def test_extract_ocr_text_raises_on_missing_choices() -> None:
    with pytest.raises(RuntimeError, match="choices"):
        extract_ocr_text({})
