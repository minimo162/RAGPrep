from __future__ import annotations

import base64
import binascii
import os
import tempfile
from dataclasses import dataclass
from threading import Lock
from typing import Any

from PIL import Image

from ragprep.config import Settings
from ragprep.model_cache import configure_model_cache

DEFAULT_TEXT_RECOGNITION_PROMPT = "Text Recognition:"


class _GlmOcrMode:
    transformers = "transformers"


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


def _decode_png_from_base64(image_base64: str) -> bytes:
    payload = _strip_data_url_prefix(image_base64)
    if not payload:
        raise ValueError("image_base64 is empty")
    _validate_base64_payload(payload)
    return base64.b64decode(payload, validate=False)


@dataclass(frozen=True)
class _TransformersClient:
    model_path: str
    processor: Any
    model: Any


_transformers_client_lock = Lock()
_transformers_client: _TransformersClient | None = None


def _load_transformers_client(model_path: str) -> _TransformersClient:
    global _transformers_client

    with _transformers_client_lock:
        if _transformers_client is not None and _transformers_client.model_path == model_path:
            return _transformers_client

        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Transformers backend selected, but required packages are missing. "
                "Install dependencies (example): "
                "pip install git+https://github.com/huggingface/transformers.git torch"
            ) from exc

        processor_cls: Any = AutoProcessor
        model_cls: Any = AutoModelForImageTextToText

        try:
            try:
                processor = processor_cls.from_pretrained(model_path, trust_remote_code=True)
            except TypeError:
                processor = processor_cls.from_pretrained(model_path)
        except Exception as exc:  # noqa: BLE001
            try:
                from transformers import __version__ as transformers_version
            except Exception:
                transformers_version = "unknown"
            raise RuntimeError(
                "Failed to load GLM-OCR processor via Transformers. "
                f"model={model_path!r}, transformers={transformers_version}. "
                "Try upgrading Transformers."
            ) from exc

        model_kwargs: dict[str, object] = {"pretrained_model_name_or_path": model_path}
        # Try to honor the upstream recommended args, but fall back if unsupported.
        for key, value in (("torch_dtype", "auto"), ("device_map", "auto")):
            model_kwargs[key] = value
        model_kwargs["trust_remote_code"] = True

        try:
            try:
                model = model_cls.from_pretrained(**model_kwargs)
            except TypeError:
                try:
                    model = model_cls.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                    )
                except TypeError:
                    model = model_cls.from_pretrained(model_path)
        except Exception as exc:  # noqa: BLE001
            try:
                from transformers import __version__ as transformers_version
            except Exception:
                transformers_version = "unknown"
            raise RuntimeError(
                "Failed to load GLM-OCR model via Transformers. "
                f"model={model_path!r}, transformers={transformers_version}. "
                "Try upgrading Transformers."
            ) from exc

        # Ensure model is on some device when device_map isn't supported.
        try:
            _ = model.device
        except Exception:
            try:
                model = model.to(torch.device("cpu"))
            except Exception:
                pass

        _transformers_client = _TransformersClient(
            model_path=model_path,
            processor=processor,
            model=model,
        )
        return _transformers_client


def _ocr_image_base64_via_transformers(image_base64: str, *, settings: Settings) -> str:
    if not image_base64:
        raise ValueError("image_base64 is empty")

    png_bytes = _decode_png_from_base64(image_base64)
    configure_model_cache(settings)

    # The official GLM-OCR transformers example uses a file path in messages.
    # Writing to a temp file avoids relying on in-memory image object support.
    tmp_path = ""
    fd: int | None = None
    try:
        fd, tmp_path = tempfile.mkstemp(prefix="ragprep-glm-ocr-", suffix=".png")
        with os.fdopen(fd, "wb") as f:
            f.write(png_bytes)
        fd = None

        with Image.open(tmp_path) as image:
            image.verify()

        client = _load_transformers_client(settings.glm_ocr_model)
        processor = client.processor
        model = client.model

        try:
            import torch
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Transformers backend selected, but torch is not installed. "
                "Install torch and retry."
            ) from exc

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": tmp_path},
                    {"type": "text", "text": DEFAULT_TEXT_RECOGNITION_PROMPT},
                ],
            }
        ]

        inputs: Any = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        try:
            inputs = inputs.to(model.device)
        except Exception:
            pass
        try:
            inputs.pop("token_type_ids", None)
        except Exception:
            pass

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=settings.glm_ocr_max_tokens,
            )

        prompt_len = inputs["input_ids"].shape[1]
        output_text = processor.decode(
            generated_ids[0][prompt_len:],
            skip_special_tokens=False,
        )
        return str(output_text)
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def ocr_image_base64(image_base64: str, *, settings: Settings) -> str:
    """
    Extract Markdown/text from a base64-encoded image using local GLM-OCR.

    Supported mode:
    - transformers (default): run in-process via Hugging Face Transformers.
    """

    mode = (settings.glm_ocr_mode or "").strip().lower()
    if not mode:
        mode = _GlmOcrMode.transformers

    if mode == _GlmOcrMode.transformers:
        return _ocr_image_base64_via_transformers(image_base64, settings=settings)

    raise RuntimeError(
        f"Unsupported GLM-OCR mode: {settings.glm_ocr_mode!r}. "
        "Supported mode: transformers."
    )
