from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from PIL import Image

DEFAULT_MODEL_ID = "lightonai/LightOnOCR-2-1B"

ENV_MODEL_ID = "LIGHTONOCR_MODEL_ID"
ENV_DEVICE = "LIGHTONOCR_DEVICE"
ENV_DTYPE = "LIGHTONOCR_DTYPE"
ENV_MAX_NEW_TOKENS = "LIGHTONOCR_MAX_NEW_TOKENS"
ENV_DRY_RUN = "LIGHTONOCR_DRY_RUN"

DRY_RUN_OUTPUT = "LIGHTONOCR_DRY_RUN=1 (no inference)"


@dataclass(frozen=True)
class LightOnOCRSettings:
    model_id: str
    device: str
    dtype: str | None
    max_new_tokens: int


def get_settings() -> LightOnOCRSettings:
    model_id = os.getenv(ENV_MODEL_ID, DEFAULT_MODEL_ID).strip() or DEFAULT_MODEL_ID
    device = (os.getenv(ENV_DEVICE, "cpu").strip() or "cpu").lower()
    dtype = os.getenv(ENV_DTYPE)
    dtype = dtype.strip().lower() if dtype is not None and dtype.strip() else None

    max_new_tokens_str = os.getenv(ENV_MAX_NEW_TOKENS, "1024").strip() or "1024"
    try:
        max_new_tokens = int(max_new_tokens_str)
    except ValueError as exc:
        raise ValueError(
            f"{ENV_MAX_NEW_TOKENS} must be an int, got: {max_new_tokens_str!r}"
        ) from exc

    return LightOnOCRSettings(
        model_id=model_id,
        device=device,
        dtype=dtype,
        max_new_tokens=max_new_tokens,
    )


def _env_truthy(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def ocr_image(image: Image.Image) -> str:
    """
    Run LightOnOCR on a single image and return extracted Markdown/text.

    Environment variables:
    - LIGHTONOCR_MODEL_ID: HF model id (default: lightonai/LightOnOCR-2-1B)
    - LIGHTONOCR_DEVICE: cpu|cuda|mps|auto (default: cpu)
    - LIGHTONOCR_DTYPE: float32|bfloat16|float16 (default: float32 on cpu/mps, bfloat16 on cuda)
    - LIGHTONOCR_MAX_NEW_TOKENS: int (default: 1024)
    - LIGHTONOCR_DRY_RUN: truthy to skip inference and return placeholder text
    """

    if _env_truthy(ENV_DRY_RUN):
        return DRY_RUN_OUTPUT

    settings = get_settings()
    runtime = _get_runtime_cached(settings.model_id, settings.device, settings.dtype)
    return _run_inference(runtime, image=image, max_new_tokens=settings.max_new_tokens)


@dataclass(frozen=True)
class _Runtime:
    torch: Any
    model: Any
    processor: Any
    device: str
    dtype: Any


def _import_deps() -> tuple[Any, Any, Any]:
    try:
        import torch
        import transformers
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            "LightOnOCR dependencies are missing. Install torch and transformers from source. "
            "See https://huggingface.co/lightonai/LightOnOCR-2-1B for the recommended install."
        ) from exc

    model_cls = getattr(transformers, "LightOnOcrForConditionalGeneration", None)
    processor_cls = getattr(transformers, "LightOnOcrProcessor", None)
    if model_cls is None or processor_cls is None:
        raise ImportError(
            "Installed transformers does not expose LightOnOcr* classes. "
            "LightOnOCR-2 requires transformers installed from source."
        )

    return torch, model_cls, processor_cls


def _auto_device(torch: Any) -> str:
    if getattr(getattr(torch, "backends", None), "mps", None) is not None:
        try:
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:  # noqa: BLE001
            pass
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def _resolve_device_and_dtype(torch: Any, device: str, dtype: str | None) -> tuple[str, Any]:
    resolved_device = device or "cpu"
    if resolved_device == "auto":
        resolved_device = _auto_device(torch)

    dtype_name = dtype
    if dtype_name is None:
        dtype_name = "bfloat16" if resolved_device == "cuda" else "float32"

    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    try:
        torch_dtype = dtype_map[dtype_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported {ENV_DTYPE}: {dtype_name!r}") from exc

    return resolved_device, torch_dtype


@lru_cache(maxsize=4)
def _get_runtime_cached(model_id: str, device: str, dtype: str | None) -> _Runtime:
    try:
        return _load_runtime(model_id=model_id, device=device, dtype=dtype)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to load LightOnOCR model={model_id!r} device={device!r} dtype={dtype!r}"
        ) from exc


def _load_runtime(model_id: str, device: str, dtype: str | None) -> _Runtime:
    torch, ModelCls, ProcessorCls = _import_deps()

    resolved_device, torch_dtype = _resolve_device_and_dtype(torch, device=device, dtype=dtype)

    model = ModelCls.from_pretrained(model_id, torch_dtype=torch_dtype).to(resolved_device)
    processor = ProcessorCls.from_pretrained(model_id)

    return _Runtime(
        torch=torch,
        model=model,
        processor=processor,
        device=resolved_device,
        dtype=torch_dtype,
    )


def _run_inference(runtime: _Runtime, image: Image.Image, max_new_tokens: int) -> str:
    conversation = [{"role": "user", "content": [{"type": "image", "image": image}]}]

    inputs = runtime.processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {
        k: (
            v.to(device=runtime.device, dtype=runtime.dtype)
            if v.is_floating_point()
            else v.to(runtime.device)
        )
        for k, v in inputs.items()
    }

    output_ids = runtime.model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    return str(runtime.processor.decode(generated_ids, skip_special_tokens=True))
