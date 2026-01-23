from __future__ import annotations

import importlib
import tempfile
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class LlamaCppSettings:
    model_path: str
    mmproj_path: str
    n_ctx: int | None
    n_threads: int | None
    n_gpu_layers: int | None


@dataclass(frozen=True)
class _Runtime:
    llama: Any
    lock: threading.Lock


def _import_llama_cpp() -> tuple[Any, Any]:
    try:
        llama_cpp = importlib.import_module("llama_cpp")
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            "llama-cpp-python is required for LIGHTONOCR_BACKEND=llama_cpp. "
            "Install it (e.g. `uv add llama-cpp-python` then `uv sync --dev`)."
        ) from exc

    Llama = getattr(llama_cpp, "Llama", None)
    if Llama is None:
        raise ImportError("llama_cpp.Llama not found. Check your llama-cpp-python install.")

    chat_format_mod = None
    for name in ("llama_cpp.llama_chat_format", "llama_cpp.llava"):
        try:
            chat_format_mod = importlib.import_module(name)
            break
        except Exception:  # noqa: BLE001
            continue

    if chat_format_mod is None:
        raise ImportError(
            "llama-cpp-python multimodal helpers not found. "
            "Expected llama_cpp.llama_chat_format or llama_cpp.llava."
        )

    chat_handler_cls = getattr(chat_format_mod, "Llava15ChatHandler", None) or getattr(
        chat_format_mod, "Llava16ChatHandler", None
    )
    if chat_handler_cls is None:
        raise ImportError(
            "Llava15ChatHandler/Llava16ChatHandler not found in llama-cpp-python. "
            "Cannot run multimodal GGUF."
        )

    return Llama, chat_handler_cls


def _validate_paths(settings: LlamaCppSettings) -> None:
    model_path = Path(settings.model_path)
    mmproj_path = Path(settings.mmproj_path)
    if not model_path.is_file():
        raise RuntimeError(f"GGUF model file not found: {model_path}")
    if not mmproj_path.is_file():
        raise RuntimeError(f"GGUF mmproj file not found: {mmproj_path}")


@lru_cache(maxsize=2)
def _get_runtime_cached(
    model_path: str,
    mmproj_path: str,
    n_ctx: int | None,
    n_threads: int | None,
    n_gpu_layers: int | None,
) -> _Runtime:
    Llama, ChatHandlerCls = _import_llama_cpp()

    init_kwargs: dict[str, Any] = {}
    init_kwargs["n_ctx"] = n_ctx if n_ctx is not None else 2048
    if n_threads is not None:
        init_kwargs["n_threads"] = n_threads
    if n_gpu_layers is not None:
        init_kwargs["n_gpu_layers"] = n_gpu_layers

    chat_handler = ChatHandlerCls(clip_model_path=mmproj_path)
    llama = Llama(model_path=model_path, chat_handler=chat_handler, **init_kwargs)
    return _Runtime(llama=llama, lock=threading.Lock())


def ocr_image(*, image: Image.Image, settings: LlamaCppSettings, max_new_tokens: int) -> str:
    _validate_paths(settings)
    runtime = _get_runtime_cached(
        settings.model_path,
        settings.mmproj_path,
        settings.n_ctx,
        settings.n_threads,
        settings.n_gpu_layers,
    )

    prompt = "Extract all text from this image and return it as Markdown."

    with tempfile.NamedTemporaryFile(
        prefix="ragprep_llamacpp_", suffix=".png", delete=False
    ) as tmp:
        image_path = Path(tmp.name)
    try:
        image.save(image_path, format="PNG")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_path.resolve().as_uri()}},
                ],
            }
        ]

        with runtime.lock:
            response = runtime.llama.create_chat_completion(
                messages=messages,
                max_tokens=max_new_tokens,
            )

        choices = response.get("choices") if isinstance(response, dict) else None
        if not choices:
            raise RuntimeError("llama.cpp returned no choices")

        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        raise RuntimeError(f"Unexpected llama.cpp response content type: {type(content)!r}")
    finally:
        try:
            image_path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
