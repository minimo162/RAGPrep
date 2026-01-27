from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

from PIL import Image

from .llamacpp_cli_runtime import LlamaCppCliSettings
from .llamacpp_cli_runtime import ocr_image as ocr_image_llama_cpp_cli

ENV_MAX_NEW_TOKENS = "LIGHTONOCR_MAX_NEW_TOKENS"
ENV_DRY_RUN = "LIGHTONOCR_DRY_RUN"
ENV_GGUF_MODEL_PATH = "LIGHTONOCR_GGUF_MODEL_PATH"
ENV_GGUF_MMPROJ_PATH = "LIGHTONOCR_GGUF_MMPROJ_PATH"
ENV_LLAVA_CLI_PATH = "LIGHTONOCR_LLAVA_CLI_PATH"
ENV_LLAMA_N_CTX = "LIGHTONOCR_LLAMA_N_CTX"
ENV_LLAMA_N_THREADS = "LIGHTONOCR_LLAMA_N_THREADS"
ENV_LLAMA_N_GPU_LAYERS = "LIGHTONOCR_LLAMA_N_GPU_LAYERS"

DRY_RUN_OUTPUT = "LIGHTONOCR_DRY_RUN=1 (no inference)"
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _standalone_root(repo_root: Path) -> Path | None:
    if repo_root.name != "app":
        return None
    parent = repo_root.parent
    if parent.name != "standalone":
        return None
    return parent


def _expected_gguf_dir(repo_root: Path) -> Path:
    standalone_root = _standalone_root(repo_root)
    if standalone_root is not None:
        return standalone_root / "data" / "models" / "lightonocr-gguf"
    return repo_root / "dist" / "standalone" / "data" / "models" / "lightonocr-gguf"


@dataclass(frozen=True)
class LightOnOCRSettings:
    llava_cli_path: str | None
    gguf_model_path: str
    gguf_mmproj_path: str
    max_new_tokens: int
    llama_n_ctx: int | None
    llama_n_threads: int | None
    llama_n_gpu_layers: int | None


def _parse_optional_int_env(name: str) -> int | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an int, got: {value!r}") from exc


def _normalize_env_path(value: str) -> str:
    raw = value.strip()
    if not raw:
        return raw

    # Users sometimes include wrappers when copy/pasting examples.
    # - CMD: `set VAR="C:\\path\\file.gguf"` includes quotes in the value.
    # - Docs placeholders: `<C:\\path\\file.gguf>`
    wrappers = [('"', '"'), ("'", "'"), ("<", ">")]
    for start, end in wrappers:
        if raw.startswith(start) and raw.endswith(end) and len(raw) >= 2:
            raw = raw[1:-1].strip()

    # Allow file:// URIs in env vars (common when copying from logs).
    # Convert to a local path string for downstream Path(...).is_file checks.
    if raw.lower().startswith("file://"):
        parsed = urlparse(raw)
        if parsed.scheme.lower() == "file":
            raw = url2pathname(parsed.path).strip()

    return raw


def get_settings() -> LightOnOCRSettings:
    max_new_tokens_str = os.getenv(ENV_MAX_NEW_TOKENS, "1024").strip() or "1024"
    try:
        max_new_tokens = int(max_new_tokens_str)
    except ValueError as exc:
        raise ValueError(
            f"{ENV_MAX_NEW_TOKENS} must be an int, got: {max_new_tokens_str!r}"
        ) from exc

    gguf_model_path = os.getenv(ENV_GGUF_MODEL_PATH)
    gguf_model_path = _normalize_env_path(gguf_model_path) if gguf_model_path is not None else None
    if not gguf_model_path:
        gguf_model_path = None

    gguf_mmproj_path = os.getenv(ENV_GGUF_MMPROJ_PATH)
    gguf_mmproj_path = (
        _normalize_env_path(gguf_mmproj_path) if gguf_mmproj_path is not None else None
    )
    if not gguf_mmproj_path:
        gguf_mmproj_path = None

    llava_cli_path = os.getenv(ENV_LLAVA_CLI_PATH)
    llava_cli_path = _normalize_env_path(llava_cli_path) if llava_cli_path is not None else None
    if not llava_cli_path:
        llava_cli_path = None

    missing = []
    if gguf_model_path is None:
        missing.append(ENV_GGUF_MODEL_PATH)
    if gguf_mmproj_path is None:
        missing.append(ENV_GGUF_MMPROJ_PATH)
    if missing:
        missing_str = ", ".join(missing)
        expected_dir = _expected_gguf_dir(_REPO_ROOT)
        raise ValueError(
            f"Set {missing_str} to local GGUF paths "
            "(model: LightOnOCR-2-1B-Q6_K.gguf, mmproj: mmproj-BF16.gguf). "
            f"Expected under: {expected_dir}. "
            "Rebuild standalone: scripts/build-standalone.ps1 -Clean."
        )

    assert gguf_model_path is not None
    assert gguf_mmproj_path is not None

    llama_n_ctx = _parse_optional_int_env(ENV_LLAMA_N_CTX)
    llama_n_threads = _parse_optional_int_env(ENV_LLAMA_N_THREADS)
    llama_n_gpu_layers = _parse_optional_int_env(ENV_LLAMA_N_GPU_LAYERS)

    return LightOnOCRSettings(
        llava_cli_path=llava_cli_path,
        gguf_model_path=gguf_model_path,
        gguf_mmproj_path=gguf_mmproj_path,
        max_new_tokens=max_new_tokens,
        llama_n_ctx=llama_n_ctx,
        llama_n_threads=llama_n_threads,
        llama_n_gpu_layers=llama_n_gpu_layers,
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
    - LIGHTONOCR_GGUF_MODEL_PATH: required path to a local .gguf model file
    - LIGHTONOCR_GGUF_MMPROJ_PATH: required path to a local mmproj .gguf file
    - LIGHTONOCR_LLAVA_CLI_PATH: optional path to `llava-cli` (if not set, use PATH)
    - LIGHTONOCR_LLAMA_N_CTX: optional int
    - LIGHTONOCR_LLAMA_N_THREADS: optional int
    - LIGHTONOCR_LLAMA_N_GPU_LAYERS: optional int
    - LIGHTONOCR_MAX_NEW_TOKENS: max tokens to generate (default: 1024)
    - LIGHTONOCR_DRY_RUN: if truthy, return a fixed string and do no inference
    """

    if _env_truthy(ENV_DRY_RUN):
        return DRY_RUN_OUTPUT

    settings = get_settings()
    llama_settings = LlamaCppCliSettings(
        llava_cli_path=settings.llava_cli_path,
        model_path=settings.gguf_model_path,
        mmproj_path=settings.gguf_mmproj_path,
        n_ctx=settings.llama_n_ctx,
        n_threads=settings.llama_n_threads,
        n_gpu_layers=settings.llama_n_gpu_layers,
    )
    return ocr_image_llama_cpp_cli(
        image=image,
        settings=llama_settings,
        max_new_tokens=settings.max_new_tokens,
    )
