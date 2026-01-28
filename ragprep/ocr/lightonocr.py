from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

from PIL import Image

from ragprep import config as app_config

from .llama_server_client import LlamaServerClientSettings
from .llama_server_client import ocr_image_base64 as ocr_image_base64_llama_server
from .llamacpp_cli_runtime import LlamaCppCliSettings
from .llamacpp_cli_runtime import ocr_image as ocr_image_llama_cpp_cli
from .llamacpp_cli_runtime import ocr_image_base64 as ocr_image_base64_llama_cpp_cli

ENV_MAX_NEW_TOKENS = "LIGHTONOCR_MAX_NEW_TOKENS"
ENV_DRY_RUN = "LIGHTONOCR_DRY_RUN"
ENV_GGUF_MODEL_PATH = "LIGHTONOCR_GGUF_MODEL_PATH"
ENV_GGUF_MMPROJ_PATH = "LIGHTONOCR_GGUF_MMPROJ_PATH"
ENV_LLAVA_CLI_PATH = "LIGHTONOCR_LLAVA_CLI_PATH"
ENV_LLAMA_N_CTX = "LIGHTONOCR_LLAMA_N_CTX"
ENV_LLAMA_N_THREADS = "LIGHTONOCR_LLAMA_N_THREADS"
ENV_LLAMA_N_GPU_LAYERS = "LIGHTONOCR_LLAMA_N_GPU_LAYERS"
ENV_TEMPERATURE = "LIGHTONOCR_TEMPERATURE"
ENV_TOP_P = "LIGHTONOCR_TOP_P"
ENV_REPEAT_PENALTY = "LIGHTONOCR_REPEAT_PENALTY"
ENV_REPEAT_LAST_N = "LIGHTONOCR_REPEAT_LAST_N"
ENV_LIGHTONOCR_BACKEND = "LIGHTONOCR_BACKEND"
ENV_LIGHTONOCR_LLAMA_SERVER_URL = "LIGHTONOCR_LLAMA_SERVER_URL"
ENV_LIGHTONOCR_MODEL = "LIGHTONOCR_MODEL"
ENV_LIGHTONOCR_REQUEST_TIMEOUT_SECONDS = "LIGHTONOCR_REQUEST_TIMEOUT_SECONDS"

DRY_RUN_OUTPUT = "LIGHTONOCR_DRY_RUN=1 (no inference)"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MAX_NEW_TOKENS = 1000
_DEFAULT_TEMPERATURE = 0.2
_DEFAULT_TOP_P = 0.9
_DEFAULT_REPEAT_PENALTY = 1.15
_DEFAULT_REPEAT_LAST_N = 128
_DEFAULT_N_GPU_LAYERS = 99


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
    temperature: float
    top_p: float
    repeat_penalty: float
    repeat_last_n: int


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


def _parse_int_env(name: str, *, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    raw = value.strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an int, got: {raw!r}") from exc


def _parse_float_env(name: str, *, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    raw = value.strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got: {raw!r}") from exc


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
    max_new_tokens_str = os.getenv(ENV_MAX_NEW_TOKENS, str(_DEFAULT_MAX_NEW_TOKENS)).strip() or str(
        _DEFAULT_MAX_NEW_TOKENS
    )
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
    if llama_n_gpu_layers is None:
        llama_n_gpu_layers = _DEFAULT_N_GPU_LAYERS

    temperature = _parse_float_env(ENV_TEMPERATURE, default=_DEFAULT_TEMPERATURE)
    top_p = _parse_float_env(ENV_TOP_P, default=_DEFAULT_TOP_P)
    repeat_penalty = _parse_float_env(ENV_REPEAT_PENALTY, default=_DEFAULT_REPEAT_PENALTY)
    repeat_last_n = _parse_int_env(ENV_REPEAT_LAST_N, default=_DEFAULT_REPEAT_LAST_N)

    return LightOnOCRSettings(
        llava_cli_path=llava_cli_path,
        gguf_model_path=gguf_model_path,
        gguf_mmproj_path=gguf_mmproj_path,
        max_new_tokens=max_new_tokens,
        llama_n_ctx=llama_n_ctx,
        llama_n_threads=llama_n_threads,
        llama_n_gpu_layers=llama_n_gpu_layers,
        temperature=temperature,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        repeat_last_n=repeat_last_n,
    )


def _env_truthy(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _image_to_png_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _build_llama_server_settings() -> LlamaServerClientSettings:
    app_settings = app_config.get_settings()
    base_url = app_settings.lightonocr_llama_server_url
    model = app_settings.lightonocr_model
    if not base_url:
        raise ValueError(
            f"{ENV_LIGHTONOCR_LLAMA_SERVER_URL} is required when "
            f"{ENV_LIGHTONOCR_BACKEND}=llama-server"
        )
    if not model:
        raise ValueError(
            f"{ENV_LIGHTONOCR_MODEL} is required when {ENV_LIGHTONOCR_BACKEND}=llama-server"
        )

    max_new_tokens = _parse_int_env(ENV_MAX_NEW_TOKENS, default=_DEFAULT_MAX_NEW_TOKENS)
    temperature = _parse_float_env(ENV_TEMPERATURE, default=_DEFAULT_TEMPERATURE)

    return LlamaServerClientSettings(
        base_url=base_url,
        model=model,
        timeout_seconds=app_settings.lightonocr_request_timeout_seconds,
        max_tokens=max_new_tokens,
        temperature=temperature,
    )


def _should_use_llama_server() -> bool:
    backend = app_config.get_settings().lightonocr_backend
    return backend == "llama-server"


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
    - LIGHTONOCR_TEMPERATURE: sampling temperature (default: 0.2)
    - LIGHTONOCR_TOP_P: sampling top_p (default: 0.9)
    - LIGHTONOCR_REPEAT_PENALTY: repeat penalty (default: 1.15)
    - LIGHTONOCR_REPEAT_LAST_N: repeat penalty window (default: 128)
    - LIGHTONOCR_MAX_NEW_TOKENS: max tokens to generate (default: 1000)
    - LIGHTONOCR_DRY_RUN: if truthy, return a fixed string and do no inference
    - LIGHTONOCR_BACKEND: cli | llama-server
    - LIGHTONOCR_LLAMA_SERVER_URL: llama-server base URL
    - LIGHTONOCR_MODEL: llama-server model name
    - LIGHTONOCR_REQUEST_TIMEOUT_SECONDS: request timeout in seconds
    """

    if _env_truthy(ENV_DRY_RUN):
        return DRY_RUN_OUTPUT

    if _should_use_llama_server():
        server_settings = _build_llama_server_settings()
        image_base64 = _image_to_png_base64(image)
        return ocr_image_base64_llama_server(
            image_base64=image_base64,
            prompt="Extract all text from this image and return it as Markdown.",
            settings=server_settings,
        )

    cli_settings = get_settings()
    llama_settings = LlamaCppCliSettings(
        llava_cli_path=cli_settings.llava_cli_path,
        model_path=cli_settings.gguf_model_path,
        mmproj_path=cli_settings.gguf_mmproj_path,
        n_ctx=cli_settings.llama_n_ctx,
        n_threads=cli_settings.llama_n_threads,
        n_gpu_layers=cli_settings.llama_n_gpu_layers,
        temperature=cli_settings.temperature,
        top_p=cli_settings.top_p,
        repeat_penalty=cli_settings.repeat_penalty,
        repeat_last_n=cli_settings.repeat_last_n,
    )
    return ocr_image_llama_cpp_cli(
        image=image,
        settings=llama_settings,
        max_new_tokens=cli_settings.max_new_tokens,
    )


def ocr_image_base64(image_base64: str) -> str:
    """
    Run LightOnOCR on a base64-encoded PNG image and return extracted Markdown/text.

    Environment variables:
    - LIGHTONOCR_GGUF_MODEL_PATH: required path to a local .gguf model file
    - LIGHTONOCR_GGUF_MMPROJ_PATH: required path to a local mmproj .gguf file
    - LIGHTONOCR_LLAVA_CLI_PATH: optional path to `llava-cli` (if not set, use PATH)
    - LIGHTONOCR_LLAMA_N_CTX: optional int
    - LIGHTONOCR_LLAMA_N_THREADS: optional int
    - LIGHTONOCR_LLAMA_N_GPU_LAYERS: optional int
    - LIGHTONOCR_TEMPERATURE: sampling temperature (default: 0.2)
    - LIGHTONOCR_TOP_P: sampling top_p (default: 0.9)
    - LIGHTONOCR_REPEAT_PENALTY: repeat penalty (default: 1.15)
    - LIGHTONOCR_REPEAT_LAST_N: repeat penalty window (default: 128)
    - LIGHTONOCR_MAX_NEW_TOKENS: max tokens to generate (default: 1000)
    - LIGHTONOCR_DRY_RUN: if truthy, return a fixed string and do no inference
    - LIGHTONOCR_BACKEND: cli | llama-server
    - LIGHTONOCR_LLAMA_SERVER_URL: llama-server base URL
    - LIGHTONOCR_MODEL: llama-server model name
    - LIGHTONOCR_REQUEST_TIMEOUT_SECONDS: request timeout in seconds
    """

    if _env_truthy(ENV_DRY_RUN):
        return DRY_RUN_OUTPUT

    if _should_use_llama_server():
        server_settings = _build_llama_server_settings()
        return ocr_image_base64_llama_server(
            image_base64=image_base64,
            prompt="Extract all text from this image and return it as Markdown.",
            settings=server_settings,
        )

    cli_settings = get_settings()
    llama_settings = LlamaCppCliSettings(
        llava_cli_path=cli_settings.llava_cli_path,
        model_path=cli_settings.gguf_model_path,
        mmproj_path=cli_settings.gguf_mmproj_path,
        n_ctx=cli_settings.llama_n_ctx,
        n_threads=cli_settings.llama_n_threads,
        n_gpu_layers=cli_settings.llama_n_gpu_layers,
        temperature=cli_settings.temperature,
        top_p=cli_settings.top_p,
        repeat_penalty=cli_settings.repeat_penalty,
        repeat_last_n=cli_settings.repeat_last_n,
    )
    return ocr_image_base64_llama_cpp_cli(
        image_base64=image_base64,
        settings=llama_settings,
        max_new_tokens=cli_settings.max_new_tokens,
    )
