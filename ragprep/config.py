from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final

ENV_MAX_UPLOAD_BYTES: Final[str] = "RAGPREP_MAX_UPLOAD_BYTES"
ENV_MAX_PAGES: Final[str] = "RAGPREP_MAX_PAGES"
ENV_RENDER_DPI: Final[str] = "RAGPREP_RENDER_DPI"
ENV_RENDER_MAX_EDGE: Final[str] = "RAGPREP_RENDER_MAX_EDGE"
ENV_MAX_CONCURRENCY: Final[str] = "RAGPREP_MAX_CONCURRENCY"
ENV_PDF_BACKEND: Final[str] = "RAGPREP_PDF_BACKEND"
ENV_MODEL_CACHE_DIR: Final[str] = "RAGPREP_MODEL_CACHE_DIR"

ENV_LIGHTON_REPO_ID: Final[str] = "RAGPREP_LIGHTON_REPO_ID"
ENV_LIGHTON_MODEL_FILE: Final[str] = "RAGPREP_LIGHTON_MODEL_FILE"
ENV_LIGHTON_MMPROJ_FILE: Final[str] = "RAGPREP_LIGHTON_MMPROJ_FILE"
ENV_LIGHTON_MODEL_DIR: Final[str] = "RAGPREP_LIGHTON_MODEL_DIR"
ENV_LIGHTON_AUTO_DOWNLOAD: Final[str] = "RAGPREP_LIGHTON_AUTO_DOWNLOAD"
ENV_LIGHTON_PROFILE: Final[str] = "RAGPREP_LIGHTON_PROFILE"
ENV_LLAMA_SERVER_PATH: Final[str] = "RAGPREP_LLAMA_SERVER_PATH"
ENV_LIGHTON_SERVER_HOST: Final[str] = "RAGPREP_LIGHTON_SERVER_HOST"
ENV_LIGHTON_SERVER_PORT: Final[str] = "RAGPREP_LIGHTON_SERVER_PORT"
ENV_LIGHTON_START_TIMEOUT_SECONDS: Final[str] = "RAGPREP_LIGHTON_START_TIMEOUT_SECONDS"
ENV_LIGHTON_REQUEST_TIMEOUT_SECONDS: Final[str] = "RAGPREP_LIGHTON_REQUEST_TIMEOUT_SECONDS"
ENV_LIGHTON_CTX_SIZE: Final[str] = "RAGPREP_LIGHTON_CTX_SIZE"
ENV_LIGHTON_N_GPU_LAYERS: Final[str] = "RAGPREP_LIGHTON_N_GPU_LAYERS"
ENV_LIGHTON_PARALLEL: Final[str] = "RAGPREP_LIGHTON_PARALLEL"
ENV_LIGHTON_THREADS: Final[str] = "RAGPREP_LIGHTON_THREADS"
ENV_LIGHTON_THREADS_BATCH: Final[str] = "RAGPREP_LIGHTON_THREADS_BATCH"
ENV_LIGHTON_FLASH_ATTN: Final[str] = "RAGPREP_LIGHTON_FLASH_ATTN"
ENV_LIGHTON_MAX_TOKENS: Final[str] = "RAGPREP_LIGHTON_MAX_TOKENS"
ENV_LIGHTON_TEMPERATURE: Final[str] = "RAGPREP_LIGHTON_TEMPERATURE"
ENV_LIGHTON_TOP_P: Final[str] = "RAGPREP_LIGHTON_TOP_P"
ENV_LIGHTON_PAGE_CONCURRENCY: Final[str] = "RAGPREP_LIGHTON_PAGE_CONCURRENCY"
ENV_LIGHTON_RENDER_DPI: Final[str] = "RAGPREP_LIGHTON_RENDER_DPI"
ENV_LIGHTON_RENDER_MAX_EDGE: Final[str] = "RAGPREP_LIGHTON_RENDER_MAX_EDGE"
ENV_LIGHTON_MERGE_POLICY: Final[str] = "RAGPREP_LIGHTON_MERGE_POLICY"
ENV_LIGHTON_FAST_PASS: Final[str] = "RAGPREP_LIGHTON_FAST_PASS"
ENV_LIGHTON_FAST_RENDER_DPI: Final[str] = "RAGPREP_LIGHTON_FAST_RENDER_DPI"
ENV_LIGHTON_FAST_RENDER_MAX_EDGE: Final[str] = "RAGPREP_LIGHTON_FAST_RENDER_MAX_EDGE"
ENV_LIGHTON_FAST_RETRY: Final[str] = "RAGPREP_LIGHTON_FAST_RETRY"
ENV_LIGHTON_SECONDARY_TABLE_REPAIR: Final[str] = "RAGPREP_LIGHTON_SECONDARY_TABLE_REPAIR"
ENV_LIGHTON_RETRY_RENDER_DPI: Final[str] = "RAGPREP_LIGHTON_RETRY_RENDER_DPI"
ENV_LIGHTON_RETRY_RENDER_MAX_EDGE: Final[str] = "RAGPREP_LIGHTON_RETRY_RENDER_MAX_EDGE"
ENV_LIGHTON_RETRY_MIN_QUALITY: Final[str] = "RAGPREP_LIGHTON_RETRY_MIN_QUALITY"
ENV_LIGHTON_RETRY_QUALITY_GAP: Final[str] = "RAGPREP_LIGHTON_RETRY_QUALITY_GAP"
ENV_LIGHTON_RETRY_MIN_PYM_TEXT_LEN: Final[str] = "RAGPREP_LIGHTON_RETRY_MIN_PYM_TEXT_LEN"
ENV_LIGHTON_FAST_NON_TABLE_MAX_EDGE: Final[str] = "RAGPREP_LIGHTON_FAST_NON_TABLE_MAX_EDGE"
ENV_LIGHTON_FAST_TABLE_LIKELIHOOD_THRESHOLD: Final[str] = (
    "RAGPREP_LIGHTON_FAST_TABLE_LIKELIHOOD_THRESHOLD"
)
ENV_LIGHTON_FAST_MAX_TOKENS_TEXT: Final[str] = "RAGPREP_LIGHTON_FAST_MAX_TOKENS_TEXT"
ENV_LIGHTON_FAST_MAX_TOKENS_TABLE: Final[str] = "RAGPREP_LIGHTON_FAST_MAX_TOKENS_TABLE"
ENV_LIGHTON_FAST_POSTPROCESS_MODE: Final[str] = "RAGPREP_LIGHTON_FAST_POSTPROCESS_MODE"
ENV_LIGHTON_PYMUPDF_PAGE_FALLBACK_MODE: Final[str] = (
    "RAGPREP_LIGHTON_PYMUPDF_PAGE_FALLBACK_MODE"
)

DEFAULT_MAX_UPLOAD_BYTES: Final[int] = 100 * 1024 * 1024
DEFAULT_MAX_PAGES: Final[int] = 200
DEFAULT_RENDER_DPI: Final[int] = 400
DEFAULT_RENDER_MAX_EDGE: Final[int] = 1540
DEFAULT_MAX_CONCURRENCY: Final[int] = 1
DEFAULT_MODEL_CACHE_DIR: Final[str] = os.path.join("~", ".ragprep", "model-cache")
DEFAULT_PDF_BACKEND: Final[str] = "lighton-ocr"
SUPPORTED_PDF_BACKENDS: Final[tuple[str, ...]] = ("lighton-ocr",)

DEFAULT_LIGHTON_REPO_ID: Final[str] = "noctrex/LightOnOCR-2-1B-GGUF"
DEFAULT_LIGHTON_MODEL_FILE: Final[str] = "LightOnOCR-2-1B-IQ4_XS.gguf"
DEFAULT_LIGHTON_MMPROJ_FILE: Final[str] = "mmproj-BF16.gguf"
DEFAULT_LIGHTON_MODEL_DIR: Final[str] = os.path.join("~", ".ragprep", "models", "lighton")
DEFAULT_LIGHTON_AUTO_DOWNLOAD: Final[bool] = True
DEFAULT_LIGHTON_PROFILE: Final[str] = "balanced"
DEFAULT_LIGHTON_SERVER_HOST: Final[str] = "127.0.0.1"
DEFAULT_LIGHTON_SERVER_PORT: Final[int] = 8080
DEFAULT_LIGHTON_START_TIMEOUT_SECONDS: Final[int] = 300
DEFAULT_LIGHTON_REQUEST_TIMEOUT_SECONDS: Final[int] = 600
DEFAULT_LIGHTON_CTX_SIZE: Final[int] = 8192
DEFAULT_LIGHTON_N_GPU_LAYERS: Final[int] = -1
DEFAULT_LIGHTON_PARALLEL: Final[int] = 2
DEFAULT_LIGHTON_THREADS: Final[int] = max(1, int(os.cpu_count() or 4))
DEFAULT_LIGHTON_THREADS_BATCH: Final[int] = max(1, int(os.cpu_count() or 4))
DEFAULT_LIGHTON_FLASH_ATTN: Final[bool] = True
DEFAULT_LIGHTON_MAX_TOKENS: Final[int] = 8192
DEFAULT_LIGHTON_TEMPERATURE: Final[float] = 0.0
DEFAULT_LIGHTON_TOP_P: Final[float] = 1.0
DEFAULT_LIGHTON_PAGE_CONCURRENCY: Final[int] = 2
DEFAULT_LIGHTON_RENDER_DPI: Final[int] = 220
DEFAULT_LIGHTON_RENDER_MAX_EDGE: Final[int] = 1280
DEFAULT_LIGHTON_MERGE_POLICY: Final[str] = "strict"
DEFAULT_LIGHTON_FAST_PASS: Final[bool] = True
DEFAULT_LIGHTON_FAST_RENDER_DPI: Final[int] = 200
DEFAULT_LIGHTON_FAST_RENDER_MAX_EDGE: Final[int] = 1100
DEFAULT_LIGHTON_FAST_RETRY: Final[bool] = False
DEFAULT_LIGHTON_SECONDARY_TABLE_REPAIR: Final[bool] = False
DEFAULT_LIGHTON_RETRY_RENDER_DPI: Final[int] = 220
DEFAULT_LIGHTON_RETRY_RENDER_MAX_EDGE: Final[int] = 1280
DEFAULT_LIGHTON_RETRY_MIN_QUALITY: Final[float] = 0.40
DEFAULT_LIGHTON_RETRY_QUALITY_GAP: Final[float] = 0.22
DEFAULT_LIGHTON_RETRY_MIN_PYM_TEXT_LEN: Final[int] = 80
DEFAULT_LIGHTON_FAST_NON_TABLE_MAX_EDGE: Final[int] = 960
DEFAULT_LIGHTON_FAST_TABLE_LIKELIHOOD_THRESHOLD: Final[float] = 0.60
DEFAULT_LIGHTON_FAST_MAX_TOKENS_TEXT: Final[int] = 4096
DEFAULT_LIGHTON_FAST_MAX_TOKENS_TABLE: Final[int] = 8192
DEFAULT_LIGHTON_FAST_POSTPROCESS_MODE: Final[str] = "light"
DEFAULT_LIGHTON_PYMUPDF_PAGE_FALLBACK_MODE: Final[str] = "repeat"
SUPPORTED_LIGHTON_PROFILES: Final[tuple[str, ...]] = ("balanced",)
SUPPORTED_LIGHTON_MERGE_POLICIES: Final[tuple[str, ...]] = ("strict", "aggressive")
SUPPORTED_LIGHTON_FAST_POSTPROCESS_MODES: Final[tuple[str, ...]] = ("full", "light", "off")
SUPPORTED_LIGHTON_PYMUPDF_PAGE_FALLBACK_MODES: Final[tuple[str, ...]] = (
    "off",
    "repeat",
    "aggressive",
)


@dataclass(frozen=True)
class Settings:
    max_upload_bytes: int
    max_pages: int
    render_dpi: int
    render_max_edge: int
    max_concurrency: int
    model_cache_dir: str
    ocr_backend: str
    pdf_backend: str
    lighton_profile: str

    lighton_repo_id: str
    lighton_model_file: str
    lighton_mmproj_file: str
    lighton_model_dir: str
    lighton_auto_download: bool
    llama_server_path: str | None
    lighton_server_host: str
    lighton_server_port: int
    lighton_start_timeout_seconds: int
    lighton_request_timeout_seconds: int
    lighton_ctx_size: int
    lighton_n_gpu_layers: int
    lighton_parallel: int
    lighton_threads: int
    lighton_threads_batch: int
    lighton_flash_attn: bool
    lighton_max_tokens: int
    lighton_temperature: float
    lighton_top_p: float
    lighton_page_concurrency: int
    lighton_render_dpi: int
    lighton_render_max_edge: int
    lighton_merge_policy: str
    lighton_fast_pass: bool
    lighton_fast_render_dpi: int
    lighton_fast_render_max_edge: int
    lighton_fast_retry: bool
    lighton_secondary_table_repair: bool
    lighton_retry_render_dpi: int
    lighton_retry_render_max_edge: int
    lighton_retry_min_quality: float
    lighton_retry_quality_gap: float
    lighton_retry_min_pym_text_len: int
    lighton_fast_non_table_max_edge: int
    lighton_fast_table_likelihood_threshold: float
    lighton_fast_max_tokens_text: int
    lighton_fast_max_tokens_table: int
    lighton_fast_postprocess_mode: str
    lighton_pymupdf_page_fallback_mode: str


def _get_positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be an int, got: {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got: {value}")
    return value


def _get_int_with_min(name: str, default: int, *, min_value: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be an int, got: {raw!r}") from exc
    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got: {value}")
    return value


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a bool (0/1/true/false), got: {raw!r}")


def _get_float_with_min(name: str, default: float, *, min_value: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got: {raw!r}") from exc
    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got: {value}")
    return value


def _get_optional_str(name: str) -> str | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    value = raw.strip()
    return value or None


def _get_trimmed_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip()


def _get_pdf_backend() -> str:
    raw = os.getenv(ENV_PDF_BACKEND)
    if raw is None or not raw.strip():
        return DEFAULT_PDF_BACKEND
    value = raw.strip().lower()
    if value not in SUPPORTED_PDF_BACKENDS:
        raise ValueError(
            f"{ENV_PDF_BACKEND} must be one of {', '.join(SUPPORTED_PDF_BACKENDS)}, got: {raw!r}"
        )
    return value


def _get_model_cache_dir() -> str:
    raw = os.getenv(ENV_MODEL_CACHE_DIR)
    if raw is None or not raw.strip():
        return DEFAULT_MODEL_CACHE_DIR
    return raw.strip()


def _get_lighton_profile() -> str:
    raw = os.getenv(ENV_LIGHTON_PROFILE)
    if raw is None or not raw.strip():
        return DEFAULT_LIGHTON_PROFILE
    value = raw.strip().lower()
    if value not in SUPPORTED_LIGHTON_PROFILES:
        raise ValueError(
            f"{ENV_LIGHTON_PROFILE} must be one of "
            f"{', '.join(SUPPORTED_LIGHTON_PROFILES)}, got: {raw!r}"
        )
    return value


def _get_lighton_merge_policy() -> str:
    raw = os.getenv(ENV_LIGHTON_MERGE_POLICY)
    if raw is None or not raw.strip():
        return DEFAULT_LIGHTON_MERGE_POLICY
    value = raw.strip().lower()
    if value not in SUPPORTED_LIGHTON_MERGE_POLICIES:
        raise ValueError(
            f"{ENV_LIGHTON_MERGE_POLICY} must be one of "
            f"{', '.join(SUPPORTED_LIGHTON_MERGE_POLICIES)}, got: {raw!r}"
        )
    return value


def _get_lighton_fast_postprocess_mode(default: str = DEFAULT_LIGHTON_FAST_POSTPROCESS_MODE) -> str:
    raw = os.getenv(ENV_LIGHTON_FAST_POSTPROCESS_MODE)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value not in SUPPORTED_LIGHTON_FAST_POSTPROCESS_MODES:
        raise ValueError(
            f"{ENV_LIGHTON_FAST_POSTPROCESS_MODE} must be one of "
            f"{', '.join(SUPPORTED_LIGHTON_FAST_POSTPROCESS_MODES)}, got: {raw!r}"
        )
    return value


def _get_lighton_pymupdf_page_fallback_mode(
    default: str = DEFAULT_LIGHTON_PYMUPDF_PAGE_FALLBACK_MODE,
) -> str:
    raw = os.getenv(ENV_LIGHTON_PYMUPDF_PAGE_FALLBACK_MODE)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value not in SUPPORTED_LIGHTON_PYMUPDF_PAGE_FALLBACK_MODES:
        raise ValueError(
            f"{ENV_LIGHTON_PYMUPDF_PAGE_FALLBACK_MODE} must be one of "
            f"{', '.join(SUPPORTED_LIGHTON_PYMUPDF_PAGE_FALLBACK_MODES)}, got: {raw!r}"
        )
    return value


def get_settings() -> Settings:
    ocr_backend = _get_pdf_backend()
    profile = _get_lighton_profile()

    default_lighton_parallel = DEFAULT_LIGHTON_PARALLEL
    default_lighton_page_concurrency = DEFAULT_LIGHTON_PAGE_CONCURRENCY
    default_lighton_max_tokens = DEFAULT_LIGHTON_MAX_TOKENS
    default_lighton_fast_pass = DEFAULT_LIGHTON_FAST_PASS
    default_lighton_fast_render_dpi = DEFAULT_LIGHTON_FAST_RENDER_DPI
    default_lighton_fast_render_max_edge = DEFAULT_LIGHTON_FAST_RENDER_MAX_EDGE
    default_lighton_fast_retry = DEFAULT_LIGHTON_FAST_RETRY
    default_lighton_secondary_table_repair = DEFAULT_LIGHTON_SECONDARY_TABLE_REPAIR
    default_lighton_fast_non_table_max_edge = DEFAULT_LIGHTON_FAST_NON_TABLE_MAX_EDGE
    default_lighton_fast_max_tokens_text = DEFAULT_LIGHTON_FAST_MAX_TOKENS_TEXT
    default_lighton_fast_max_tokens_table = DEFAULT_LIGHTON_FAST_MAX_TOKENS_TABLE
    default_lighton_fast_postprocess_mode = DEFAULT_LIGHTON_FAST_POSTPROCESS_MODE

    return Settings(
        max_upload_bytes=_get_positive_int(ENV_MAX_UPLOAD_BYTES, DEFAULT_MAX_UPLOAD_BYTES),
        max_pages=_get_positive_int(ENV_MAX_PAGES, DEFAULT_MAX_PAGES),
        render_dpi=_get_positive_int(ENV_RENDER_DPI, DEFAULT_RENDER_DPI),
        render_max_edge=_get_positive_int(ENV_RENDER_MAX_EDGE, DEFAULT_RENDER_MAX_EDGE),
        max_concurrency=_get_positive_int(ENV_MAX_CONCURRENCY, DEFAULT_MAX_CONCURRENCY),
        model_cache_dir=_get_model_cache_dir(),
        ocr_backend=ocr_backend,
        pdf_backend=ocr_backend,
        lighton_profile=profile,
        lighton_repo_id=_get_trimmed_str(ENV_LIGHTON_REPO_ID, DEFAULT_LIGHTON_REPO_ID),
        lighton_model_file=_get_trimmed_str(ENV_LIGHTON_MODEL_FILE, DEFAULT_LIGHTON_MODEL_FILE),
        lighton_mmproj_file=_get_trimmed_str(ENV_LIGHTON_MMPROJ_FILE, DEFAULT_LIGHTON_MMPROJ_FILE),
        lighton_model_dir=_get_trimmed_str(ENV_LIGHTON_MODEL_DIR, DEFAULT_LIGHTON_MODEL_DIR),
        lighton_auto_download=_get_bool(ENV_LIGHTON_AUTO_DOWNLOAD, DEFAULT_LIGHTON_AUTO_DOWNLOAD),
        llama_server_path=_get_optional_str(ENV_LLAMA_SERVER_PATH),
        lighton_server_host=_get_trimmed_str(ENV_LIGHTON_SERVER_HOST, DEFAULT_LIGHTON_SERVER_HOST),
        lighton_server_port=_get_positive_int(ENV_LIGHTON_SERVER_PORT, DEFAULT_LIGHTON_SERVER_PORT),
        lighton_start_timeout_seconds=_get_positive_int(
            ENV_LIGHTON_START_TIMEOUT_SECONDS,
            DEFAULT_LIGHTON_START_TIMEOUT_SECONDS,
        ),
        lighton_request_timeout_seconds=_get_positive_int(
            ENV_LIGHTON_REQUEST_TIMEOUT_SECONDS,
            DEFAULT_LIGHTON_REQUEST_TIMEOUT_SECONDS,
        ),
        lighton_ctx_size=_get_positive_int(ENV_LIGHTON_CTX_SIZE, DEFAULT_LIGHTON_CTX_SIZE),
        lighton_n_gpu_layers=_get_int_with_min(
            ENV_LIGHTON_N_GPU_LAYERS,
            DEFAULT_LIGHTON_N_GPU_LAYERS,
            min_value=-1,
        ),
        lighton_parallel=_get_positive_int(ENV_LIGHTON_PARALLEL, default_lighton_parallel),
        lighton_threads=_get_positive_int(ENV_LIGHTON_THREADS, DEFAULT_LIGHTON_THREADS),
        lighton_threads_batch=_get_positive_int(
            ENV_LIGHTON_THREADS_BATCH,
            DEFAULT_LIGHTON_THREADS_BATCH,
        ),
        lighton_flash_attn=_get_bool(ENV_LIGHTON_FLASH_ATTN, DEFAULT_LIGHTON_FLASH_ATTN),
        lighton_max_tokens=_get_positive_int(ENV_LIGHTON_MAX_TOKENS, default_lighton_max_tokens),
        lighton_temperature=_get_float_with_min(
            ENV_LIGHTON_TEMPERATURE,
            DEFAULT_LIGHTON_TEMPERATURE,
            min_value=0.0,
        ),
        lighton_top_p=_get_float_with_min(ENV_LIGHTON_TOP_P, DEFAULT_LIGHTON_TOP_P, min_value=0.0),
        lighton_page_concurrency=_get_positive_int(
            ENV_LIGHTON_PAGE_CONCURRENCY,
            default_lighton_page_concurrency,
        ),
        lighton_render_dpi=_get_positive_int(ENV_LIGHTON_RENDER_DPI, DEFAULT_LIGHTON_RENDER_DPI),
        lighton_render_max_edge=_get_positive_int(
            ENV_LIGHTON_RENDER_MAX_EDGE,
            DEFAULT_LIGHTON_RENDER_MAX_EDGE,
        ),
        lighton_merge_policy=_get_lighton_merge_policy(),
        lighton_fast_pass=_get_bool(ENV_LIGHTON_FAST_PASS, default_lighton_fast_pass),
        lighton_fast_render_dpi=_get_positive_int(
            ENV_LIGHTON_FAST_RENDER_DPI,
            default_lighton_fast_render_dpi,
        ),
        lighton_fast_render_max_edge=_get_positive_int(
            ENV_LIGHTON_FAST_RENDER_MAX_EDGE,
            default_lighton_fast_render_max_edge,
        ),
        lighton_fast_retry=_get_bool(ENV_LIGHTON_FAST_RETRY, default_lighton_fast_retry),
        lighton_secondary_table_repair=_get_bool(
            ENV_LIGHTON_SECONDARY_TABLE_REPAIR,
            default_lighton_secondary_table_repair,
        ),
        lighton_retry_render_dpi=_get_positive_int(
            ENV_LIGHTON_RETRY_RENDER_DPI,
            DEFAULT_LIGHTON_RETRY_RENDER_DPI,
        ),
        lighton_retry_render_max_edge=_get_positive_int(
            ENV_LIGHTON_RETRY_RENDER_MAX_EDGE,
            DEFAULT_LIGHTON_RETRY_RENDER_MAX_EDGE,
        ),
        lighton_retry_min_quality=_get_float_with_min(
            ENV_LIGHTON_RETRY_MIN_QUALITY,
            DEFAULT_LIGHTON_RETRY_MIN_QUALITY,
            min_value=0.0,
        ),
        lighton_retry_quality_gap=_get_float_with_min(
            ENV_LIGHTON_RETRY_QUALITY_GAP,
            DEFAULT_LIGHTON_RETRY_QUALITY_GAP,
            min_value=0.0,
        ),
        lighton_retry_min_pym_text_len=_get_positive_int(
            ENV_LIGHTON_RETRY_MIN_PYM_TEXT_LEN,
            DEFAULT_LIGHTON_RETRY_MIN_PYM_TEXT_LEN,
        ),
        lighton_fast_non_table_max_edge=_get_int_with_min(
            ENV_LIGHTON_FAST_NON_TABLE_MAX_EDGE,
            default_lighton_fast_non_table_max_edge,
            min_value=0,
        ),
        lighton_fast_table_likelihood_threshold=_get_float_with_min(
            ENV_LIGHTON_FAST_TABLE_LIKELIHOOD_THRESHOLD,
            DEFAULT_LIGHTON_FAST_TABLE_LIKELIHOOD_THRESHOLD,
            min_value=0.0,
        ),
        lighton_fast_max_tokens_text=_get_positive_int(
            ENV_LIGHTON_FAST_MAX_TOKENS_TEXT,
            default_lighton_fast_max_tokens_text,
        ),
        lighton_fast_max_tokens_table=_get_positive_int(
            ENV_LIGHTON_FAST_MAX_TOKENS_TABLE,
            default_lighton_fast_max_tokens_table,
        ),
        lighton_fast_postprocess_mode=_get_lighton_fast_postprocess_mode(
            default_lighton_fast_postprocess_mode
        ),
        lighton_pymupdf_page_fallback_mode=_get_lighton_pymupdf_page_fallback_mode(),
    )
