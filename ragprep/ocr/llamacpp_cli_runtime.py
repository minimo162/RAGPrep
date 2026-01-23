from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

_MAX_PROCESS_OUTPUT_CHARS = 8000
ENV_IMAGE_TMP_DIR = "LIGHTONOCR_IMAGE_TMP_DIR"
_WINDOWS_LONG_PATH_THRESHOLD = 240


@dataclass(frozen=True)
class LlamaCppCliSettings:
    llava_cli_path: str | None
    model_path: str
    mmproj_path: str
    n_ctx: int | None
    n_threads: int | None
    n_gpu_layers: int | None


def _validate_paths(settings: LlamaCppCliSettings) -> None:
    model_path = Path(settings.model_path)
    mmproj_path = Path(settings.mmproj_path)
    try:
        model_is_file = model_path.is_file()
    except OSError as exc:
        raise RuntimeError(f"Invalid GGUF model path: {model_path!s}") from exc
    try:
        mmproj_is_file = mmproj_path.is_file()
    except OSError as exc:
        raise RuntimeError(f"Invalid GGUF mmproj path: {mmproj_path!s}") from exc

    if not model_is_file:
        raise RuntimeError(f"GGUF model file not found: {model_path}")
    if not mmproj_is_file:
        raise RuntimeError(f"GGUF mmproj file not found: {mmproj_path}")


def _default_bundled_llava_cli() -> Path | None:
    """
    In the Windows standalone layout, build scripts bundle llama.cpp under:
      <standalone_root>/bin/llama.cpp/llava-cli.exe

    When running from the standalone `app/` directory, this module resolves to:
      <standalone_root>/app/ragprep/ocr/llamacpp_cli_runtime.py
    """

    try:
        standalone_root = Path(__file__).resolve().parents[3]
    except Exception:  # noqa: BLE001
        return None

    candidate = standalone_root / "bin" / "llama.cpp" / "llava-cli.exe"
    if candidate.is_file():
        return candidate
    return None


def _resolve_llava_cli(settings: LlamaCppCliSettings) -> str:
    candidates: list[str] = []

    if settings.llava_cli_path:
        candidates.append(settings.llava_cli_path)
    else:
        bundled = _default_bundled_llava_cli()
        if bundled is not None:
            candidates.append(str(bundled))

    candidates.extend(["llava-cli", "llava-cli.exe", "llama-llava-cli", "llama-llava-cli.exe"])

    for candidate in candidates:
        candidate_path = Path(candidate)
        if candidate_path.is_file():
            return str(candidate_path)

        found = shutil.which(candidate)
        if found:
            return found

    raise RuntimeError(
        "llava-cli executable not found. "
        "Set LIGHTONOCR_LLAVA_CLI_PATH to the full path of llava-cli(.exe) "
        "(or llama-llava-cli(.exe)), or ensure it is available on PATH."
    )


def _contains_non_ascii(text: str) -> bool:
    return any(ord(ch) > 127 for ch in text)


def _needs_windows_short_path(path: str) -> bool:
    if os.name != "nt":
        return False
    if _contains_non_ascii(path):
        return True
    return len(path) >= _WINDOWS_LONG_PATH_THRESHOLD


def _get_windows_short_path(path: str) -> str:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_short_path_name_w = kernel32.GetShortPathNameW
    get_short_path_name_w.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    get_short_path_name_w.restype = wintypes.DWORD

    input_path = str(Path(path))

    buffer_len = 260
    while True:
        buffer = ctypes.create_unicode_buffer(buffer_len)
        result = get_short_path_name_w(input_path, buffer, buffer_len)
        if result == 0:
            error = ctypes.get_last_error()
            raise OSError(error, f"GetShortPathNameW failed for {input_path!r}")
        if result < buffer_len:
            return buffer.value
        buffer_len = int(result) + 1


def _maybe_windows_short_path(path: str) -> str:
    if not _needs_windows_short_path(path):
        return path
    try:
        return _get_windows_short_path(path)
    except Exception:  # noqa: BLE001
        return path


def _strip_wrappers(value: str) -> str:
    raw = value.strip()
    if not raw:
        return raw

    wrappers = [('"', '"'), ("'", "'"), ("<", ">")]
    for start, end in wrappers:
        if raw.startswith(start) and raw.endswith(end) and len(raw) >= 2:
            raw = raw[1:-1].strip()
    return raw


def _resolve_image_temp_dir() -> str | None:
    raw = os.getenv(ENV_IMAGE_TMP_DIR)
    if raw is None:
        return None

    raw = _strip_wrappers(raw)
    if not raw:
        return None

    temp_dir = Path(raw)
    try:
        temp_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"{ENV_IMAGE_TMP_DIR} must be a writable directory, got: {raw!r}"
        ) from exc
    if not temp_dir.is_dir():
        raise RuntimeError(f"{ENV_IMAGE_TMP_DIR} must be a directory, got: {raw!r}")
    return str(temp_dir)


def _build_argv(
    *,
    llava_cli: str,
    settings: LlamaCppCliSettings,
    image_path: Path,
    prompt: str,
    max_new_tokens: int,
) -> list[str]:
    model_path = _maybe_windows_short_path(settings.model_path)
    mmproj_path = _maybe_windows_short_path(settings.mmproj_path)
    image_arg = _maybe_windows_short_path(str(image_path))

    argv = [
        llava_cli,
        "-m",
        model_path,
        "--mmproj",
        mmproj_path,
        "--image",
        image_arg,
        "-p",
        prompt,
        "-n",
        str(max_new_tokens),
    ]

    if settings.n_ctx is not None:
        argv.extend(["-c", str(settings.n_ctx)])
    if settings.n_threads is not None:
        argv.extend(["-t", str(settings.n_threads)])
    if settings.n_gpu_layers is not None:
        argv.extend(["-ngl", str(settings.n_gpu_layers)])

    return argv


def _extract_text(stdout: str) -> str:
    text = stdout.replace("\r\n", "\n").strip()
    if not text:
        return ""

    markers = [
        "ASSISTANT:",
        "Assistant:",
        "assistant:",
        "### Assistant:",
        "### assistant:",
    ]
    for marker in markers:
        idx = text.rfind(marker)
        if idx != -1:
            return text[idx + len(marker) :].strip()

    return text


def _normalize_process_output(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _format_process_output(name: str, text: str) -> str:
    normalized = _normalize_process_output(text)
    if not normalized:
        return ""

    truncated = False
    if len(normalized) > _MAX_PROCESS_OUTPUT_CHARS:
        normalized = normalized[-_MAX_PROCESS_OUTPUT_CHARS:]
        truncated = True

    label = f"{name} (truncated):" if truncated else f"{name}:"
    return f"{label}\n{normalized}"


def ocr_image(*, image: Image.Image, settings: LlamaCppCliSettings, max_new_tokens: int) -> str:
    _validate_paths(settings)
    llava_cli = _resolve_llava_cli(settings)

    prompt = "Extract all text from this image and return it as Markdown."

    temp_dir = _resolve_image_temp_dir()
    with tempfile.NamedTemporaryFile(
        prefix="ragprep_llava_", suffix=".png", delete=False, dir=temp_dir
    ) as tmp:
        image_path = Path(tmp.name)
    try:
        image.save(image_path, format="PNG")

        argv = _build_argv(
            llava_cli=llava_cli,
            settings=settings,
            image_path=image_path,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

        result = subprocess.run(argv, capture_output=True, text=True, check=False)  # noqa: S603
        if result.returncode != 0:
            stdout_summary = _format_process_output("stdout", result.stdout or "")
            stderr_summary = _format_process_output("stderr", result.stderr or "")
            details = "\n\n".join(
                summary for summary in (stdout_summary, stderr_summary) if summary
            )
            details = f"\n\n{details}" if details else ""
            raise RuntimeError(f"llava-cli failed with exit code {result.returncode}.{details}")

        return _extract_text(result.stdout or "")
    finally:
        try:
            image_path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
