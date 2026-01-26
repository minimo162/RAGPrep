from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


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


def _resolve_llava_cli(settings: LlamaCppCliSettings) -> str:
    candidates: list[str] = []

    if settings.llava_cli_path:
        candidates.append(settings.llava_cli_path)

    candidates.extend(["llava-cli", "llava-cli.exe"])

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
        "or ensure llava-cli is available on PATH."
    )


def _build_argv(
    *,
    llava_cli: str,
    settings: LlamaCppCliSettings,
    image_path: Path,
    prompt: str,
    max_new_tokens: int,
) -> list[str]:
    argv = [
        llava_cli,
        "-m",
        settings.model_path,
        "--mmproj",
        settings.mmproj_path,
        "--image",
        str(image_path),
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


def ocr_image(*, image: Image.Image, settings: LlamaCppCliSettings, max_new_tokens: int) -> str:
    _validate_paths(settings)
    llava_cli = _resolve_llava_cli(settings)

    prompt = "Extract all text from this image and return it as Markdown."

    with tempfile.NamedTemporaryFile(
        prefix="ragprep_llava_", suffix=".png", delete=False
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
            stderr = (result.stderr or "").strip()
            stderr_summary = f"\n\nstderr:\n{stderr}" if stderr else ""
            raise RuntimeError(
                f"llava-cli failed with exit code {result.returncode}.{stderr_summary}"
            )

        return _extract_text(result.stdout or "")
    finally:
        try:
            image_path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
