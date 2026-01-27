from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_GGUF_MODEL = "LIGHTONOCR_GGUF_MODEL_PATH"
_ENV_GGUF_MMPROJ = "LIGHTONOCR_GGUF_MMPROJ_PATH"
_BUILD_HINT = "Rebuild standalone: scripts/build-standalone.ps1 -Clean"


@dataclass(frozen=True)
class LlamaCppCliSettings:
    llava_cli_path: str | None
    model_path: str
    mmproj_path: str
    n_ctx: int | None
    n_threads: int | None
    n_gpu_layers: int | None


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


def _format_path_error(
    *,
    env_name: str,
    path_value: str,
    expected_dir: Path,
    issue: str,
) -> RuntimeError:
    issue_detail = f"{issue}: {path_value}" if path_value else issue
    if path_value:
        env_detail = f"Env: {env_name}={path_value!r}"
    else:
        env_detail = f"Set {env_name} to a valid .gguf file."
    details = [
        issue_detail,
        env_detail,
        f"Expected under: {expected_dir}",
        _BUILD_HINT,
    ]
    return RuntimeError("\n".join(details))


def _normalize_path_value(value: str | None) -> str:
    if value is None:
        return ""
    raw = value.strip()
    if not raw:
        return ""

    wrappers = [('"', '"'), ("'", "'"), ("<", ">")]
    for start, end in wrappers:
        if raw.startswith(start) and raw.endswith(end) and len(raw) >= 2:
            raw = raw[1:-1].strip()

    if raw.lower().startswith("file://"):
        try:
            parsed = urlparse(raw)
            if parsed.scheme.lower() == "file":
                raw = url2pathname(parsed.path).strip()
        except Exception:
            return raw

    return raw


def _validate_paths(settings: LlamaCppCliSettings) -> tuple[str, str]:
    model_path_value = _normalize_path_value(settings.model_path)
    mmproj_path_value = _normalize_path_value(settings.mmproj_path)
    expected_dir = _expected_gguf_dir(_REPO_ROOT)
    if not model_path_value:
        raise _format_path_error(
            env_name=_ENV_GGUF_MODEL,
            path_value=model_path_value,
            expected_dir=expected_dir,
            issue="GGUF model path is empty",
        )
    if not mmproj_path_value:
        raise _format_path_error(
            env_name=_ENV_GGUF_MMPROJ,
            path_value=mmproj_path_value,
            expected_dir=expected_dir,
            issue="GGUF mmproj path is empty",
        )

    model_path = Path(model_path_value)
    mmproj_path = Path(mmproj_path_value)
    try:
        model_is_file = model_path.is_file()
    except OSError as exc:
        raise _format_path_error(
            env_name=_ENV_GGUF_MODEL,
            path_value=model_path_value,
            expected_dir=expected_dir,
            issue="Invalid GGUF model path",
        ) from exc
    try:
        mmproj_is_file = mmproj_path.is_file()
    except OSError as exc:
        raise _format_path_error(
            env_name=_ENV_GGUF_MMPROJ,
            path_value=mmproj_path_value,
            expected_dir=expected_dir,
            issue="Invalid GGUF mmproj path",
        ) from exc

    if not model_is_file:
        raise _format_path_error(
            env_name=_ENV_GGUF_MODEL,
            path_value=model_path_value,
            expected_dir=expected_dir,
            issue="GGUF model file not found",
        )
    if not mmproj_is_file:
        raise _format_path_error(
            env_name=_ENV_GGUF_MMPROJ,
            path_value=mmproj_path_value,
            expected_dir=expected_dir,
            issue="GGUF mmproj file not found",
        )
    return model_path_value, mmproj_path_value


def _standalone_cli_candidates(repo_root: Path) -> list[Path]:
    base_dir = repo_root / "dist" / "standalone" / "bin" / "llama.cpp"
    names = [
        "llama-mtmd-cli.exe",
        "llama-mtmd-cli",
        "llava-cli.exe",
        "llava-cli",
        "llama-llava-cli.exe",
        "llama-llava-cli",
    ]
    return [base_dir / name for name in names]


def _resolve_llava_cli(
    settings: LlamaCppCliSettings,
    *,
    repo_root: Path | None = None,
) -> str:
    resolved_root = _REPO_ROOT if repo_root is None else repo_root
    explicit = _normalize_path_value(settings.llava_cli_path)

    path_candidates: list[Path] = []
    if explicit:
        path_candidates.append(Path(explicit))
    path_candidates.extend(_standalone_cli_candidates(resolved_root))

    for path_candidate in path_candidates:
        if path_candidate.is_file():
            return str(path_candidate)

    name_candidates = [
        "llama-mtmd-cli",
        "llama-mtmd-cli.exe",
        "llava-cli",
        "llava-cli.exe",
        "llama-llava-cli",
        "llama-llava-cli.exe",
    ]
    for name_candidate in name_candidates:
        found: str | None = shutil.which(name_candidate)
        if found is not None:
            return found

    details: list[str] = []
    if explicit:
        details.append(f"Explicit path not found: {explicit}")
    details.append(f"Standalone dir checked: {resolved_root / 'dist' / 'standalone'}")

    detail_text = ("\n" + "\n".join(details)) if details else ""
    raise RuntimeError(
        "llama.cpp multimodal CLI executable not found. "
        "Set LIGHTONOCR_LLAVA_CLI_PATH to the full path of llama-mtmd-cli(.exe) "
        "(or llava-cli(.exe) / llama-llava-cli(.exe)), "
        "or ensure it is available on PATH." + detail_text
    )


def _build_argv(
    *,
    llava_cli: str,
    settings: LlamaCppCliSettings,
    model_path: str,
    mmproj_path: str,
    image_path: Path,
    prompt: str,
    max_new_tokens: int,
) -> list[str]:
    argv = [
        llava_cli,
        "-m",
        model_path,
        "--mmproj",
        mmproj_path,
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
    model_path, mmproj_path = _validate_paths(settings)
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
            model_path=model_path,
            mmproj_path=mmproj_path,
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
