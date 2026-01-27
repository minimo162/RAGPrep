from __future__ import annotations

import base64
import binascii
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
_VARIANT_VULKAN = "vulkan"
_VARIANT_AVX2 = "avx2"


@dataclass(frozen=True)
class _CliCandidate:
    path: Path
    variant: str


@dataclass(frozen=True)
class LlamaCppCliSettings:
    llava_cli_path: str | None
    model_path: str
    mmproj_path: str
    n_ctx: int | None
    n_threads: int | None
    n_gpu_layers: int | None
    temperature: float
    top_p: float
    repeat_penalty: float
    repeat_last_n: int


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


def _llama_cpp_cli_names() -> list[str]:
    return [
        "llama-mtmd-cli.exe",
        "llama-mtmd-cli",
        "llava-cli.exe",
        "llava-cli",
        "llama-llava-cli.exe",
        "llama-llava-cli",
    ]


def _llama_cpp_base_dirs(repo_root: Path) -> list[Path]:
    """
    Return llama.cpp base directories to probe in priority order.

    - Standalone layout: <standalone>/bin/llama.cpp
    - Dev layout:        <repo>/dist/standalone/bin/llama.cpp
    """
    base_dirs: list[Path] = []
    standalone_root = _standalone_root(repo_root)
    if standalone_root is not None:
        base_dirs.append(standalone_root / "bin" / "llama.cpp")
    base_dirs.append(repo_root / "dist" / "standalone" / "bin" / "llama.cpp")
    return base_dirs


def _standalone_variant_candidates(repo_root: Path) -> list[_CliCandidate]:
    names = _llama_cpp_cli_names()
    candidates: list[_CliCandidate] = []
    for base_dir in _llama_cpp_base_dirs(repo_root):
        variant_dirs = [
            (base_dir / _VARIANT_VULKAN, _VARIANT_VULKAN),
            (base_dir / _VARIANT_AVX2, _VARIANT_AVX2),
            # Root-level copy is kept for backward compatibility and maps to AVX2.
            (base_dir, _VARIANT_AVX2),
        ]
        for variant_dir, variant_name in variant_dirs:
            for name in names:
                candidates.append(_CliCandidate(path=variant_dir / name, variant=variant_name))
    return candidates


def _standalone_cli_candidates(repo_root: Path) -> list[Path]:
    return [candidate.path for candidate in _standalone_variant_candidates(repo_root)]


def _resolve_llava_cli(
    settings: LlamaCppCliSettings,
    *,
    repo_root: Path | None = None,
) -> str:
    resolved_root = _REPO_ROOT if repo_root is None else repo_root
    explicit = _normalize_path_value(settings.llava_cli_path)
    base_dirs = _llama_cpp_base_dirs(resolved_root)

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
    if base_dirs:
        checked_dirs = ", ".join(str(path) for path in base_dirs)
        details.append(f"Standalone dirs checked: {checked_dirs}")

    detail_text = ("\n" + "\n".join(details)) if details else ""
    raise RuntimeError(
        "llama.cpp multimodal CLI executable not found. "
        "Set LIGHTONOCR_LLAVA_CLI_PATH to the full path of llama-mtmd-cli(.exe) "
        "(or llava-cli(.exe) / llama-llava-cli(.exe)), "
        "or ensure it is available on PATH." + detail_text
    )


def _variant_from_path(path: Path) -> str | None:
    parts = {part.lower() for part in path.parts}
    if _VARIANT_VULKAN in parts:
        return _VARIANT_VULKAN
    if _VARIANT_AVX2 in parts:
        return _VARIANT_AVX2
    if path.parent.name.lower() == "llama.cpp":
        # Root-level copy is kept for backward compatibility and maps to AVX2.
        return _VARIANT_AVX2
    return None


def _resolve_llava_cli_variant(
    settings: LlamaCppCliSettings,
    *,
    variant: str,
    repo_root: Path | None = None,
) -> str | None:
    resolved_root = _REPO_ROOT if repo_root is None else repo_root
    variant_key = variant.strip().lower()

    for candidate in _standalone_variant_candidates(resolved_root):
        if candidate.variant != variant_key:
            continue
        if candidate.path.is_file():
            return str(candidate.path)

    for name_candidate in _llama_cpp_cli_names():
        found = shutil.which(name_candidate)
        if found is not None:
            return found

    return None


def _run_llava_cli(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, check=False)  # noqa: S603


def _format_cli_failure(result: subprocess.CompletedProcess[str], *, cli_path: str) -> str:
    stderr = (result.stderr or "").strip()
    stderr_summary = f"\n\nstderr:\n{stderr}" if stderr else ""
    return f"llava-cli failed ({cli_path}) with exit code {result.returncode}.{stderr_summary}"


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
        "--temp",
        str(settings.temperature),
        "--top-p",
        str(settings.top_p),
        "--repeat-penalty",
        str(settings.repeat_penalty),
        "--repeat-last-n",
        str(settings.repeat_last_n),
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
    with tempfile.NamedTemporaryFile(prefix="ragprep_llava_", suffix=".png", delete=False) as tmp:
        image_path = Path(tmp.name)
    try:
        image.save(image_path, format="PNG")
        return _ocr_with_image_path(
            image_path=image_path,
            settings=settings,
            max_new_tokens=max_new_tokens,
        )
    finally:
        try:
            image_path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass


def ocr_image_base64(
    *,
    image_base64: str,
    settings: LlamaCppCliSettings,
    max_new_tokens: int,
) -> str:
    if not image_base64:
        raise ValueError("image_base64 is empty")

    payload = image_base64.strip()
    if payload.startswith("data:"):
        comma_index = payload.find(",")
        if comma_index != -1:
            payload = payload[comma_index + 1 :].strip()

    try:
        image_bytes = base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image_base64 is not valid base64") from exc

    if not image_bytes:
        raise ValueError("image_base64 is empty")

    with tempfile.NamedTemporaryFile(prefix="ragprep_llava_", suffix=".png", delete=False) as tmp:
        image_path = Path(tmp.name)
        tmp.write(image_bytes)

    try:
        return _ocr_with_image_path(
            image_path=image_path,
            settings=settings,
            max_new_tokens=max_new_tokens,
        )
    finally:
        try:
            image_path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass


def _ocr_with_image_path(
    *,
    image_path: Path,
    settings: LlamaCppCliSettings,
    max_new_tokens: int,
) -> str:
    model_path, mmproj_path = _validate_paths(settings)
    explicit = _normalize_path_value(settings.llava_cli_path)
    explicit_is_file = False
    if explicit:
        try:
            explicit_is_file = Path(explicit).is_file()
        except OSError:
            explicit_is_file = False
    llava_cli = _resolve_llava_cli(settings)
    fallback_cli: str | None = None
    if not explicit_is_file:
        preferred_variant = _variant_from_path(Path(llava_cli))
        if preferred_variant == _VARIANT_VULKAN:
            fallback_cli = _resolve_llava_cli_variant(settings, variant=_VARIANT_AVX2)
            if fallback_cli == llava_cli:
                fallback_cli = None

    prompt = "Extract all text from this image and return it as Markdown."

    argv = _build_argv(
        llava_cli=llava_cli,
        settings=settings,
        model_path=model_path,
        mmproj_path=mmproj_path,
        image_path=image_path,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )

    result = _run_llava_cli(argv)
    if result.returncode == 0:
        return _extract_text(result.stdout or "")

    primary_error = _format_cli_failure(result, cli_path=llava_cli)
    if fallback_cli:
        fallback_argv = _build_argv(
            llava_cli=fallback_cli,
            settings=settings,
            model_path=model_path,
            mmproj_path=mmproj_path,
            image_path=image_path,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        fallback_result = _run_llava_cli(fallback_argv)
        if fallback_result.returncode == 0:
            return _extract_text(fallback_result.stdout or "")

        fallback_error = _format_cli_failure(fallback_result, cli_path=fallback_cli)
        raise RuntimeError(
            f"{primary_error}\n\nFallback to AVX2 also failed.\n\n{fallback_error}"
        )

    raise RuntimeError(primary_error)
