from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


def test_lightonocr_module_is_importable() -> None:
    from ragprep.ocr import lightonocr

    assert callable(lightonocr.ocr_image)


def test_lightonocr_dry_run_returns_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    from PIL import Image

    from ragprep.ocr import lightonocr

    monkeypatch.setenv("LIGHTONOCR_DRY_RUN", "1")
    image = Image.new("RGB", (1, 1), color=(255, 255, 255))
    assert lightonocr.ocr_image(image) == lightonocr.DRY_RUN_OUTPUT


def test_llamacpp_extract_text_strips_markers() -> None:
    from ragprep.ocr import llamacpp_cli_runtime as runtime

    text = "prefix\n### Assistant:\nHello"
    assert runtime._extract_text(text) == "Hello"


def test_llamacpp_validate_paths_reports_missing_model(tmp_path: Path) -> None:
    from ragprep.ocr import llamacpp_cli_runtime as runtime

    mmproj_path = tmp_path / "mmproj.gguf"
    mmproj_path.write_text("x", encoding="utf-8")
    settings = runtime.LlamaCppCliSettings(
        llava_cli_path=None,
        model_path=str(tmp_path / "missing.gguf"),
        mmproj_path=str(mmproj_path),
        n_ctx=None,
        n_threads=None,
        n_gpu_layers=None,
        temperature=0.2,
        repeat_penalty=1.15,
        repeat_last_n=128,
    )

    with pytest.raises(RuntimeError, match="GGUF model file not found"):
        runtime._validate_paths(settings)


def test_llamacpp_validate_paths_includes_env_and_expected_dir(tmp_path: Path) -> None:
    from ragprep.ocr import llamacpp_cli_runtime as runtime

    mmproj_path = tmp_path / "mmproj.gguf"
    mmproj_path.write_text("x", encoding="utf-8")
    settings = runtime.LlamaCppCliSettings(
        llava_cli_path=None,
        model_path=str(tmp_path / "missing.gguf"),
        mmproj_path=str(mmproj_path),
        n_ctx=None,
        n_threads=None,
        n_gpu_layers=None,
        temperature=0.2,
        repeat_penalty=1.15,
        repeat_last_n=128,
    )

    with pytest.raises(RuntimeError) as excinfo:
        runtime._validate_paths(settings)
    message = str(excinfo.value)
    assert "LIGHTONOCR_GGUF_MODEL_PATH" in message
    assert "Expected under:" in message


def test_llamacpp_resolve_cli_prefers_explicit_path(tmp_path: Path) -> None:
    from ragprep.ocr import llamacpp_cli_runtime as runtime

    cli_path = tmp_path / "llama-mtmd-cli.exe"
    cli_path.write_text("x", encoding="utf-8")
    settings = runtime.LlamaCppCliSettings(
        llava_cli_path=f' "{cli_path}" ',
        model_path="model.gguf",
        mmproj_path="mmproj.gguf",
        n_ctx=None,
        n_threads=None,
        n_gpu_layers=None,
        temperature=0.2,
        repeat_penalty=1.15,
        repeat_last_n=128,
    )

    resolved = runtime._resolve_llava_cli(settings, repo_root=tmp_path)
    assert resolved == str(cli_path)


def test_llamacpp_resolve_cli_uses_standalone_before_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ragprep.ocr import llamacpp_cli_runtime as runtime

    cli_path = tmp_path / "dist" / "standalone" / "bin" / "llama.cpp" / "llama-mtmd-cli.exe"
    cli_path.parent.mkdir(parents=True, exist_ok=True)
    cli_path.write_text("x", encoding="utf-8")

    settings = runtime.LlamaCppCliSettings(
        llava_cli_path=None,
        model_path="model.gguf",
        mmproj_path="mmproj.gguf",
        n_ctx=None,
        n_threads=None,
        n_gpu_layers=None,
        temperature=0.2,
        repeat_penalty=1.15,
        repeat_last_n=128,
    )

    def _forbidden_which(_name: str) -> str | None:
        raise AssertionError("PATH search should not be used when standalone exists")

    monkeypatch.setattr(runtime.shutil, "which", _forbidden_which)

    resolved = runtime._resolve_llava_cli(settings, repo_root=tmp_path)
    assert resolved == str(cli_path)


def test_llamacpp_resolve_cli_prefers_vulkan_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ragprep.ocr import llamacpp_cli_runtime as runtime

    vulkan_cli = (
        tmp_path
        / "dist"
        / "standalone"
        / "bin"
        / "llama.cpp"
        / "vulkan"
        / "llama-mtmd-cli.exe"
    )
    avx2_cli = (
        tmp_path
        / "dist"
        / "standalone"
        / "bin"
        / "llama.cpp"
        / "avx2"
        / "llama-mtmd-cli.exe"
    )
    vulkan_cli.parent.mkdir(parents=True, exist_ok=True)
    avx2_cli.parent.mkdir(parents=True, exist_ok=True)
    vulkan_cli.write_text("x", encoding="utf-8")
    avx2_cli.write_text("x", encoding="utf-8")

    settings = runtime.LlamaCppCliSettings(
        llava_cli_path=None,
        model_path="model.gguf",
        mmproj_path="mmproj.gguf",
        n_ctx=None,
        n_threads=None,
        n_gpu_layers=None,
        temperature=0.2,
        repeat_penalty=1.15,
        repeat_last_n=128,
    )

    monkeypatch.setattr(runtime.shutil, "which", lambda _name: None)

    resolved = runtime._resolve_llava_cli(settings, repo_root=tmp_path)
    assert resolved == str(vulkan_cli)


def test_llamacpp_resolve_cli_supports_standalone_root_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ragprep.ocr import llamacpp_cli_runtime as runtime

    standalone_root = tmp_path / "standalone"
    app_root = standalone_root / "app"
    vulkan_cli = (
        standalone_root / "bin" / "llama.cpp" / "vulkan" / "llama-mtmd-cli.exe"
    )
    vulkan_cli.parent.mkdir(parents=True, exist_ok=True)
    app_root.mkdir(parents=True, exist_ok=True)
    vulkan_cli.write_text("x", encoding="utf-8")

    settings = runtime.LlamaCppCliSettings(
        llava_cli_path=None,
        model_path="model.gguf",
        mmproj_path="mmproj.gguf",
        n_ctx=None,
        n_threads=None,
        n_gpu_layers=None,
        temperature=0.2,
        repeat_penalty=1.15,
        repeat_last_n=128,
    )

    monkeypatch.setattr(runtime.shutil, "which", lambda _name: None)

    resolved = runtime._resolve_llava_cli(settings, repo_root=app_root)
    assert resolved == str(vulkan_cli)


def test_llamacpp_resolve_cli_missing_raises_clear_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ragprep.ocr import llamacpp_cli_runtime as runtime

    settings = runtime.LlamaCppCliSettings(
        llava_cli_path=None,
        model_path="model.gguf",
        mmproj_path="mmproj.gguf",
        n_ctx=None,
        n_threads=None,
        n_gpu_layers=None,
        temperature=0.2,
        repeat_penalty=1.15,
        repeat_last_n=128,
    )

    monkeypatch.setattr(runtime.shutil, "which", lambda _name: None)

    with pytest.raises(RuntimeError, match="LIGHTONOCR_LLAVA_CLI_PATH"):
        runtime._resolve_llava_cli(settings, repo_root=tmp_path)


def test_llamacpp_ocr_image_builds_argv_and_normalizes_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from PIL import Image

    from ragprep.ocr import llamacpp_cli_runtime as runtime

    model_path = tmp_path / "LightOnOCR-2-1B-Q6_K.gguf"
    mmproj_path = tmp_path / "mmproj-BF16.gguf"
    cli_path = tmp_path / "llama-mtmd-cli.exe"

    model_path.write_text("x", encoding="utf-8")
    mmproj_path.write_text("x", encoding="utf-8")
    cli_path.write_text("x", encoding="utf-8")

    captured: dict[str, list[str]] = {}

    def _fake_run(
        argv: list[str],
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> SimpleNamespace:
        captured["argv"] = argv
        return SimpleNamespace(returncode=0, stdout="Assistant: OK", stderr="")

    monkeypatch.setattr(runtime.subprocess, "run", _fake_run)

    settings = runtime.LlamaCppCliSettings(
        llava_cli_path=f" <{cli_path}> ",
        model_path=model_path.resolve().as_uri(),
        mmproj_path=f' "{mmproj_path}" ',
        n_ctx=2048,
        n_threads=4,
        n_gpu_layers=0,
        temperature=0.2,
        repeat_penalty=1.15,
        repeat_last_n=128,
    )

    image = Image.new("RGB", (2, 2), color=(0, 0, 0))
    text = runtime.ocr_image(image=image, settings=settings, max_new_tokens=64)

    assert text == "OK"
    argv = captured["argv"]
    model_value = argv[argv.index("-m") + 1]
    mmproj_value = argv[argv.index("--mmproj") + 1]
    temp_value = argv[argv.index("--temp") + 1]
    repeat_penalty_value = argv[argv.index("--repeat-penalty") + 1]
    repeat_last_n_value = argv[argv.index("--repeat-last-n") + 1]

    assert Path(model_value) == model_path
    assert Path(mmproj_value) == mmproj_path
    assert temp_value == "0.2"
    assert repeat_penalty_value == "1.15"
    assert repeat_last_n_value == "128"
    assert "-c" in argv and "2048" in argv
    assert "-t" in argv and "4" in argv
    assert "-ngl" in argv and "0" in argv


def test_llamacpp_ocr_image_falls_back_from_vulkan_to_avx2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from PIL import Image

    from ragprep.ocr import llamacpp_cli_runtime as runtime

    model_path = tmp_path / "LightOnOCR-2-1B-Q6_K.gguf"
    mmproj_path = tmp_path / "mmproj-BF16.gguf"
    model_path.write_text("x", encoding="utf-8")
    mmproj_path.write_text("x", encoding="utf-8")

    vulkan_cli = (
        tmp_path
        / "dist"
        / "standalone"
        / "bin"
        / "llama.cpp"
        / "vulkan"
        / "llama-mtmd-cli.exe"
    )
    avx2_cli = (
        tmp_path
        / "dist"
        / "standalone"
        / "bin"
        / "llama.cpp"
        / "avx2"
        / "llama-mtmd-cli.exe"
    )
    vulkan_cli.parent.mkdir(parents=True, exist_ok=True)
    avx2_cli.parent.mkdir(parents=True, exist_ok=True)
    vulkan_cli.write_text("x", encoding="utf-8")
    avx2_cli.write_text("x", encoding="utf-8")

    monkeypatch.setattr(runtime, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(runtime.shutil, "which", lambda _name: None)

    calls: list[str] = []

    def _fake_run(
        argv: list[str],
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> SimpleNamespace:
        cli_path = str(argv[0])
        calls.append(cli_path)
        if "vulkan" in cli_path.lower():
            return SimpleNamespace(returncode=1, stdout="", stderr="vulkan unavailable")
        return SimpleNamespace(returncode=0, stdout="Assistant: OK", stderr="")

    monkeypatch.setattr(runtime.subprocess, "run", _fake_run)

    settings = runtime.LlamaCppCliSettings(
        llava_cli_path=None,
        model_path=str(model_path),
        mmproj_path=str(mmproj_path),
        n_ctx=None,
        n_threads=None,
        n_gpu_layers=None,
        temperature=0.2,
        repeat_penalty=1.15,
        repeat_last_n=128,
    )

    image = Image.new("RGB", (2, 2), color=(0, 0, 0))
    text = runtime.ocr_image(image=image, settings=settings, max_new_tokens=64)

    assert text == "OK"
    assert len(calls) >= 2
    assert "vulkan" in calls[0].lower()
    assert "avx2" in calls[1].lower()
