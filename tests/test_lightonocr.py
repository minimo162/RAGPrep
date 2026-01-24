from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from ragprep.ocr import lightonocr


def test_dry_run_skips_settings_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(lightonocr.ENV_DRY_RUN, "1")
    monkeypatch.delenv(lightonocr.ENV_GGUF_MODEL_PATH, raising=False)
    monkeypatch.delenv(lightonocr.ENV_GGUF_MMPROJ_PATH, raising=False)

    def should_not_read_settings() -> lightonocr.LightOnOCRSettings:
        raise AssertionError("Settings should not be read in dry-run mode.")

    monkeypatch.setattr(lightonocr, "get_settings", should_not_read_settings)

    image = Image.new("RGB", (2, 2), color=(0, 0, 0))
    assert lightonocr.ocr_image(image) == lightonocr.DRY_RUN_OUTPUT


def test_missing_gguf_paths_raise_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(lightonocr.ENV_DRY_RUN, raising=False)
    monkeypatch.delenv(lightonocr.ENV_GGUF_MODEL_PATH, raising=False)
    monkeypatch.delenv(lightonocr.ENV_GGUF_MMPROJ_PATH, raising=False)

    image = Image.new("RGB", (2, 2), color=(0, 0, 0))
    with pytest.raises(ValueError) as excinfo:
        lightonocr.ocr_image(image)

    message = str(excinfo.value)
    assert lightonocr.ENV_GGUF_MODEL_PATH in message
    assert lightonocr.ENV_GGUF_MMPROJ_PATH in message


def test_missing_llama_cpp_import_is_actionable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import ragprep.ocr.llamacpp_cli_runtime as cli_runtime

    model_path = tmp_path / "model.gguf"
    mmproj_path = tmp_path / "mmproj.gguf"
    model_path.write_text("x", encoding="utf-8")
    mmproj_path.write_text("y", encoding="utf-8")

    monkeypatch.setenv(lightonocr.ENV_GGUF_MODEL_PATH, str(model_path))
    monkeypatch.setenv(lightonocr.ENV_GGUF_MMPROJ_PATH, str(mmproj_path))
    monkeypatch.delenv(lightonocr.ENV_LLAVA_CLI_PATH, raising=False)
    monkeypatch.setattr(cli_runtime.shutil, "which", lambda _name: None)

    image = Image.new("RGB", (2, 2), color=(0, 0, 0))
    with pytest.raises(RuntimeError, match="llava-cli"):
        lightonocr.ocr_image(image)


def test_prefers_llama_mtmd_cli_when_available_on_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import os
    import subprocess

    import ragprep.ocr.llamacpp_cli_runtime as cli_runtime

    model_path = tmp_path / "model.gguf"
    mmproj_path = tmp_path / "mmproj.gguf"
    model_path.write_text("x", encoding="utf-8")
    mmproj_path.write_text("y", encoding="utf-8")

    monkeypatch.delenv(lightonocr.ENV_DRY_RUN, raising=False)
    monkeypatch.setenv(lightonocr.ENV_GGUF_MODEL_PATH, str(model_path))
    monkeypatch.setenv(lightonocr.ENV_GGUF_MMPROJ_PATH, str(mmproj_path))
    monkeypatch.delenv(lightonocr.ENV_LLAVA_CLI_PATH, raising=False)

    mtmd_name = "llama-mtmd-cli.exe" if os.name == "nt" else "llama-mtmd-cli"
    mtmd_cli_path = tmp_path / mtmd_name
    mtmd_cli_path.write_text("x", encoding="utf-8")

    def fake_which(name: str) -> str | None:
        if name in {"llama-mtmd-cli", "llama-mtmd-cli.exe"}:
            return str(mtmd_cli_path)
        return None

    calls: list[list[str]] = []

    def fake_run(
        argv: list[str], capture_output: bool, text: bool, check: bool
    ) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(
            args=argv, returncode=0, stdout="ASSISTANT: OK\n", stderr=""
        )

    monkeypatch.setattr(cli_runtime.shutil, "which", fake_which)
    monkeypatch.setattr(cli_runtime.subprocess, "run", fake_run)

    image = Image.new("RGB", (2, 2), color=(0, 0, 0))
    assert lightonocr.ocr_image(image) == "OK"

    assert calls
    assert calls[0][0] == str(mtmd_cli_path)


def test_gguf_env_paths_strip_quotes_and_angle_brackets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "model.gguf"
    mmproj_path = tmp_path / "mmproj.gguf"
    model_path.write_text("x", encoding="utf-8")
    mmproj_path.write_text("y", encoding="utf-8")

    monkeypatch.delenv(lightonocr.ENV_DRY_RUN, raising=False)
    monkeypatch.setenv(lightonocr.ENV_GGUF_MODEL_PATH, f'"{model_path}"')
    monkeypatch.setenv(lightonocr.ENV_GGUF_MMPROJ_PATH, f"<{mmproj_path}>")

    settings = lightonocr.get_settings()
    assert settings.gguf_model_path == str(model_path)
    assert settings.gguf_mmproj_path == str(mmproj_path)


def test_gguf_env_paths_support_file_uri(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model_path = tmp_path / "model.gguf"
    mmproj_path = tmp_path / "mmproj.gguf"
    model_path.write_text("x", encoding="utf-8")
    mmproj_path.write_text("y", encoding="utf-8")

    monkeypatch.delenv(lightonocr.ENV_DRY_RUN, raising=False)
    monkeypatch.setenv(lightonocr.ENV_GGUF_MODEL_PATH, model_path.resolve().as_uri())
    monkeypatch.setenv(lightonocr.ENV_GGUF_MMPROJ_PATH, mmproj_path.resolve().as_uri())

    settings = lightonocr.get_settings()
    assert settings.gguf_model_path == str(model_path)
    assert settings.gguf_mmproj_path == str(mmproj_path)


def test_llama_cpp_executes_with_mock_and_is_cached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import subprocess

    import ragprep.ocr.llamacpp_cli_runtime as cli_runtime

    model_path = tmp_path / "model.gguf"
    mmproj_path = tmp_path / "mmproj.gguf"
    model_path.write_text("x", encoding="utf-8")
    mmproj_path.write_text("y", encoding="utf-8")

    llava_cli_path = tmp_path / "llava-cli.exe"
    llava_cli_path.write_text("x", encoding="utf-8")

    monkeypatch.setenv(lightonocr.ENV_GGUF_MODEL_PATH, str(model_path))
    monkeypatch.setenv(lightonocr.ENV_GGUF_MMPROJ_PATH, str(mmproj_path))
    monkeypatch.setenv(lightonocr.ENV_LLAVA_CLI_PATH, str(llava_cli_path))
    monkeypatch.setenv(lightonocr.ENV_LLAMA_N_CTX, "1234")
    monkeypatch.setenv(lightonocr.ENV_LLAMA_N_THREADS, "2")
    monkeypatch.setenv(lightonocr.ENV_LLAMA_N_GPU_LAYERS, "7")
    monkeypatch.setenv(lightonocr.ENV_MAX_NEW_TOKENS, "7")

    calls: list[list[str]] = []

    def fake_run(
        argv: list[str], capture_output: bool, text: bool, check: bool
    ) -> subprocess.CompletedProcess[str]:
        assert capture_output is True
        assert text is True
        assert check is False
        calls.append(argv)

        image_flag_idx = argv.index("--image")
        image_path = Path(argv[image_flag_idx + 1])
        assert image_path.is_file()

        return subprocess.CompletedProcess(
            args=argv, returncode=0, stdout="ASSISTANT: OK\n", stderr=""
        )

    monkeypatch.setattr(cli_runtime.subprocess, "run", fake_run)

    image = Image.new("RGB", (2, 2), color=(0, 0, 0))
    assert lightonocr.ocr_image(image) == "OK"

    assert len(calls) == 1
    argv = calls[0]
    assert argv[0] == str(llava_cli_path)
    assert argv[argv.index("-m") + 1] == str(model_path)
    assert argv[argv.index("--mmproj") + 1] == str(mmproj_path)
    assert argv[argv.index("-n") + 1] == "7"
    assert argv[argv.index("-c") + 1] == "1234"
    assert argv[argv.index("-t") + 1] == "2"
    assert argv[argv.index("-ngl") + 1] == "7"


def test_llava_cli_nonzero_exit_is_actionable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import subprocess

    import ragprep.ocr.llamacpp_cli_runtime as cli_runtime

    model_path = tmp_path / "model.gguf"
    mmproj_path = tmp_path / "mmproj.gguf"
    model_path.write_text("x", encoding="utf-8")
    mmproj_path.write_text("y", encoding="utf-8")

    llava_cli_path = tmp_path / "llava-cli.exe"
    llava_cli_path.write_text("x", encoding="utf-8")

    monkeypatch.setenv(lightonocr.ENV_GGUF_MODEL_PATH, str(model_path))
    monkeypatch.setenv(lightonocr.ENV_GGUF_MMPROJ_PATH, str(mmproj_path))
    monkeypatch.setenv(lightonocr.ENV_LLAVA_CLI_PATH, str(llava_cli_path))

    def fake_run(
        argv: list[str], capture_output: bool, text: bool, check: bool
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=argv,
            returncode=2,
            stdout="WARNING: deprecated\nPlease use llama-mtmd-cli\n",
            stderr="usage: llava-cli ...",
        )

    monkeypatch.setattr(cli_runtime.subprocess, "run", fake_run)

    image = Image.new("RGB", (2, 2), color=(0, 0, 0))
    with pytest.raises(RuntimeError) as excinfo:
        lightonocr.ocr_image(image)

    message = str(excinfo.value)
    assert "exit code 2" in message
    assert "stdout:" in message
    assert "deprecated" in message
    assert "stderr:" in message
    assert "usage: llava-cli" in message


def test_build_argv_uses_short_paths_for_non_ascii_on_windows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import os

    import ragprep.ocr.llamacpp_cli_runtime as cli_runtime

    model_path = tmp_path / "モデル.gguf"
    mmproj_path = tmp_path / "mmproj-モデル.gguf"
    image_path = tmp_path / "画像.png"
    model_path.write_text("x", encoding="utf-8")
    mmproj_path.write_text("y", encoding="utf-8")
    image_path.write_text("z", encoding="utf-8")

    short_map: dict[str, str] = {}

    def fake_get_windows_short_path(path: str) -> str:
        short = f"C:\\SHORT\\{Path(path).name}"
        short_map[path] = short
        return short

    monkeypatch.setattr(cli_runtime, "_get_windows_short_path", fake_get_windows_short_path)

    settings = cli_runtime.LlamaCppCliSettings(
        llava_cli_path="llava-cli.exe",
        model_path=str(model_path),
        mmproj_path=str(mmproj_path),
        n_ctx=None,
        n_threads=None,
        n_gpu_layers=None,
    )
    argv = cli_runtime._build_argv(
        llava_cli="llava-cli.exe",
        settings=settings,
        image_path=image_path,
        prompt="p",
        max_new_tokens=1,
    )

    if os.name == "nt":
        assert argv[argv.index("-m") + 1] == short_map[str(model_path)]
        assert argv[argv.index("--mmproj") + 1] == short_map[str(mmproj_path)]
        assert argv[argv.index("--image") + 1] == short_map[str(image_path)]
    else:
        assert argv[argv.index("-m") + 1] == str(model_path)
        assert argv[argv.index("--mmproj") + 1] == str(mmproj_path)
        assert argv[argv.index("--image") + 1] == str(image_path)


def test_llava_cli_image_temp_dir_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import subprocess

    import ragprep.ocr.llamacpp_cli_runtime as cli_runtime

    override_dir = tmp_path / "imgtmp"
    monkeypatch.setenv("LIGHTONOCR_IMAGE_TMP_DIR", str(override_dir))

    llava_cli_path = tmp_path / "llava-cli.exe"
    llava_cli_path.write_text("x", encoding="utf-8")
    model_path = tmp_path / "model.gguf"
    mmproj_path = tmp_path / "mmproj.gguf"
    model_path.write_text("x", encoding="utf-8")
    mmproj_path.write_text("y", encoding="utf-8")

    captured_image_parents: list[Path] = []

    def fake_run(
        argv: list[str], capture_output: bool, text: bool, check: bool
    ) -> subprocess.CompletedProcess[str]:
        image_flag_idx = argv.index("--image")
        image_path = Path(argv[image_flag_idx + 1])
        assert image_path.is_file()
        captured_image_parents.append(image_path.parent)
        return subprocess.CompletedProcess(
            args=argv, returncode=0, stdout="ASSISTANT: OK\n", stderr=""
        )

    monkeypatch.setattr(cli_runtime.subprocess, "run", fake_run)

    settings = cli_runtime.LlamaCppCliSettings(
        llava_cli_path=str(llava_cli_path),
        model_path=str(model_path),
        mmproj_path=str(mmproj_path),
        n_ctx=None,
        n_threads=None,
        n_gpu_layers=None,
    )

    image = Image.new("RGB", (2, 2), color=(0, 0, 0))
    assert cli_runtime.ocr_image(image=image, settings=settings, max_new_tokens=7) == "OK"
    assert override_dir.is_dir()
    assert captured_image_parents == [override_dir]
