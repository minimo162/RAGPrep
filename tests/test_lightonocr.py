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
