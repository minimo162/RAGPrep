from __future__ import annotations

from pathlib import Path


def _read_build_standalone_ps1() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "build-standalone.ps1"
    return script_path.read_text(encoding="utf-8", errors="replace")


def _read_verify_standalone_ps1() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "verify-standalone.ps1"
    return script_path.read_text(encoding="utf-8", errors="replace")


def test_run_cmd_template_does_not_mkdir_empty_hf_home() -> None:
    content = _read_build_standalone_ps1()
    assert 'if not exist "%ROOT%data\\hf" mkdir "%ROOT%data\\hf"' in content
    assert 'if not exist "%HF_HOME%" mkdir "%HF_HOME%"' not in content
    assert 'if "%RAGPREP_PDF_BACKEND%"=="" (' in content
    assert "set RAGPREP_PDF_BACKEND=glm-ocr" in content
    assert "set RAGPREP_PDF_BACKEND=lightonocr" in content
    assert "Invoke-WebRequest -UseBasicParsing -TimeoutSec 2" in content
    assert 'if /I "%RAGPREP_PDF_BACKEND%"=="lightonocr" (' in content
    expected = (
        '"%ROOT%python\\python.exe" -m ragprep.desktop --host %BIND_HOST% '
        "--port %PORT%"
    )
    assert expected in content


def test_run_ps1_template_avoids_host_automatic_variable() -> None:
    content = _read_build_standalone_ps1()
    assert '[Alias("Host")]' in content
    assert '[string]`$BindHost = "127.0.0.1",' in content
    assert "--host `$BindHost" in content
    assert "& `$pythonExe -m ragprep.desktop --host `$BindHost --port `$Port" in content
    assert '[string]`$Host = "127.0.0.1",' not in content
    assert "Invoke-WebRequest -UseBasicParsing -TimeoutSec 2" in content
    assert "`$env:RAGPREP_PDF_BACKEND = \"glm-ocr\"" in content
    assert "`$env:RAGPREP_PDF_BACKEND = \"lightonocr\"" in content
    assert "if (`$env:RAGPREP_PDF_BACKEND -eq \"lightonocr\") {" in content


def test_build_standalone_prefetch_has_direct_download_and_artifact_checks() -> None:
    content = _read_build_standalone_ps1()
    assert "falling back to direct download" in content
    assert "resolve/main/{filename}?download=1" in content
    assert "GGUF artifact missing after prefetch" in content
    assert "GGUF artifact is empty after prefetch" in content


def test_run_scripts_have_gguf_preflight_checks() -> None:
    content = _read_build_standalone_ps1()
    assert "Missing GGUF artifact" in content
    assert "Missing GGUF model" in content
    assert "Missing GGUF mmproj" in content


def test_build_standalone_calls_verify_script() -> None:
    content = _read_build_standalone_ps1()
    assert 'verify standalone output' in content
    assert 'scripts/verify-standalone.ps1' in content
    assert 'Assert-LastExitCode "verify standalone"' in content


def test_build_standalone_bundles_vulkan_and_avx2() -> None:
    content = _read_build_standalone_ps1()
    assert 'Name = "avx2"' in content
    assert 'Name = "vulkan"' in content
    assert "bin-win-cpu-x64.zip" in content
    assert "bin-win-vulkan-x64.zip" in content


def test_verify_standalone_checks_required_artifacts() -> None:
    content = _read_verify_standalone_ps1()
    assert "python/python.exe" in content
    assert '"app"' in content
    assert '"site-packages"' in content
    assert "bin/llama.cpp" in content
    assert '"avx2"' in content
    assert '"vulkan"' in content
    assert "llama-mtmd-cli.exe" in content
    assert "data/models/lightonocr-gguf" in content
    assert "LightOnOCR-2-1B-Q6_K.gguf" in content
    assert "mmproj-BF16.gguf" in content
