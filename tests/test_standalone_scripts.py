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
    assert "set RAGPREP_PDF_BACKEND=lighton-ocr" in content
    assert "llama-server not found" in content
    assert "RAGPREP_LIGHTON" not in content
    assert "/v1/models" not in content
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
    assert "`$env:RAGPREP_PDF_BACKEND = \"lighton-ocr\"" in content
    assert "RAGPREP_LLAMA_SERVER_PATH" in content
    assert "llama-server not found" in content
    assert "RAGPREP_LIGHTON" not in content
    assert "/v1/models" not in content


def test_build_standalone_calls_verify_script() -> None:
    content = _read_build_standalone_ps1()
    assert 'verify standalone output' in content
    assert 'scripts/verify-standalone.ps1' in content
    assert 'Assert-LastExitCode "verify standalone"' in content


def test_verify_standalone_checks_required_artifacts() -> None:
    content = _read_verify_standalone_ps1()
    assert "python/python.exe" in content
    assert '"app"' in content
    assert '"site-packages"' in content
    assert '"run.ps1"' in content
    assert '"run.cmd"' in content
    assert "start-lighton-ocr" not in content
