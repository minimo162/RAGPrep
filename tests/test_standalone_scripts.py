from __future__ import annotations

from pathlib import Path


def _read_build_standalone_ps1() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "build-standalone.ps1"
    return script_path.read_text(encoding="utf-8", errors="replace")


def test_run_cmd_template_does_not_mkdir_empty_hf_home() -> None:
    content = _read_build_standalone_ps1()
    assert 'if not exist "%ROOT%data\\hf" mkdir "%ROOT%data\\hf"' in content
    assert 'if not exist "%HF_HOME%" mkdir "%HF_HOME%"' not in content
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
