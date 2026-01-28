from __future__ import annotations

from pathlib import Path


def _read_template(name: str) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    template_path = repo_root / "ragprep" / "web" / "templates" / name
    return template_path.read_text(encoding="utf-8")


def test_partial_output_has_no_streaming_output_marker() -> None:
    content = _read_template("_job_status.html")
    assert "streaming-output" not in content


def test_index_template_has_no_streaming_scroll_style() -> None:
    content = _read_template("index.html")
    assert "max-height: 320px" not in content
    assert "pre.streaming-output" not in content
    assert "streaming-output" not in content
