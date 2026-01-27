from __future__ import annotations

from ragprep.web.app import _download_filename_from_upload


def test_download_filename_from_upload_uses_pdf_stem() -> None:
    assert _download_filename_from_upload("test.pdf", suffix="md") == "test.md"
    assert _download_filename_from_upload(r"C:\tmp\foo.pdf", suffix="md") == "foo.md"
    assert _download_filename_from_upload("dir/subdir/bar.pdf", suffix="md") == "bar.md"
    assert _download_filename_from_upload("dir/subdir/bar.pdf", suffix="md") == "bar.md"


def test_download_filename_from_upload_strips_controls_and_quotes() -> None:
    result = _download_filename_from_upload('evil.pdf"\r\nX: y', suffix="md")
    assert result == "evil.md"
    assert "\r" not in result
    assert "\n" not in result
    assert '"' not in result

