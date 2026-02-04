from __future__ import annotations

from ragprep.web.app import _download_filename_from_upload


def test_download_filename_from_upload_uses_pdf_stem() -> None:
    assert _download_filename_from_upload("test.pdf", suffix="html") == "test.html"
    assert _download_filename_from_upload(r"C:\tmp\foo.pdf", suffix="html") == "foo.html"
    assert _download_filename_from_upload("dir/subdir/bar.pdf", suffix="html") == "bar.html"
    assert _download_filename_from_upload("dir/subdir/bar.pdf", suffix="html") == "bar.html"


def test_download_filename_from_upload_strips_controls_and_quotes() -> None:
    result = _download_filename_from_upload('evil.pdf"\r\nX: y', suffix="html")
    assert result == "evil.html"
    assert "\r" not in result
    assert "\n" not in result
    assert '"' not in result

