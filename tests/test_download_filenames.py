from __future__ import annotations

from ragprep.web.app import _build_download_content_disposition, _download_filename_from_upload


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


def test_build_download_content_disposition_uses_plain_filename_for_ascii() -> None:
    content_disposition = _build_download_content_disposition("test.pdf", suffix="html")
    assert content_disposition == 'attachment; filename="test.html"'


def test_build_download_content_disposition_uses_dual_filename_for_unicode() -> None:
    content_disposition = _build_download_content_disposition("report【1】.pdf", suffix="html")
    assert 'filename="report_1.html"' in content_disposition
    assert "filename*=UTF-8''report%E3%80%901%E3%80%91.html" in content_disposition
    assert content_disposition.isascii()

