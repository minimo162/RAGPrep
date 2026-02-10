from __future__ import annotations

import pytest

from ragprep import config


def test_default_backend_is_lighton(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAGPREP_PDF_BACKEND", raising=False)
    monkeypatch.delenv("RAGPREP_LIGHTON_PROFILE", raising=False)
    settings = config.get_settings()
    assert settings.ocr_backend == "lighton-ocr"
    assert settings.pdf_backend == "lighton-ocr"
    assert settings.lighton_profile == "ocr-fastest"
    assert settings.lighton_repo_id == "noctrex/LightOnOCR-2-1B-GGUF"
    assert settings.lighton_model_file == "LightOnOCR-2-1B-IQ4_XS.gguf"
    assert settings.lighton_mmproj_file == "mmproj-BF16.gguf"


def test_pdf_backend_rejects_non_lighton(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_PDF_BACKEND", "glm-ocr")
    with pytest.raises(ValueError, match="RAGPREP_PDF_BACKEND"):
        config.get_settings()


def test_legacy_glm_or_layout_envs_do_not_affect_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RAGPREP_PDF_BACKEND", raising=False)
    monkeypatch.setenv("RAGPREP_GLM_OCR_MODE", "transformers")
    monkeypatch.setenv("RAGPREP_LAYOUT_MODE", "local-paddle")
    monkeypatch.setenv("RAGPREP_OCR_BACKEND", "glm-ocr")

    settings = config.get_settings()
    assert settings.pdf_backend == "lighton-ocr"
    assert settings.ocr_backend == "lighton-ocr"


def test_lighton_profile_rejects_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_LIGHTON_PROFILE", "invalid")
    with pytest.raises(ValueError, match="RAGPREP_LIGHTON_PROFILE"):
        config.get_settings()


def test_ocr_fastest_profile_applies_speed_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAGPREP_LIGHTON_PROFILE", "ocr-fastest")
    monkeypatch.delenv("RAGPREP_LIGHTON_PARALLEL", raising=False)
    monkeypatch.delenv("RAGPREP_LIGHTON_PAGE_CONCURRENCY", raising=False)
    monkeypatch.delenv("RAGPREP_LIGHTON_MAX_TOKENS", raising=False)
    monkeypatch.delenv("RAGPREP_LIGHTON_FAST_PASS", raising=False)
    monkeypatch.delenv("RAGPREP_LIGHTON_FAST_RENDER_DPI", raising=False)
    monkeypatch.delenv("RAGPREP_LIGHTON_FAST_RENDER_MAX_EDGE", raising=False)
    monkeypatch.delenv("RAGPREP_LIGHTON_FAST_RETRY", raising=False)
    monkeypatch.delenv("RAGPREP_LIGHTON_SECONDARY_TABLE_REPAIR", raising=False)
    monkeypatch.delenv("RAGPREP_LIGHTON_FAST_NON_TABLE_MAX_EDGE", raising=False)
    monkeypatch.delenv("RAGPREP_LIGHTON_FAST_MAX_TOKENS_TEXT", raising=False)
    monkeypatch.delenv("RAGPREP_LIGHTON_FAST_MAX_TOKENS_TABLE", raising=False)
    monkeypatch.delenv("RAGPREP_LIGHTON_FAST_POSTPROCESS_MODE", raising=False)

    settings = config.get_settings()
    assert settings.lighton_profile == "ocr-fastest"
    assert settings.lighton_parallel == 1
    assert settings.lighton_page_concurrency == 1
    assert settings.lighton_max_tokens == 1024
    assert settings.lighton_fast_pass is True
    assert settings.lighton_fast_render_dpi == 96
    assert settings.lighton_fast_render_max_edge == 640
    assert settings.lighton_fast_retry is False
    assert settings.lighton_secondary_table_repair is False
    assert settings.lighton_fast_non_table_max_edge == 520
    assert settings.lighton_fast_max_tokens_text == 512
    assert settings.lighton_fast_max_tokens_table == 1024
    assert settings.lighton_fast_postprocess_mode == "off"


def test_ocr_fastest_profile_keeps_explicit_env_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAGPREP_LIGHTON_PROFILE", "ocr-fastest")
    monkeypatch.setenv("RAGPREP_LIGHTON_PARALLEL", "3")
    monkeypatch.setenv("RAGPREP_LIGHTON_PAGE_CONCURRENCY", "4")
    monkeypatch.setenv("RAGPREP_LIGHTON_MAX_TOKENS", "2048")
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_PASS", "0")
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_RENDER_DPI", "140")
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_RENDER_MAX_EDGE", "900")
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_RETRY", "1")
    monkeypatch.setenv("RAGPREP_LIGHTON_SECONDARY_TABLE_REPAIR", "1")
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_NON_TABLE_MAX_EDGE", "700")
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_MAX_TOKENS_TEXT", "1500")
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_MAX_TOKENS_TABLE", "2500")
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_POSTPROCESS_MODE", "light")

    settings = config.get_settings()
    assert settings.lighton_profile == "ocr-fastest"
    assert settings.lighton_parallel == 3
    assert settings.lighton_page_concurrency == 4
    assert settings.lighton_max_tokens == 2048
    assert settings.lighton_fast_pass is False
    assert settings.lighton_fast_render_dpi == 140
    assert settings.lighton_fast_render_max_edge == 900
    assert settings.lighton_fast_retry is True
    assert settings.lighton_secondary_table_repair is True
    assert settings.lighton_fast_non_table_max_edge == 700
    assert settings.lighton_fast_max_tokens_text == 1500
    assert settings.lighton_fast_max_tokens_table == 2500
    assert settings.lighton_fast_postprocess_mode == "light"


def test_lighton_merge_policy_accepts_supported_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAGPREP_LIGHTON_MERGE_POLICY", "strict")
    assert config.get_settings().lighton_merge_policy == "strict"

    monkeypatch.setenv("RAGPREP_LIGHTON_MERGE_POLICY", "aggressive")
    assert config.get_settings().lighton_merge_policy == "aggressive"


def test_lighton_merge_policy_rejects_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_LIGHTON_MERGE_POLICY", "invalid")
    with pytest.raises(ValueError, match="RAGPREP_LIGHTON_MERGE_POLICY"):
        config.get_settings()


def test_lighton_n_gpu_layers_accepts_minus_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_LIGHTON_N_GPU_LAYERS", "-1")
    settings = config.get_settings()
    assert settings.lighton_n_gpu_layers == -1


def test_lighton_n_gpu_layers_rejects_less_than_minus_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAGPREP_LIGHTON_N_GPU_LAYERS", "-2")
    with pytest.raises(ValueError, match="RAGPREP_LIGHTON_N_GPU_LAYERS"):
        config.get_settings()


def test_recommended_lighton_defaults_are_applied_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RAGPREP_LIGHTON_PROFILE", raising=False)
    monkeypatch.delenv("RAGPREP_LIGHTON_START_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("RAGPREP_LIGHTON_REQUEST_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("RAGPREP_LIGHTON_PAGE_CONCURRENCY", raising=False)
    monkeypatch.delenv("RAGPREP_LIGHTON_PARALLEL", raising=False)
    monkeypatch.delenv("RAGPREP_LIGHTON_FAST_PASS", raising=False)
    monkeypatch.delenv("RAGPREP_LIGHTON_FAST_RENDER_DPI", raising=False)
    monkeypatch.delenv("RAGPREP_LIGHTON_FAST_RENDER_MAX_EDGE", raising=False)
    monkeypatch.delenv("RAGPREP_LIGHTON_FAST_RETRY", raising=False)
    monkeypatch.delenv("RAGPREP_LIGHTON_SECONDARY_TABLE_REPAIR", raising=False)
    monkeypatch.delenv("RAGPREP_LIGHTON_RETRY_RENDER_DPI", raising=False)
    monkeypatch.delenv("RAGPREP_LIGHTON_RETRY_RENDER_MAX_EDGE", raising=False)
    monkeypatch.delenv("RAGPREP_LIGHTON_RETRY_MIN_QUALITY", raising=False)
    monkeypatch.delenv("RAGPREP_LIGHTON_RETRY_QUALITY_GAP", raising=False)
    monkeypatch.delenv("RAGPREP_LIGHTON_RETRY_MIN_PYM_TEXT_LEN", raising=False)
    monkeypatch.delenv("RAGPREP_LIGHTON_FAST_NON_TABLE_MAX_EDGE", raising=False)
    monkeypatch.delenv("RAGPREP_LIGHTON_FAST_TABLE_LIKELIHOOD_THRESHOLD", raising=False)
    monkeypatch.delenv("RAGPREP_LIGHTON_FAST_MAX_TOKENS_TEXT", raising=False)
    monkeypatch.delenv("RAGPREP_LIGHTON_FAST_MAX_TOKENS_TABLE", raising=False)
    monkeypatch.delenv("RAGPREP_LIGHTON_FAST_POSTPROCESS_MODE", raising=False)

    settings = config.get_settings()
    assert settings.lighton_profile == "ocr-fastest"
    assert settings.lighton_start_timeout_seconds == 300
    assert settings.lighton_request_timeout_seconds == 600
    assert settings.lighton_page_concurrency == 1
    assert settings.lighton_parallel == 1
    assert settings.lighton_max_tokens == 1024
    assert settings.lighton_fast_pass is True
    assert settings.lighton_fast_render_dpi == 96
    assert settings.lighton_fast_render_max_edge == 640
    assert settings.lighton_fast_retry is False
    assert settings.lighton_secondary_table_repair is False
    assert settings.lighton_retry_render_dpi == 220
    assert settings.lighton_retry_render_max_edge == 1280
    assert settings.lighton_retry_min_quality == pytest.approx(0.40)
    assert settings.lighton_retry_quality_gap == pytest.approx(0.22)
    assert settings.lighton_retry_min_pym_text_len == 80
    assert settings.lighton_fast_non_table_max_edge == 520
    assert settings.lighton_fast_table_likelihood_threshold == pytest.approx(0.60)
    assert settings.lighton_fast_max_tokens_text == 512
    assert settings.lighton_fast_max_tokens_table == 1024
    assert settings.lighton_fast_postprocess_mode == "off"


def test_balanced_profile_keeps_previous_default_tuning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAGPREP_LIGHTON_PROFILE", "balanced")
    monkeypatch.delenv("RAGPREP_LIGHTON_PARALLEL", raising=False)
    monkeypatch.delenv("RAGPREP_LIGHTON_PAGE_CONCURRENCY", raising=False)
    monkeypatch.delenv("RAGPREP_LIGHTON_MAX_TOKENS", raising=False)
    monkeypatch.delenv("RAGPREP_LIGHTON_FAST_RENDER_DPI", raising=False)
    monkeypatch.delenv("RAGPREP_LIGHTON_FAST_RENDER_MAX_EDGE", raising=False)
    monkeypatch.delenv("RAGPREP_LIGHTON_FAST_NON_TABLE_MAX_EDGE", raising=False)
    monkeypatch.delenv("RAGPREP_LIGHTON_FAST_MAX_TOKENS_TEXT", raising=False)
    monkeypatch.delenv("RAGPREP_LIGHTON_FAST_MAX_TOKENS_TABLE", raising=False)
    monkeypatch.delenv("RAGPREP_LIGHTON_FAST_POSTPROCESS_MODE", raising=False)

    settings = config.get_settings()
    assert settings.lighton_profile == "balanced"
    assert settings.lighton_parallel == 2
    assert settings.lighton_page_concurrency == 2
    assert settings.lighton_max_tokens == 8192
    assert settings.lighton_fast_render_dpi == 200
    assert settings.lighton_fast_render_max_edge == 1100
    assert settings.lighton_fast_non_table_max_edge == 960
    assert settings.lighton_fast_max_tokens_text == 4096
    assert settings.lighton_fast_max_tokens_table == 8192
    assert settings.lighton_fast_postprocess_mode == "light"


def test_lighton_fast_retry_env_overrides_are_applied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_PASS", "0")
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_RENDER_DPI", "195")
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_RENDER_MAX_EDGE", "1000")
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_RETRY", "1")
    monkeypatch.setenv("RAGPREP_LIGHTON_SECONDARY_TABLE_REPAIR", "1")
    monkeypatch.setenv("RAGPREP_LIGHTON_RETRY_RENDER_DPI", "240")
    monkeypatch.setenv("RAGPREP_LIGHTON_RETRY_RENDER_MAX_EDGE", "1400")
    monkeypatch.setenv("RAGPREP_LIGHTON_RETRY_MIN_QUALITY", "0.55")
    monkeypatch.setenv("RAGPREP_LIGHTON_RETRY_QUALITY_GAP", "0.18")
    monkeypatch.setenv("RAGPREP_LIGHTON_RETRY_MIN_PYM_TEXT_LEN", "120")
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_NON_TABLE_MAX_EDGE", "920")
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_TABLE_LIKELIHOOD_THRESHOLD", "0.72")
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_MAX_TOKENS_TEXT", "3500")
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_MAX_TOKENS_TABLE", "7000")
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_POSTPROCESS_MODE", "off")

    settings = config.get_settings()
    assert settings.lighton_fast_pass is False
    assert settings.lighton_fast_render_dpi == 195
    assert settings.lighton_fast_render_max_edge == 1000
    assert settings.lighton_fast_retry is True
    assert settings.lighton_secondary_table_repair is True
    assert settings.lighton_retry_render_dpi == 240
    assert settings.lighton_retry_render_max_edge == 1400
    assert settings.lighton_retry_min_quality == pytest.approx(0.55)
    assert settings.lighton_retry_quality_gap == pytest.approx(0.18)
    assert settings.lighton_retry_min_pym_text_len == 120
    assert settings.lighton_fast_non_table_max_edge == 920
    assert settings.lighton_fast_table_likelihood_threshold == pytest.approx(0.72)
    assert settings.lighton_fast_max_tokens_text == 3500
    assert settings.lighton_fast_max_tokens_table == 7000
    assert settings.lighton_fast_postprocess_mode == "off"


def test_lighton_fast_postprocess_mode_rejects_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAGPREP_LIGHTON_FAST_POSTPROCESS_MODE", "invalid")
    with pytest.raises(ValueError, match="RAGPREP_LIGHTON_FAST_POSTPROCESS_MODE"):
        config.get_settings()
