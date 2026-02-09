from __future__ import annotations

import pytest

from ragprep import config


def test_default_backend_is_lighton(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAGPREP_PDF_BACKEND", raising=False)
    settings = config.get_settings()
    assert settings.ocr_backend == "lighton-ocr"
    assert settings.pdf_backend == "lighton-ocr"
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
