"""Tests for configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from notulen.config import AudioConfig, load_config


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate each test from variables set on the host machine."""
    for name in ("HF_TOKEN", "LLM_API_KEY", "OLLAMA_HOST"):
        monkeypatch.delenv(name, raising=False)


def test_default_config_loads(tmp_path: Path) -> None:
    config = load_config(env_path=tmp_path / "missing.env")

    assert config.audio.sample_rate == 16000
    assert config.asr.language == "id"
    assert config.secrets.hf_token is None


def test_environment_supplies_secrets_and_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_example")
    monkeypatch.setenv("OLLAMA_HOST", "http://10.0.0.5:11434")

    config = load_config(env_path=tmp_path / "missing.env")

    assert config.secrets.hf_token is not None
    assert config.secrets.hf_token.get_secret_value() == "hf_example"
    assert "hf_example" not in repr(config)
    assert config.llm.ollama.host == "http://10.0.0.5:11434"


def test_overlap_must_be_smaller_than_chunk() -> None:
    with pytest.raises(ValidationError):
        AudioConfig(sample_rate=16000, channels=1, max_chunk_seconds=10, chunk_overlap_seconds=10)
