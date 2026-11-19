"""Load pipeline settings from YAML and secrets from the environment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"


class StrictModel(BaseModel):
    """Immutable base model that rejects unknown keys, catching config typos."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsConfig(StrictModel):
    """Output locations."""

    output_dir: Path
    temp_dir: Path


class LoggingConfig(StrictModel):
    """Logging settings."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class AudioConfig(StrictModel):
    """Audio normalization and chunking settings."""

    sample_rate: int = Field(gt=0)
    channels: int = Field(ge=1, le=2)
    max_chunk_seconds: float = Field(gt=0)
    chunk_overlap_seconds: float = Field(ge=0)

    @model_validator(mode="after")
    def check_overlap(self) -> AudioConfig:
        """Ensure each chunk advances past the previous one."""
        if self.chunk_overlap_seconds >= self.max_chunk_seconds:
            raise ValueError("chunk_overlap_seconds must be smaller than max_chunk_seconds")
        return self


class VadConfig(StrictModel):
    """Silero VAD settings."""

    enabled: bool = True
    threshold: float = Field(gt=0, lt=1)
    min_speech_duration_ms: int = Field(ge=0)
    min_silence_duration_ms: int = Field(ge=0)
    speech_pad_ms: int = Field(ge=0)


class AsrConfig(StrictModel):
    """faster-whisper settings."""

    model: str
    device: Literal["cuda", "cpu", "auto"]
    compute_type: str
    language: str
    beam_size: int = Field(ge=1)
    word_timestamps: bool
    vad_filter: bool
    condition_on_previous_text: bool


class DiarizationConfig(StrictModel):
    """pyannote settings."""

    pipeline: str
    device: Literal["cuda", "cpu"]
    num_speakers: int | None = Field(default=None, ge=1)
    min_speakers: int | None = Field(default=None, ge=1)
    max_speakers: int | None = Field(default=None, ge=1)


class MergeConfig(StrictModel):
    """Word-to-speaker alignment settings."""

    max_gap_seconds: float = Field(ge=0)
    speaker_label_format: str

    @model_validator(mode="after")
    def check_label_format(self) -> MergeConfig:
        """Ensure each speaker receives a distinct label."""
        if "{index}" not in self.speaker_label_format:
            raise ValueError("speaker_label_format must contain {index}")
        return self


class OllamaConfig(StrictModel):
    """Ollama backend settings."""

    host: str
    num_ctx: int = Field(gt=0)


class OpenAICompatibleConfig(StrictModel):
    """OpenAI-compatible backend settings."""

    base_url: str


class LlmConfig(StrictModel):
    """LLM backend settings."""

    backend: Literal["ollama", "openai_compatible"]
    model: str
    temperature: float = Field(ge=0)
    timeout_seconds: float = Field(gt=0)
    max_retries: int = Field(ge=0)
    ollama: OllamaConfig
    openai_compatible: OpenAICompatibleConfig


class SummarizeConfig(StrictModel):
    """Map-reduce summarization settings."""

    chunk_max_chars: int = Field(gt=0)
    chunk_overlap_chars: int = Field(ge=0)
    require_action_item_quote: bool


class RuntimeConfig(StrictModel):
    """Resource management settings."""

    release_models_between_stages: bool


class Secrets(StrictModel):
    """Credentials read from the environment, never from YAML."""

    hf_token: SecretStr | None = None
    llm_api_key: SecretStr | None = None


class AppConfig(StrictModel):
    """Complete application configuration."""

    paths: PathsConfig
    logging: LoggingConfig
    audio: AudioConfig
    vad: VadConfig
    asr: AsrConfig
    diarization: DiarizationConfig
    merge: MergeConfig
    llm: LlmConfig
    summarize: SummarizeConfig
    runtime: RuntimeConfig
    secrets: Secrets = Field(default_factory=Secrets)


def _read_secret(name: str) -> SecretStr | None:
    """Return an environment variable as a secret, treating blank values as unset."""
    value = os.getenv(name, "").strip()
    return SecretStr(value) if value else None


def load_config(
    config_path: Path = DEFAULT_CONFIG_PATH, env_path: Path = DEFAULT_ENV_PATH
) -> AppConfig:
    """Load the YAML config, then apply secrets and overrides from the environment."""
    load_dotenv(env_path, override=False)

    raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")

    ollama_host = os.getenv("OLLAMA_HOST", "").strip()
    if ollama_host:
        raw.setdefault("llm", {}).setdefault("ollama", {})["host"] = ollama_host

    raw["secrets"] = Secrets(
        hf_token=_read_secret("HF_TOKEN"), llm_api_key=_read_secret("LLM_API_KEY")
    )
    return AppConfig.model_validate(raw)
