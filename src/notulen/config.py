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
    # Retry on CPU when the GPU cannot run Whisper (missing libraries or kernels).
    cpu_fallback: bool = False
    cpu_compute_type: str = "int8"


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
    # How long Ollama keeps the model in memory after a reply, e.g. "5m" or "0s".
    keep_alive: str = "5m"


class OpenAICompatibleConfig(StrictModel):
    """OpenAI-compatible backend settings."""

    base_url: str
    # Authenticate with HF_TOKEN when LLM_API_KEY is not set, e.g. for the Hugging Face router.
    use_hf_token: bool = False


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
    quote_match_threshold: float = Field(default=0.8, gt=0, le=1)
    json_attempts: int = Field(default=2, ge=1)

    @model_validator(mode="after")
    def check_overlap(self) -> SummarizeConfig:
        """Ensure each chunk adds new text."""
        if self.chunk_overlap_chars >= self.chunk_max_chars:
            raise ValueError("chunk_overlap_chars must be smaller than chunk_max_chars")
        return self


class RuntimeConfig(StrictModel):
    """Resource management settings."""

    release_models_between_stages: bool


class LlmProviderConfig(StrictModel):
    """An LLM provider the web demo offers, with its endpoint and suggested models."""

    label: str
    backend: Literal["ollama", "openai_compatible"]
    # Chat completions endpoint for openai_compatible; empty lets the user type one.
    base_url: str = ""
    # Suggestions shown in the model list; users may type any other model name.
    models: tuple[str, ...] = ()
    # When false, an empty API key is allowed (e.g. a local OpenAI-compatible server).
    requires_key: bool = True
    # Where users can create an API key for this provider.
    key_url: str | None = None
    enabled: bool = True


class DemoConfig(StrictModel):
    """Web demo settings."""

    # Longer uploads are rejected; null means no limit.
    max_audio_seconds: float | None = Field(default=None, gt=0)
    # Provider selected when the page opens; null picks the first enabled one.
    default_llm_provider: str | None = None
    llm_providers: dict[str, LlmProviderConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_default_provider(self) -> DemoConfig:
        provider = self.llm_providers.get(self.default_llm_provider or "")
        if self.default_llm_provider is not None and (provider is None or not provider.enabled):
            raise ValueError(
                f"demo.default_llm_provider {self.default_llm_provider!r} is not an enabled "
                "entry of demo.llm_providers"
            )
        return self


class DatasetBuildConfig(StrictModel):
    """How scripts/build_test_set.py samples public datasets."""

    seed: int
    fleurs_utterances: int = Field(ge=1)
    meetings: int = Field(ge=1)
    turns_per_meeting: int = Field(ge=2)
    speakers_per_meeting: tuple[int, int]
    min_gap_seconds: float = Field(ge=0)
    max_gap_seconds: float = Field(ge=0)

    @model_validator(mode="after")
    def check_ranges(self) -> DatasetBuildConfig:
        """Ensure the ranges are ordered and meetings have at least two speakers."""
        low, high = self.speakers_per_meeting
        if not 2 <= low <= high:
            raise ValueError("speakers_per_meeting must be an ordered range starting at 2 or more")
        if self.min_gap_seconds > self.max_gap_seconds:
            raise ValueError("min_gap_seconds must not exceed max_gap_seconds")
        return self


class EvaluationConfig(StrictModel):
    """Evaluation settings."""

    test_set_dir: Path
    output_dir: Path
    results_path: Path
    wer_exclude_subsets: tuple[str, ...] = ()
    der_collar_seconds: float = Field(ge=0)
    bertscore_enabled: bool
    bertscore_model: str
    action_match_threshold: float = Field(gt=0, le=1)
    build: DatasetBuildConfig


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
    evaluation: EvaluationConfig
    demo: DemoConfig = Field(default_factory=DemoConfig)
    secrets: Secrets = Field(default_factory=Secrets)


def _read_secret(name: str) -> SecretStr | None:
    """Return an environment variable as a secret, treating blank values as unset."""
    value = os.getenv(name, "").strip()
    return SecretStr(value) if value else None


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge override into base recursively; override wins for non-mapping values."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Read a config file, applying "extends: other.yaml" relative to its folder."""
    path = path.resolve()
    if path in seen:
        raise ValueError(f"Config files extend each other in a loop: {path}")
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")

    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    return _deep_merge(_read_yaml(path.parent / str(parent), (*seen, path)), raw)


def resolve_config_path(config_path: Path | None = None) -> Path:
    """Use the given path, else NOTULEN_CONFIG from the environment, else the default."""
    if config_path is not None:
        return config_path
    from_env = os.getenv("NOTULEN_CONFIG", "").strip()
    if not from_env:
        return DEFAULT_CONFIG_PATH
    path = Path(from_env)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(config_path: Path | None = None, env_path: Path = DEFAULT_ENV_PATH) -> AppConfig:
    """Load the YAML config, then apply secrets and overrides from the environment."""
    load_dotenv(env_path, override=False)
    raw = _read_yaml(resolve_config_path(config_path))

    ollama_host = os.getenv("OLLAMA_HOST", "").strip()
    if ollama_host:
        raw.setdefault("llm", {}).setdefault("ollama", {})["host"] = ollama_host

    # Relative paths are anchored to the project root so scripts work from any folder.
    for section in ("paths", "evaluation"):
        values = raw.get(section)
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            is_path = key.endswith(("_dir", "_path"))
            if is_path and isinstance(value, str) and not Path(value).is_absolute():
                values[key] = str(PROJECT_ROOT / value)

    raw["secrets"] = Secrets(
        hf_token=_read_secret("HF_TOKEN"), llm_api_key=_read_secret("LLM_API_KEY")
    )
    return AppConfig.model_validate(raw)
