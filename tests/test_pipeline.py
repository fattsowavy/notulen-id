"""Tests for pipeline orchestration, with every model replaced by a fake."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from notulen.audio.preprocess import AudioChunk, PreprocessedAudio
from notulen.config import AppConfig, PathsConfig, RuntimeConfig, load_config
from notulen.pipeline import NotulenPipeline
from notulen.schemas import Segment, SpeakerTurn, Summary, Transcript, Word
from notulen.summarize.llm_client import LlmError

events: list[str] = []
CLIENT_CALLS: list[tuple[tuple[Any, ...], dict[str, Any]]] = []


def word(text: str, start: float, end: float) -> Word:
    return Word(text=text, start=start, end=end, probability=0.9)


class FakeTranscriber:
    def __init__(self, *args: Any) -> None:
        events.append("asr:init")

    def transcribe(self, audio: PreprocessedAudio) -> Transcript:
        events.append("asr:run")
        words = (word("selamat", 0.0, 0.5), word("pagi", 0.5, 1.0), word("ya", 2.0, 2.5))
        return Transcript(
            source=audio.source.name,
            language="id",
            model="medium",
            duration=audio.duration,
            segments=(Segment(start=0.0, end=2.5, text="selamat pagi ya", words=words),),
        )

    def close(self) -> None:
        events.append("asr:close")


class FakeDiarizer:
    def __init__(self, *args: Any) -> None:
        events.append("diar:init")

    def diarize(self, wav_path: Path, num_speakers: int | None = None) -> list[SpeakerTurn]:
        events.append(f"diar:run:{num_speakers}")
        return [
            SpeakerTurn(start=0.0, end=1.5, speaker="SPEAKER_00"),
            SpeakerTurn(start=1.5, end=3.0, speaker="SPEAKER_01"),
        ]

    def close(self) -> None:
        events.append("diar:close")


class FakeClient:
    def __init__(self, fail: bool) -> None:
        self.fail = fail
        self.prompts: list[str] = []

    def __enter__(self) -> FakeClient:
        return self

    def __exit__(self, *args: Any) -> None:
        events.append("llm:close")

    def complete(self, prompt: str, json_schema: Any = None) -> str:
        self.prompts.append(prompt)
        if self.fail:
            raise LlmError("Cannot reach Ollama")
        if '"tindak_lanjut"' in prompt:
            return json.dumps(
                {"tindak_lanjut": [{"tugas": "Menyapa", "kutipan_sumber": "selamat pagi"}]}
            )
        return json.dumps({"topik": ["Sapaan pagi"], "keputusan": [], "poin_penting": []})


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppConfig:
    for name in ("HF_TOKEN", "LLM_API_KEY", "OLLAMA_HOST"):
        monkeypatch.delenv(name, raising=False)
    base = load_config(env_path=tmp_path / "missing.env")
    return base.model_copy(
        update={"paths": PathsConfig(output_dir=tmp_path / "out", temp_dir=tmp_path / "tmp")}
    )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    events.clear()
    fake = FakeClient(fail=False)

    def fake_preprocess(path: Path, audio: Any, vad: Any, work_dir: Path) -> PreprocessedAudio:
        work_dir.mkdir(parents=True)
        wav = work_dir / "audio.wav"
        wav.write_bytes(b"")
        return PreprocessedAudio(
            source=path.resolve(),
            wav_path=wav,
            sample_rate=16000,
            duration=3.0,
            speech_regions=(),
            chunks=(AudioChunk(index=0, path=wav, start=0.0, end=3.0),),
        )

    monkeypatch.setattr("notulen.pipeline.preprocess_audio", fake_preprocess)
    monkeypatch.setattr("notulen.pipeline.Transcriber", FakeTranscriber)
    monkeypatch.setattr("notulen.pipeline.Diarizer", FakeDiarizer)

    def factory(*args: Any, **kwargs: Any) -> Any:
        CLIENT_CALLS.append((args, kwargs))
        return fake

    CLIENT_CALLS.clear()
    monkeypatch.setattr("notulen.pipeline.create_llm_client", factory)
    return fake


def test_summarize_can_use_another_llm_for_one_call(config: AppConfig, client: FakeClient) -> None:
    other = config.llm.model_copy(update={"backend": "openai_compatible", "model": "llama"})

    with NotulenPipeline(config) as pipeline:
        result = pipeline.run(Path("rapat.mp3"), summarize=False)
        pipeline.summarize(result, llm=other, api_key="sk-user")

    (args, kwargs) = CLIENT_CALLS[0]
    assert args[0] is other
    assert kwargs == {"user_api_key": "sk-user", "configured": config.llm}
    assert result.summary is not None


def test_run_writes_every_output(config: AppConfig, client: FakeClient, tmp_path: Path) -> None:
    stages: list[str] = []

    with NotulenPipeline(config) as pipeline:
        result = pipeline.run(Path("rapat.mp3"), num_speakers=2, on_stage=stages.append)

    output = tmp_path / "out" / "rapat"
    assert stages == ["preprocess", "asr", "diarization", "alignment", "summary"]
    assert set(result.files) == {
        "transcript_json",
        "transcript_md",
        "rttm",
        "summary",
        "summary_json",
        "timings",
    }
    assert all(path.parent == output and path.is_file() for path in result.files.values())

    markdown = (output / "transcript.md").read_text(encoding="utf-8")
    assert "**Pembicara 1** [00:00:00]: selamat pagi" in markdown
    assert "**Pembicara 2** [00:00:02]: ya" in markdown
    assert Transcript.load_json(output / "transcript.json") == result.transcript
    assert "[00:00:00] Pembicara 1: selamat pagi" in client.prompts[0]
    assert "Sapaan pagi" in (output / "summary.md").read_text(encoding="utf-8")
    assert result.summary is not None
    assert result.summary.tindak_lanjut[0].kutipan_sumber == "selamat pagi"
    assert Summary.model_validate_json((output / "summary.json").read_text("utf-8")) == (
        result.summary
    )
    assert "diar:run:2" in events


def test_timings_cover_each_stage(config: AppConfig, client: FakeClient, tmp_path: Path) -> None:
    result = NotulenPipeline(config).run(Path("rapat.mp3"))

    timings = json.loads(result.files["timings"].read_text(encoding="utf-8"))
    assert set(timings["stages_seconds"]) == {
        "preprocess",
        "asr",
        "diarization",
        "alignment",
        "summary",
    }
    assert timings["audio_seconds"] == 3.0
    assert timings["total_seconds"] == result.timings["total"]


def test_models_are_released_between_stages(config: AppConfig, client: FakeClient) -> None:
    NotulenPipeline(config).run(Path("rapat.mp3"))

    assert events.index("asr:close") < events.index("diar:init")
    assert "diar:close" in events


def test_models_stay_loaded_when_release_is_disabled(config: AppConfig, client: FakeClient) -> None:
    config = config.model_copy(
        update={"runtime": RuntimeConfig(release_models_between_stages=False)}
    )

    with NotulenPipeline(config) as pipeline:
        pipeline.run(Path("rapat.mp3"))
        assert "asr:close" not in events
        pipeline.run(Path("rapat.mp3"))

    assert events.count("asr:init") == 1
    assert events.count("asr:close") == 1


def test_failed_summary_keeps_the_transcript(
    config: AppConfig, client: FakeClient, tmp_path: Path
) -> None:
    client.fail = True

    result = NotulenPipeline(config).run(Path("rapat.mp3"))

    assert result.summary is None
    assert result.summary_error == "Cannot reach Ollama"
    assert "summary" not in result.files
    assert result.files["transcript_md"].is_file()


def test_skipping_diarization_and_summary(config: AppConfig, client: FakeClient) -> None:
    result = NotulenPipeline(config).run(Path("rapat.mp3"), diarize=False, summarize=False)

    assert "diar:init" not in events
    assert client.prompts == []
    assert "rttm" not in result.files
    assert {segment.speaker for segment in result.transcript.segments} == {None}


def test_temporary_files_are_removed(config: AppConfig, client: FakeClient, tmp_path: Path) -> None:
    NotulenPipeline(config).run(Path("rapat.mp3"))

    assert list((tmp_path / "tmp").iterdir()) == []


def test_relative_config_paths_are_anchored_to_project(tmp_path: Path) -> None:
    config = load_config(env_path=tmp_path / "missing.env")

    assert config.paths.output_dir.is_absolute()
    assert config.paths.output_dir.name == "outputs"
