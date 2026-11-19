"""Tests for the diarizer, using a fake pyannote pipeline."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from pydantic import SecretStr

from notulen.config import DiarizationConfig
from notulen.diarization.diarizer import (
    FORCE_NO_WEIGHTS_ONLY,
    DiarizationError,
    Diarizer,
    annotation_to_turns,
    read_rttm,
    trusted_checkpoint_loading,
    write_rttm,
)
from notulen.schemas import SpeakerTurn

CONFIG = DiarizationConfig(pipeline="pyannote/speaker-diarization-3.1", device="cpu")
TOKEN = SecretStr("hf_example")


class FakeAnnotation:
    """Mimic the part of pyannote's Annotation used by the diarizer."""

    def __init__(self, tracks: list[tuple[float, float, str]]) -> None:
        self.tracks = tracks

    def itertracks(self, yield_label: bool = False) -> Iterator[tuple[Any, str, str]]:
        assert yield_label
        for index, (start, end, label) in enumerate(self.tracks):
            yield SimpleNamespace(start=start, end=end), str(index), label


class FakePipeline:
    """Record how the pipeline is called."""

    def __init__(self) -> None:
        self.device: Any = None
        self.calls: list[tuple[dict[str, Any], dict[str, Any]]] = []

    def to(self, device: Any) -> FakePipeline:
        self.device = device
        return self

    def __call__(self, file: dict[str, Any], **kwargs: Any) -> FakeAnnotation:
        self.calls.append((file, kwargs))
        return FakeAnnotation([(0.0, 1.0, "SPEAKER_00")])


@pytest.fixture
def fake_pipeline(monkeypatch: pytest.MonkeyPatch) -> FakePipeline:
    pytest.importorskip("pyannote.audio")
    pipeline = FakePipeline()
    monkeypatch.setattr("pyannote.audio.Pipeline.from_pretrained", lambda *args, **kwargs: pipeline)
    return pipeline


@pytest.fixture
def wav_path(tmp_path: Path) -> Path:
    path = tmp_path / "audio.wav"
    sf.write(path, np.zeros(16000, dtype=np.float32), 16000, subtype="PCM_16")
    return path


def test_annotation_to_turns_numbers_speakers_by_first_appearance() -> None:
    annotation = FakeAnnotation(
        [(5.0, 6.0, "SPEAKER_00"), (0.0, 2.0, "SPEAKER_03"), (2.5, 4.0, "SPEAKER_00")]
    )

    turns = annotation_to_turns(annotation)

    assert [turn.speaker for turn in turns] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_01"]
    assert [turn.start for turn in turns] == [0.0, 2.5, 5.0]


def test_annotation_to_turns_skips_empty_tracks() -> None:
    assert annotation_to_turns(FakeAnnotation([(1.0, 1.0, "SPEAKER_00")])) == []


def test_rttm_round_trip(tmp_path: Path) -> None:
    turns = [
        SpeakerTurn(start=0.5, end=2.25, speaker="SPEAKER_00"),
        SpeakerTurn(start=2.25, end=4.0, speaker="SPEAKER_01"),
    ]

    path = write_rttm(turns, tmp_path / "out" / "audio.rttm", "rapat mingguan")

    assert path.read_text(encoding="utf-8").splitlines()[0] == (
        "SPEAKER rapat_mingguan 1 0.500 1.750 <NA> <NA> SPEAKER_00 <NA> <NA>"
    )
    assert read_rttm(path) == turns


def test_read_rttm_ignores_other_record_types(tmp_path: Path) -> None:
    path = tmp_path / "audio.rttm"
    path.write_text(
        "\nSPKR-INFO a 1 <NA> <NA> <NA> unknown S1 <NA> <NA>\n"
        "SPEAKER a 1 1.000 2.000 <NA> <NA> S1 <NA> <NA>\n",
        encoding="utf-8",
    )

    assert read_rttm(path) == [SpeakerTurn(start=1.0, end=3.0, speaker="S1")]


def test_read_rttm_rejects_malformed_line(tmp_path: Path) -> None:
    path = tmp_path / "audio.rttm"
    path.write_text("SPEAKER a 1 1.000\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        read_rttm(path)


def test_load_requires_token() -> None:
    with pytest.raises(DiarizationError, match="HF_TOKEN"):
        Diarizer(CONFIG, None).load()


def test_load_explains_gated_model(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("pyannote.audio")
    monkeypatch.setattr("pyannote.audio.Pipeline.from_pretrained", lambda *args, **kwargs: None)

    with pytest.raises(DiarizationError, match="terms"):
        Diarizer(CONFIG, TOKEN).load()


def test_checkpoint_override_is_scoped_to_pipeline_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pyannote.audio")
    monkeypatch.delenv(FORCE_NO_WEIGHTS_ONLY, raising=False)
    seen: list[str | None] = []

    def from_pretrained(*args: Any, **kwargs: Any) -> FakePipeline:
        seen.append(os.environ.get(FORCE_NO_WEIGHTS_ONLY))
        return FakePipeline()

    monkeypatch.setattr("pyannote.audio.Pipeline.from_pretrained", from_pretrained)

    Diarizer(CONFIG, TOKEN).load()

    assert seen == ["1"]
    assert FORCE_NO_WEIGHTS_ONLY not in os.environ


def test_checkpoint_override_restores_previous_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(FORCE_NO_WEIGHTS_ONLY, "0")

    with pytest.raises(RuntimeError), trusted_checkpoint_loading():
        assert os.environ[FORCE_NO_WEIGHTS_ONLY] == "1"
        raise RuntimeError

    assert os.environ[FORCE_NO_WEIGHTS_ONLY] == "0"


def test_diarize_passes_audio_in_memory(fake_pipeline: FakePipeline, wav_path: Path) -> None:
    with Diarizer(CONFIG, TOKEN) as diarizer:
        turns = diarizer.diarize(wav_path)

    file, options = fake_pipeline.calls[0]
    assert file["sample_rate"] == 16000
    assert tuple(file["waveform"].shape) == (1, 16000)
    assert options == {}
    assert turns == [SpeakerTurn(start=0.0, end=1.0, speaker="SPEAKER_00")]


def test_exact_speaker_count_takes_priority(fake_pipeline: FakePipeline, wav_path: Path) -> None:
    config = CONFIG.model_copy(update={"min_speakers": 2, "max_speakers": 5})

    Diarizer(config, TOKEN).diarize(wav_path, num_speakers=3)

    assert fake_pipeline.calls[0][1] == {"num_speakers": 3}


def test_call_arguments_override_config_bounds(fake_pipeline: FakePipeline, wav_path: Path) -> None:
    config = CONFIG.model_copy(update={"min_speakers": 2, "max_speakers": 5})

    Diarizer(config, TOKEN).diarize(wav_path, max_speakers=3)

    assert fake_pipeline.calls[0][1] == {"min_speakers": 2, "max_speakers": 3}


@pytest.mark.parametrize("options", [{"num_speakers": 0}, {"min_speakers": 4, "max_speakers": 2}])
def test_invalid_speaker_options_are_rejected(wav_path: Path, options: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        Diarizer(CONFIG, TOKEN).diarize(wav_path, **options)


def test_close_allows_reloading(fake_pipeline: FakePipeline) -> None:
    diarizer = Diarizer(CONFIG, TOKEN)

    assert diarizer.load() is fake_pipeline
    diarizer.close()

    assert diarizer._pipeline is None
