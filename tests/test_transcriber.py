"""Tests for the transcriber, using a fake Whisper model."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from notulen.asr.transcriber import (
    Transcriber,
    TranscriptionError,
    convert_segment,
    merge_chunk_segments,
)
from notulen.audio.preprocess import AudioChunk, PreprocessedAudio
from notulen.config import AsrConfig, VadConfig
from notulen.schemas import Segment, Word

ASR = AsrConfig(
    model="medium",
    device="cpu",
    compute_type="int8",
    language="id",
    beam_size=5,
    word_timestamps=True,
    vad_filter=True,
    condition_on_previous_text=False,
)
VAD = VadConfig(
    enabled=True,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=200,
)


def raw_word(text: str, start: float, end: float, probability: float = 0.9) -> SimpleNamespace:
    return SimpleNamespace(word=text, start=start, end=end, probability=probability)


def raw_segment(text: str, start: float, end: float, words: list[Any] | None) -> SimpleNamespace:
    return SimpleNamespace(
        text=text, start=start, end=end, words=words, avg_logprob=-0.2, no_speech_prob=0.01
    )


def make_segment(*words: tuple[str, float, float]) -> Segment:
    items = tuple(
        Word(text=text, start=start, end=end, probability=0.9) for text, start, end in words
    )
    return Segment(
        start=items[0].start,
        end=items[-1].end,
        text=" ".join(item.text for item in items),
        words=items,
    )


def make_chunk(index: int, start: float, end: float) -> AudioChunk:
    return AudioChunk(index=index, path=Path(f"chunk_{index}.wav"), start=start, end=end)


class FakeWhisperModel:
    """Record constructor and transcribe arguments and return canned segments."""

    instances: list[FakeWhisperModel] = []
    segments_by_path: dict[str, list[SimpleNamespace]] = {}

    def __init__(self, model: str, **kwargs: Any) -> None:
        self.model = model
        self.init_kwargs = kwargs
        self.calls: list[dict[str, Any]] = []
        FakeWhisperModel.instances.append(self)

    def transcribe(self, path: str, **kwargs: Any) -> tuple[Iterator[SimpleNamespace], None]:
        self.calls.append(kwargs)
        return iter(self.segments_by_path.get(path, [])), None


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> type[FakeWhisperModel]:
    pytest.importorskip("faster_whisper")
    FakeWhisperModel.instances = []
    FakeWhisperModel.segments_by_path = {}
    monkeypatch.setattr("faster_whisper.WhisperModel", FakeWhisperModel)
    return FakeWhisperModel


def test_convert_segment_applies_offset_and_strips_words() -> None:
    raw = raw_segment(
        " selamat pagi", 1.0, 2.0, [raw_word(" selamat", 1.0, 1.5), raw_word(" pagi", 1.5, 2.0)]
    )

    segment = convert_segment(raw, offset=10.0)

    assert segment is not None
    assert segment.text == "selamat pagi"
    assert (segment.start, segment.end) == (11.0, 12.0)
    assert segment.words[1].text == "pagi"
    assert segment.words[1].start == 11.5


def test_convert_segment_repairs_inverted_word_times_and_probability() -> None:
    raw = raw_segment(" ya", 1.0, 1.0, [raw_word(" ya", 1.2, 1.1, probability=1.0000001)])

    segment = convert_segment(raw, offset=0.0)

    assert segment is not None
    assert segment.words[0].end == segment.words[0].start
    assert segment.words[0].probability == 1.0


def test_convert_segment_uses_text_without_word_timestamps() -> None:
    segment = convert_segment(raw_segment(" halo semua ", 1.0, 2.0, None), offset=5.0)

    assert segment is not None
    assert segment.text == "halo semua"
    assert (segment.start, segment.end) == (6.0, 7.0)


def test_convert_segment_drops_empty_text() -> None:
    assert convert_segment(raw_segment("   ", 0.0, 1.0, []), offset=0.0) is None


def test_merge_removes_duplicate_words_in_overlap() -> None:
    first = make_chunk(0, 0.0, 10.0)
    second = make_chunk(1, 8.0, 18.0)
    left = make_segment(("satu", 7.0, 7.5), ("dua", 8.5, 8.8), ("tiga", 9.5, 9.8))
    right = make_segment(("dua", 8.5, 8.8), ("tiga", 9.5, 9.8), ("empat", 10.5, 11.0))

    merged = merge_chunk_segments([(first, [left]), (second, [right])])

    assert [segment.text for segment in merged] == ["satu dua", "tiga empat"]
    assert merged[0].end == 8.8
    assert merged[1].start == 9.5


def test_merge_keeps_everything_when_chunks_touch() -> None:
    first = make_chunk(0, 0.0, 5.0)
    second = make_chunk(1, 5.0, 10.0)
    left = make_segment(("halo", 4.0, 4.5))
    right = make_segment(("dunia", 5.5, 6.0))

    merged = merge_chunk_segments([(first, [left]), (second, [right])])

    assert [segment.text for segment in merged] == ["halo", "dunia"]


def test_merge_filters_segments_without_words_by_midpoint() -> None:
    first = make_chunk(0, 0.0, 10.0)
    second = make_chunk(1, 8.0, 18.0)
    duplicate = Segment(start=9.5, end=10.0, text="ulang")
    kept = Segment(start=9.5, end=10.0, text="ulang")

    merged = merge_chunk_segments([(first, [duplicate]), (second, [kept])])

    assert len(merged) == 1


def test_transcribe_passes_configured_options(fake_model: type[FakeWhisperModel]) -> None:
    chunk = make_chunk(0, 0.0, 2.0)
    fake_model.segments_by_path = {
        str(chunk.path): [raw_segment(" halo", 0.0, 1.0, [raw_word(" halo", 0.0, 1.0)])]
    }
    audio = PreprocessedAudio(
        source=Path("rapat.mp3").resolve(),
        wav_path=chunk.path,
        sample_rate=16000,
        duration=2.0,
        speech_regions=(),
        chunks=(chunk,),
    )

    with Transcriber(ASR, VAD) as transcriber:
        transcript = transcriber.transcribe(audio)

    call = fake_model.instances[0].calls[0]
    assert call["language"] == "id"
    assert call["word_timestamps"] is True
    assert call["vad_filter"] is True
    assert call["condition_on_previous_text"] is False
    assert call["vad_parameters"]["min_silence_duration_ms"] == 500
    assert transcript.source == "rapat.mp3"
    assert transcript.text == "halo"


def test_close_releases_model(fake_model: type[FakeWhisperModel]) -> None:
    transcriber = Transcriber(ASR, VAD)

    first = transcriber.load()
    assert transcriber.load() is first
    transcriber.close()

    assert transcriber.load() is not first


def test_load_imports_torch_before_whisper(
    fake_model: type[FakeWhisperModel], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr("notulen.asr.transcriber.load_torch_first", lambda: calls.append("torch"))

    Transcriber(ASR, VAD).load()

    assert calls == ["torch"]


def test_missing_cuda_library_gives_actionable_error(
    fake_model: type[FakeWhisperModel], monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(*args: Any, **kwargs: Any) -> tuple[Iterator[Any], None]:
        def generate() -> Iterator[Any]:
            raise RuntimeError("Library cublas12.dll is not found or cannot be loaded")
            yield

        return generate(), None

    monkeypatch.setattr(FakeWhisperModel, "transcribe", failing)

    with pytest.raises(TranscriptionError, match="gpu"):
        Transcriber(ASR, VAD).transcribe_chunk(make_chunk(0, 0.0, 1.0))
