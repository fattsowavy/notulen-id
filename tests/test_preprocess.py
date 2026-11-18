"""Tests for audio preprocessing."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from notulen.audio.preprocess import (
    AudioPreprocessError,
    SpeechRegion,
    convert_to_wav,
    detect_speech,
    load_wav,
    plan_chunks,
    preprocess_audio,
    write_chunks,
)
from notulen.config import AudioConfig, VadConfig

requires_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")

VAD = VadConfig(
    enabled=True,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=200,
)


def write_tone(path: Path, seconds: float, sample_rate: int = 44100, channels: int = 2) -> Path:
    """Write a sine tone to a WAV file."""
    times = np.arange(int(seconds * sample_rate)) / sample_rate
    tone = 0.3 * np.sin(2 * np.pi * 440 * times)
    data = np.column_stack([tone] * channels) if channels > 1 else tone
    sf.write(path, data, sample_rate, subtype="PCM_16")
    return path


@requires_ffmpeg
def test_convert_to_wav_normalizes_format(tmp_path: Path) -> None:
    source = write_tone(tmp_path / "stereo.wav", seconds=2.0)

    output = convert_to_wav(source, tmp_path / "out" / "audio.wav")

    info = sf.info(output)
    assert (info.samplerate, info.channels, info.subtype) == (16000, 1, "PCM_16")
    assert info.duration == pytest.approx(2.0, abs=0.05)
    assert not (tmp_path / "out" / "audio.part.wav").exists()


def test_convert_to_wav_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        convert_to_wav(tmp_path / "missing.mp3", tmp_path / "audio.wav")


def test_convert_to_wav_rejects_same_input_and_output(tmp_path: Path) -> None:
    source = write_tone(tmp_path / "audio.wav", seconds=0.1)

    with pytest.raises(ValueError):
        convert_to_wav(source, source)


@requires_ffmpeg
def test_convert_to_wav_reports_invalid_audio(tmp_path: Path) -> None:
    source = tmp_path / "broken.mp3"
    source.write_bytes(b"this is not audio")

    with pytest.raises(AudioPreprocessError):
        convert_to_wav(source, tmp_path / "audio.wav")

    assert not (tmp_path / "audio.wav").exists()
    assert not (tmp_path / "audio.part.wav").exists()


def test_plan_chunks_keeps_short_audio_whole() -> None:
    assert plan_chunks(30.0, 60.0, 2.0) == [(0.0, 30.0)]


def test_plan_chunks_returns_nothing_for_empty_audio() -> None:
    assert plan_chunks(0.0, 60.0, 2.0) == []


def test_plan_chunks_overlaps_when_no_silence_is_available() -> None:
    assert plan_chunks(25.0, 10.0, 2.0) == [(0.0, 10.0), (8.0, 18.0), (16.0, 25.0)]


def test_plan_chunks_cuts_inside_silence_without_overlap() -> None:
    regions = [SpeechRegion(0.0, 7.0), SpeechRegion(8.0, 15.0)]

    assert plan_chunks(15.0, 10.0, 2.0, regions) == [(0.0, 7.5), (7.5, 15.0)]


def test_plan_chunks_prefers_latest_silence() -> None:
    regions = [SpeechRegion(0.0, 6.0), SpeechRegion(6.5, 8.0), SpeechRegion(9.0, 12.0)]

    assert plan_chunks(12.0, 10.0, 2.0, regions)[0] == (0.0, 8.5)


def test_plan_chunks_ignores_silence_early_in_the_chunk() -> None:
    regions = [SpeechRegion(0.0, 2.0), SpeechRegion(3.0, 25.0)]

    assert plan_chunks(25.0, 10.0, 2.0, regions)[0] == (0.0, 10.0)


@pytest.mark.parametrize(("maximum", "overlap"), [(0.0, 0.0), (10.0, 10.0), (10.0, -1.0)])
def test_plan_chunks_rejects_invalid_settings(maximum: float, overlap: float) -> None:
    with pytest.raises(ValueError):
        plan_chunks(30.0, maximum, overlap)


def test_write_chunks_reuses_file_for_single_span(tmp_path: Path) -> None:
    wav = write_tone(tmp_path / "audio.wav", seconds=1.0, sample_rate=16000, channels=1)

    chunks = write_chunks(wav, [(0.0, 1.0)], tmp_path / "chunks")

    assert [chunk.path for chunk in chunks] == [wav]
    assert not (tmp_path / "chunks").exists()


def test_write_chunks_preserves_samples_and_offsets(tmp_path: Path) -> None:
    sample_rate = 1000
    samples = np.arange(3000, dtype=np.int16)
    wav = tmp_path / "audio.wav"
    sf.write(wav, samples, sample_rate, subtype="PCM_16")
    stale = tmp_path / "chunks" / "chunk_0099.wav"
    stale.parent.mkdir()
    stale.write_bytes(b"old")

    chunks = write_chunks(wav, [(0.0, 2.0), (1.5, 3.0)], tmp_path / "chunks")

    first, _ = sf.read(chunks[0].path, dtype="int16")
    second, _ = sf.read(chunks[1].path, dtype="int16")
    assert np.array_equal(first, samples[:2000])
    assert np.array_equal(second, samples[1500:])
    assert (chunks[1].start, chunks[1].end) == (1.5, 3.0)
    assert not stale.exists()


def test_detect_speech_converts_samples_to_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("faster_whisper")
    monkeypatch.setattr(
        "faster_whisper.vad.get_speech_timestamps",
        lambda *args, **kwargs: [{"start": 8000, "end": 24000}],
    )

    regions = detect_speech(np.zeros(32000, dtype=np.float32), 16000, VAD)

    assert regions == [SpeechRegion(0.5, 1.5)]


def test_detect_speech_finds_nothing_in_silence() -> None:
    pytest.importorskip("faster_whisper")

    assert detect_speech(np.zeros(32000, dtype=np.float32), 16000, VAD) == []


def test_detect_speech_requires_16_khz() -> None:
    with pytest.raises(ValueError):
        detect_speech(np.zeros(100, dtype=np.float32), 8000, VAD)


@requires_ffmpeg
def test_preprocess_audio_end_to_end(tmp_path: Path) -> None:
    source = write_tone(tmp_path / "meeting.wav", seconds=5.0)
    audio = AudioConfig(
        sample_rate=16000, channels=1, max_chunk_seconds=2.0, chunk_overlap_seconds=0.5
    )
    vad = VAD.model_copy(update={"enabled": False})

    result = preprocess_audio(source, audio, vad, tmp_path / "work")

    samples, sample_rate = load_wav(result.wav_path)
    assert sample_rate == 16000
    assert result.duration == pytest.approx(5.0, abs=0.05)
    assert samples.size == pytest.approx(80000, abs=800)
    assert [chunk.start for chunk in result.chunks] == [0.0, 1.5, 3.0]
    assert result.chunks[-1].end == pytest.approx(result.duration)
    assert all(chunk.path.is_file() for chunk in result.chunks)
