"""Normalize audio with ffmpeg, detect speech, and split long recordings."""

from __future__ import annotations

import logging
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from notulen.config import AudioConfig, VadConfig
from notulen.runtime import load_torch_first

logger = logging.getLogger(__name__)

VAD_SAMPLE_RATE = 16000
COPY_BLOCK_FRAMES = 65536


class AudioPreprocessError(RuntimeError):
    """Raised when an audio file cannot be converted or read."""


@dataclass(frozen=True)
class SpeechRegion:
    """A span of detected speech, in seconds."""

    start: float
    end: float


@dataclass(frozen=True)
class AudioChunk:
    """A WAV file covering a span of the normalized recording, in seconds."""

    index: int
    path: Path
    start: float
    end: float


@dataclass(frozen=True)
class PreprocessedAudio:
    """Result of preprocessing one input file."""

    source: Path
    wav_path: Path
    sample_rate: int
    duration: float
    speech_regions: tuple[SpeechRegion, ...]
    chunks: tuple[AudioChunk, ...]


def find_ffmpeg() -> str:
    """Return the ffmpeg executable path, or raise if it is not on PATH."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise AudioPreprocessError(
            "ffmpeg was not found on PATH. Install it with: conda install -c conda-forge ffmpeg"
        )
    return ffmpeg


def convert_to_wav(
    input_path: Path, output_path: Path, sample_rate: int = 16000, channels: int = 1
) -> Path:
    """Convert the first audio stream of a file to 16-bit PCM WAV."""
    source = input_path.resolve()
    target = output_path.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Audio file not found: {source}")
    if source == target:
        raise ValueError(f"Input and output must be different files: {source}")

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.stem}.part.wav")
    command = [
        find_ffmpeg(),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(partial),
    ]
    result = subprocess.run(command, capture_output=True, text=True, errors="replace", check=False)
    if result.returncode != 0:
        partial.unlink(missing_ok=True)
        raise AudioPreprocessError(f"ffmpeg failed for {source}: {result.stderr.strip()}")

    partial.replace(target)
    logger.info("Converted %s to %d Hz, %d channel(s)", source.name, sample_rate, channels)
    return target


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a WAV file as mono float32 samples."""
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sample_rate


def detect_speech(audio: np.ndarray, sample_rate: int, vad: VadConfig) -> list[SpeechRegion]:
    """Find speech regions with the Silero VAD bundled in faster-whisper."""
    if sample_rate != VAD_SAMPLE_RATE:
        raise ValueError(f"VAD requires {VAD_SAMPLE_RATE} Hz audio, got {sample_rate} Hz")
    if audio.size == 0:
        return []

    load_torch_first()
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    options = VadOptions(
        threshold=vad.threshold,
        min_speech_duration_ms=vad.min_speech_duration_ms,
        min_silence_duration_ms=vad.min_silence_duration_ms,
        speech_pad_ms=vad.speech_pad_ms,
    )
    timestamps = get_speech_timestamps(audio, vad_options=options, sampling_rate=sample_rate)
    return [
        SpeechRegion(start=item["start"] / sample_rate, end=item["end"] / sample_rate)
        for item in timestamps
    ]


def _silence_gaps(regions: Sequence[SpeechRegion], duration: float) -> list[tuple[float, float]]:
    """Return the silent spans between speech regions, in time order."""
    if not regions:
        return []

    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for region in sorted(regions, key=lambda item: item.start):
        if region.start > cursor:
            gaps.append((cursor, region.start))
        cursor = max(cursor, region.end)
    if cursor < duration:
        gaps.append((cursor, duration))
    return gaps


def _find_silent_cut(
    gaps: Sequence[tuple[float, float]], lower: float, upper: float
) -> float | None:
    """Return the midpoint of the latest silence inside the window, if any."""
    for gap_start, gap_end in reversed(gaps):
        start = max(gap_start, lower)
        end = min(gap_end, upper)
        if start < end:
            return (start + end) / 2
    return None


def plan_chunks(
    duration: float,
    max_chunk_seconds: float,
    overlap_seconds: float,
    speech_regions: Sequence[SpeechRegion] = (),
) -> list[tuple[float, float]]:
    """Plan chunk spans, cutting in silence and overlapping only when cutting through speech.

    A silent cut is searched in the second half of each chunk. Without one, the
    chunk is cut at its maximum length and the next chunk overlaps it.
    """
    if max_chunk_seconds <= 0:
        raise ValueError("max_chunk_seconds must be positive")
    if not 0 <= overlap_seconds < max_chunk_seconds:
        raise ValueError("overlap_seconds must be in the range [0, max_chunk_seconds)")
    if duration <= 0:
        return []

    gaps = _silence_gaps(speech_regions, duration)
    spans: list[tuple[float, float]] = []
    start = 0.0
    while start + max_chunk_seconds < duration:
        limit = start + max_chunk_seconds
        cut = _find_silent_cut(gaps, start + max_chunk_seconds / 2, limit)
        if cut is None:
            spans.append((start, limit))
            start = limit - overlap_seconds
        else:
            spans.append((start, cut))
            start = cut
    spans.append((start, duration))
    return spans


def write_chunks(
    wav_path: Path, spans: Sequence[tuple[float, float]], output_dir: Path
) -> tuple[AudioChunk, ...]:
    """Write each span to its own WAV file, reusing the input when one span is enough."""
    if len(spans) == 1:
        start, end = spans[0]
        return (AudioChunk(index=0, path=wav_path, start=start, end=end),)

    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("chunk_*.wav"):
        stale.unlink()

    chunks: list[AudioChunk] = []
    with sf.SoundFile(wav_path) as source:
        for index, (start, end) in enumerate(spans):
            first_frame = round(start * source.samplerate)
            remaining = min(round(end * source.samplerate), source.frames) - first_frame
            chunk_path = output_dir / f"chunk_{index:04d}.wav"

            source.seek(first_frame)
            with sf.SoundFile(
                chunk_path,
                mode="w",
                samplerate=source.samplerate,
                channels=source.channels,
                subtype="PCM_16",
            ) as target:
                while remaining > 0:
                    block = source.read(min(remaining, COPY_BLOCK_FRAMES), dtype="int16")
                    if len(block) == 0:
                        break
                    target.write(block)
                    remaining -= len(block)

            chunks.append(AudioChunk(index=index, path=chunk_path, start=start, end=end))
    return tuple(chunks)


def preprocess_audio(
    input_path: Path, audio: AudioConfig, vad: VadConfig, work_dir: Path
) -> PreprocessedAudio:
    """Convert an input file to normalized WAV, detect speech, and split it into chunks."""
    wav_path = convert_to_wav(input_path, work_dir / "audio.wav", audio.sample_rate, audio.channels)
    samples, sample_rate = load_wav(wav_path)
    duration = samples.size / sample_rate
    if duration == 0:
        raise AudioPreprocessError(f"Audio file contains no samples: {input_path}")

    if vad.enabled:
        regions = detect_speech(samples, sample_rate, vad)
        speech_seconds = sum(region.end - region.start for region in regions)
        logger.info(
            "VAD found %d region(s), %.1f s of speech in %.1f s of audio",
            len(regions),
            speech_seconds,
            duration,
        )
        if not regions:
            logger.warning("No speech detected in %s", input_path)
    else:
        regions = [SpeechRegion(start=0.0, end=duration)]

    spans = plan_chunks(duration, audio.max_chunk_seconds, audio.chunk_overlap_seconds, regions)
    chunks = write_chunks(wav_path, spans, work_dir / "chunks")
    logger.info("Prepared %d chunk(s) from %s", len(chunks), input_path)

    return PreprocessedAudio(
        source=input_path.resolve(),
        wav_path=wav_path,
        sample_rate=sample_rate,
        duration=duration,
        speech_regions=tuple(regions),
        chunks=chunks,
    )
