"""Transcribe preprocessed audio with faster-whisper.

Usage:
    python -m notulen.asr.transcriber path/to/audio.mp3
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import time
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Any

from notulen.audio.preprocess import AudioChunk, PreprocessedAudio, preprocess_audio
from notulen.config import AsrConfig, VadConfig, load_config
from notulen.logging_utils import setup_logging
from notulen.runtime import load_torch_first
from notulen.schemas import Segment, Transcript, Word

logger = logging.getLogger(__name__)

CUDA_ERROR_MARKERS = ("cublas", "cudnn", "cuda", ".dll")


class TranscriptionError(RuntimeError):
    """Raised when the speech recognition model cannot run."""


def _wrap_runtime_error(exc: RuntimeError) -> Exception:
    """Attach installation guidance to CUDA library errors."""
    message = str(exc)
    if any(marker in message.lower() for marker in CUDA_ERROR_MARKERS):
        return TranscriptionError(
            f"CUDA libraries could not be loaded: {message}. "
            'Run: pip install -e ".[gpu]", or set asr.device to cpu and '
            "asr.compute_type to int8 in the config."
        )
    return exc


def convert_segment(raw: Any, offset: float) -> Segment | None:
    """Convert a faster-whisper segment to the project schema, shifted by offset seconds."""
    words = tuple(
        Word(
            text=word.word.strip(),
            start=round(word.start + offset, 3),
            end=round(max(word.start, word.end) + offset, 3),
            probability=min(max(word.probability, 0.0), 1.0),
        )
        for word in raw.words or ()
        if word.word.strip()
    )
    text = " ".join(word.text for word in words) if words else raw.text.strip()
    if not text:
        return None

    return Segment(
        start=words[0].start if words else round(raw.start + offset, 3),
        end=words[-1].end if words else round(max(raw.start, raw.end) + offset, 3),
        text=text,
        words=words,
        avg_logprob=raw.avg_logprob,
        no_speech_prob=raw.no_speech_prob,
    )


def _clip_segment(segment: Segment, lower: float, upper: float) -> Segment | None:
    """Keep only the part of a segment whose midpoints fall in [lower, upper)."""
    if not segment.words:
        midpoint = (segment.start + segment.end) / 2
        return segment if lower <= midpoint < upper else None

    kept = tuple(word for word in segment.words if lower <= (word.start + word.end) / 2 < upper)
    if not kept:
        return None
    if len(kept) == len(segment.words):
        return segment
    return segment.model_copy(
        update={
            "start": kept[0].start,
            "end": kept[-1].end,
            "text": " ".join(word.text for word in kept),
            "words": kept,
        }
    )


def merge_chunk_segments(
    results: Sequence[tuple[AudioChunk, Sequence[Segment]]],
) -> tuple[Segment, ...]:
    """Join chunk results, splitting each overlap at its midpoint to avoid duplicate words."""
    merged: list[Segment] = []
    for position, (chunk, segments) in enumerate(results):
        lower = -math.inf
        upper = math.inf
        if position > 0:
            lower = (results[position - 1][0].end + chunk.start) / 2
        if position < len(results) - 1:
            upper = (chunk.end + results[position + 1][0].start) / 2

        clipped = (_clip_segment(segment, lower, upper) for segment in segments)
        merged.extend(segment for segment in clipped if segment is not None)
    return tuple(sorted(merged, key=lambda segment: segment.start))


class Transcriber:
    """Load a Whisper model on demand and transcribe preprocessed audio."""

    def __init__(self, asr: AsrConfig, vad: VadConfig) -> None:
        self.asr = asr
        self.vad = vad
        self._model: Any | None = None

    def __enter__(self) -> Transcriber:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def load(self) -> Any:
        """Load the model once and return it."""
        if self._model is not None:
            return self._model

        load_torch_first()
        from faster_whisper import WhisperModel

        logger.info(
            "Loading Whisper model %s (%s, %s)",
            self.asr.model,
            self.asr.device,
            self.asr.compute_type,
        )
        try:
            self._model = WhisperModel(
                self.asr.model, device=self.asr.device, compute_type=self.asr.compute_type
            )
        except RuntimeError as exc:
            raise _wrap_runtime_error(exc) from exc
        return self._model

    def close(self) -> None:
        """Release the model so the next pipeline stage can use the GPU."""
        if self._model is not None:
            self._model = None
            gc.collect()
            logger.info("Released Whisper model")

    def transcribe_chunk(self, chunk: AudioChunk) -> list[Segment]:
        """Transcribe one chunk and return segments on the recording's timeline."""
        model = self.load()
        try:
            raw_segments, _ = model.transcribe(
                str(chunk.path),
                language=self.asr.language,
                beam_size=self.asr.beam_size,
                word_timestamps=self.asr.word_timestamps,
                vad_filter=self.asr.vad_filter,
                vad_parameters={
                    "threshold": self.vad.threshold,
                    "min_speech_duration_ms": self.vad.min_speech_duration_ms,
                    "min_silence_duration_ms": self.vad.min_silence_duration_ms,
                    "speech_pad_ms": self.vad.speech_pad_ms,
                },
                condition_on_previous_text=self.asr.condition_on_previous_text,
            )
            # faster-whisper decodes lazily, so inference errors surface during iteration.
            converted = [convert_segment(raw, chunk.start) for raw in raw_segments]
        except RuntimeError as exc:
            raise _wrap_runtime_error(exc) from exc
        return [segment for segment in converted if segment is not None]

    def transcribe(self, audio: PreprocessedAudio) -> Transcript:
        """Transcribe every chunk of a preprocessed recording."""
        started = time.perf_counter()
        results: list[tuple[AudioChunk, Sequence[Segment]]] = []
        for chunk in audio.chunks:
            logger.info(
                "Transcribing chunk %d/%d (%.1f-%.1f s)",
                chunk.index + 1,
                len(audio.chunks),
                chunk.start,
                chunk.end,
            )
            results.append((chunk, self.transcribe_chunk(chunk)))

        segments = merge_chunk_segments(results)
        elapsed = time.perf_counter() - started
        logger.info(
            "Transcribed %.1f s of audio in %.1f s (%.2fx realtime), %d segment(s)",
            audio.duration,
            elapsed,
            audio.duration / elapsed if elapsed > 0 else 0.0,
            len(segments),
        )
        return Transcript(
            source=audio.source.name,
            language=self.asr.language,
            model=self.asr.model,
            duration=round(audio.duration, 3),
            segments=segments,
        )


def main() -> int:
    """Transcribe one audio file and write the transcript as JSON."""
    parser = argparse.ArgumentParser(description="Transcribe an audio file to JSON.")
    parser.add_argument("audio", type=Path, help="Path to the input audio file.")
    parser.add_argument("--output", type=Path, help="Output JSON path.")
    args = parser.parse_args()

    config = load_config()
    setup_logging(config.logging.level)

    stem = args.audio.stem
    audio = preprocess_audio(args.audio, config.audio, config.vad, config.paths.temp_dir / stem)
    with Transcriber(config.asr, config.vad) as transcriber:
        transcript = transcriber.transcribe(audio)

    output = args.output or config.paths.output_dir / stem / "transcript_asr.json"
    transcript.save_json(output)
    logger.info("Transcript written to %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
