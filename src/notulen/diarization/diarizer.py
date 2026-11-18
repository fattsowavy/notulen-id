"""Identify who spoke when with pyannote.audio.

Usage:
    python -m notulen.diarization.diarizer path/to/audio.mp3 --num-speakers 2
"""

from __future__ import annotations

import argparse
import gc
import logging
import re
import time
from pathlib import Path
from types import TracebackType
from typing import Any

import soundfile as sf
from pydantic import SecretStr

from notulen.audio.preprocess import convert_to_wav
from notulen.config import DiarizationConfig, load_config
from notulen.logging_utils import setup_logging
from notulen.schemas import SpeakerTurn

logger = logging.getLogger(__name__)

SPEAKER_LABEL = "SPEAKER_{index:02d}"
RTTM_FIELD_COUNT = 9


class DiarizationError(RuntimeError):
    """Raised when the diarization pipeline cannot be loaded."""


def annotation_to_turns(annotation: Any) -> list[SpeakerTurn]:
    """Convert a pyannote annotation to turns, numbering speakers by first appearance."""
    tracks = sorted(
        (
            (float(segment.start), float(segment.end), str(label))
            for segment, _, label in annotation.itertracks(yield_label=True)
        ),
        key=lambda track: (track[0], track[1]),
    )

    labels: dict[str, str] = {}
    turns: list[SpeakerTurn] = []
    for start, end, label in tracks:
        if end <= start:
            continue
        speaker = labels.setdefault(label, SPEAKER_LABEL.format(index=len(labels)))
        turns.append(SpeakerTurn(start=round(start, 3), end=round(end, 3), speaker=speaker))
    return turns


def write_rttm(turns: list[SpeakerTurn], path: Path, uri: str) -> Path:
    """Write speaker turns to an RTTM file."""
    file_id = re.sub(r"\s+", "_", uri.strip()) or "audio"
    lines = [
        f"SPEAKER {file_id} 1 {turn.start:.3f} {turn.end - turn.start:.3f} "
        f"<NA> <NA> {turn.speaker} <NA> <NA>\n"
        for turn in turns
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")
    return path


def read_rttm(path: Path) -> list[SpeakerTurn]:
    """Read speaker turns from an RTTM file."""
    turns: list[SpeakerTurn] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split()
        if not fields or fields[0] != "SPEAKER":
            continue
        if len(fields) < RTTM_FIELD_COUNT:
            raise ValueError(f"Malformed RTTM line {number} in {path}: {line!r}")
        start = float(fields[3])
        turns.append(
            SpeakerTurn(start=start, end=round(start + float(fields[4]), 3), speaker=fields[7])
        )
    return sorted(turns, key=lambda turn: (turn.start, turn.end))


class Diarizer:
    """Load a pyannote pipeline on demand and diarize WAV files."""

    def __init__(self, config: DiarizationConfig, hf_token: SecretStr | None) -> None:
        self.config = config
        self.hf_token = hf_token
        self._pipeline: Any | None = None

    def __enter__(self) -> Diarizer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def load(self) -> Any:
        """Load the pipeline once and move it to the configured device."""
        if self._pipeline is not None:
            return self._pipeline
        if self.hf_token is None:
            raise DiarizationError(
                "HF_TOKEN is not set. Add a Hugging Face read token to the .env file."
            )

        import torch
        from pyannote.audio import Pipeline

        logger.info("Loading diarization pipeline %s", self.config.pipeline)
        pipeline = Pipeline.from_pretrained(
            self.config.pipeline, use_auth_token=self.hf_token.get_secret_value()
        )
        if pipeline is None:
            raise DiarizationError(
                f"Could not load {self.config.pipeline}. Accept the model terms on Hugging Face "
                "for pyannote/speaker-diarization-3.1 and pyannote/segmentation-3.0."
            )

        device = self.config.device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA is not available, running diarization on CPU")
            device = "cpu"
        pipeline.to(torch.device(device))
        self._pipeline = pipeline
        return pipeline

    def close(self) -> None:
        """Release the pipeline and its GPU memory."""
        if self._pipeline is None:
            return
        self._pipeline = None
        gc.collect()

        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Released diarization pipeline")

    def _speaker_options(
        self, num_speakers: int | None, min_speakers: int | None, max_speakers: int | None
    ) -> dict[str, int]:
        """Combine call arguments with config defaults, preferring an exact speaker count."""
        exact = num_speakers if num_speakers is not None else self.config.num_speakers
        if exact is not None:
            if exact < 1:
                raise ValueError("num_speakers must be at least 1")
            return {"num_speakers": exact}

        lower = min_speakers if min_speakers is not None else self.config.min_speakers
        upper = max_speakers if max_speakers is not None else self.config.max_speakers
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("min_speakers must not be greater than max_speakers")

        options: dict[str, int] = {}
        if lower is not None:
            options["min_speakers"] = lower
        if upper is not None:
            options["max_speakers"] = upper
        return options

    def diarize(
        self,
        wav_path: Path,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> list[SpeakerTurn]:
        """Diarize a WAV file, passing audio in memory to avoid torchaudio decoding."""
        options = self._speaker_options(num_speakers, min_speakers, max_speakers)
        pipeline = self.load()

        import torch

        samples, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(samples.T)
        duration = samples.shape[0] / sample_rate

        started = time.perf_counter()
        annotation = pipeline({"waveform": waveform, "sample_rate": sample_rate}, **options)
        elapsed = time.perf_counter() - started

        turns = annotation_to_turns(annotation)
        logger.info(
            "Diarized %.1f s of audio in %.1f s: %d turn(s), %d speaker(s)",
            duration,
            elapsed,
            len(turns),
            len({turn.speaker for turn in turns}),
        )
        return turns


def main() -> int:
    """Diarize one audio file and write the result as RTTM."""
    parser = argparse.ArgumentParser(description="Diarize an audio file to RTTM.")
    parser.add_argument("audio", type=Path, help="Path to the input audio file.")
    parser.add_argument("--num-speakers", type=int, help="Exact number of speakers, if known.")
    parser.add_argument("--min-speakers", type=int, help="Lower bound on the number of speakers.")
    parser.add_argument("--max-speakers", type=int, help="Upper bound on the number of speakers.")
    parser.add_argument("--output", type=Path, help="Output RTTM path.")
    args = parser.parse_args()

    config = load_config()
    setup_logging(config.logging.level)

    stem = args.audio.stem
    wav_path = convert_to_wav(
        args.audio,
        config.paths.temp_dir / stem / "audio.wav",
        config.audio.sample_rate,
        config.audio.channels,
    )
    with Diarizer(config.diarization, config.secrets.hf_token) as diarizer:
        turns = diarizer.diarize(wav_path, args.num_speakers, args.min_speakers, args.max_speakers)

    output = args.output or config.paths.output_dir / stem / "diarization.rttm"
    write_rttm(turns, output, stem)
    logger.info("RTTM written to %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
