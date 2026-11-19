"""Turn an audio file into a speaker-labelled transcript and summary.

Usage:
    python scripts/run_pipeline.py data/samples/rapat.mp3 --num-speakers 3
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from notulen.asr.transcriber import TranscriptionError
from notulen.audio.preprocess import AudioPreprocessError
from notulen.config import DEFAULT_CONFIG_PATH, load_config
from notulen.diarization.diarizer import DiarizationError
from notulen.logging_utils import setup_logging
from notulen.pipeline import NotulenPipeline

logger = logging.getLogger("run_pipeline")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("audio", type=Path, help="Input audio or video file.")
    parser.add_argument("--num-speakers", type=int, help="Exact number of speakers, if known.")
    parser.add_argument("--output-dir", type=Path, help="Folder for the results.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="YAML config.")
    parser.add_argument("--no-diarization", action="store_true", help="Skip speaker labels.")
    parser.add_argument("--no-summary", action="store_true", help="Skip the LLM summary.")
    return parser.parse_args()


def main() -> int:
    """Entry point. Returns 1 on failure, including a failed summary."""
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config.logging.level)

    try:
        with NotulenPipeline(config) as pipeline:
            result = pipeline.run(
                args.audio,
                output_dir=args.output_dir,
                num_speakers=args.num_speakers,
                diarize=not args.no_diarization,
                summarize=not args.no_summary,
            )
    except (
        FileNotFoundError,
        AudioPreprocessError,
        TranscriptionError,
        DiarizationError,
    ) as exc:
        logger.error("%s", exc)
        return 1

    for name, seconds in result.timings.items():
        logger.info("%-12s %7.1f s", name, seconds)
    for name, path in result.files.items():
        logger.info("%-16s %s", name, path)

    if result.summary_error:
        logger.error("Transcript saved, but the summary failed: %s", result.summary_error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
