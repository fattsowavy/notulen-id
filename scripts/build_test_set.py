"""Build the evaluation test set from public Indonesian speech datasets.

- FLEURS Indonesian test split (CC-BY 4.0): read sentences from many speakers,
  used for WER and CER.
- LibriVox Indonesia test split (public domain, CC-0): utterances with reader
  IDs, stitched into simulated meetings with exact speaker labels for DER.

Downloads are cached in data/raw/ and audio is written to data/test_set/audio/,
both gitignored. Transcripts and RTTM files are small text files that can be
committed. Hand-written meeting scripts (naskah_*) are never touched.

Usage:
    python scripts/build_test_set.py
"""

from __future__ import annotations

import csv
import gzip
import logging
import random
import tarfile
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import httpx
import numpy as np
import soundfile as sf

from notulen.audio.preprocess import convert_to_wav, detect_speech, load_wav
from notulen.config import PROJECT_ROOT, AppConfig, DatasetBuildConfig, load_config
from notulen.diarization.diarizer import write_rttm
from notulen.logging_utils import setup_logging
from notulen.schemas import SpeakerTurn

logger = logging.getLogger("build_test_set")

FLEURS_URL = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/id_id"
LIBRIVOX_URL = "https://huggingface.co/datasets/indonesian-nlp/librivox-indonesia/resolve/main/data"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
GENERATED_PREFIXES = ("fleurs_", "rapat_")
SAMPLE_RATE = 16000
LOG_EVERY_BYTES = 50 * 1024 * 1024
DOWNLOAD_ATTEMPTS = 8


@dataclass(frozen=True)
class Clip:
    """One utterance from a source dataset."""

    archive_name: str
    text: str
    speaker: str


def download(url: str, target: Path) -> Path:
    """Download a file once, writing to a temporary name until it is complete."""
    if target.is_file():
        logger.info("Using cached %s", target.name)
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    logger.info("Downloading %s", url)
    # Only consecutive failures without progress count, so a slow connection that
    # keeps dropping still finishes.
    stalled = 0
    while True:
        before = partial.stat().st_size if partial.exists() else 0
        error: Exception | None = None
        try:
            if _download_part(url, partial):
                partial.replace(target)
                return target
        except httpx.TransportError as exc:
            error = exc
        after = partial.stat().st_size if partial.exists() else 0
        stalled = 0 if after > before else stalled + 1
        if stalled >= DOWNLOAD_ATTEMPTS:
            raise RuntimeError(f"Download of {url} stopped making progress") from error
        logger.warning(
            "Download interrupted at %d MB (%s), resuming",
            after // (1024 * 1024),
            error or "incomplete response",
        )
        time.sleep(min(2 ** (stalled + 1), 30))


def _download_part(url: str, partial: Path) -> bool:
    """Append to a partial download, resuming with an HTTP Range request.

    Returns True when the file is complete.
    """
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=60) as response:
        if response.status_code == 416:
            return True
        response.raise_for_status()
        if offset and response.status_code != 206:
            # The server ignored the range, so start over.
            offset = 0
        total = offset + int(response.headers.get("content-length", 0))
        received = offset
        next_log = (received // LOG_EVERY_BYTES + 1) * LOG_EVERY_BYTES
        with partial.open("ab" if offset else "wb") as handle:
            for block in response.iter_bytes(1024 * 1024):
                handle.write(block)
                received += len(block)
                if received >= next_log:
                    logger.info("  %d / %d MB", received // 2**20, total // 2**20)
                    next_log += LOG_EVERY_BYTES
    return total == offset or received >= total


def extract_files(archive: Path, wanted: Iterable[str], target_dir: Path) -> dict[str, Path]:
    """Copy files out of a tar archive by base name.

    Archive paths are never used as output paths, so a malicious archive cannot
    write outside target_dir.
    """
    remaining = set(wanted)
    found: dict[str, Path] = {}
    target_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tar:
        for member in tar:
            name = PurePosixPath(member.name).name
            if not member.isfile() or name not in remaining:
                continue
            source = tar.extractfile(member)
            if source is None:
                continue
            path = target_dir / name
            path.write_bytes(source.read())
            found[name] = path
            remaining.discard(name)
            if not remaining:
                break
    if remaining:
        raise FileNotFoundError(f"{len(remaining)} file(s) missing from {archive.name}")
    return found


def clean_generated(test_set: Path) -> None:
    """Remove files from a previous build, keeping hand-written items."""
    for folder in ("audio", "transcripts", "rttm"):
        for path in (test_set / folder).glob("*"):
            if path.name.startswith(GENERATED_PREFIXES):
                path.unlink()


def build_fleurs(config: AppConfig, test_set: Path) -> int:
    """Sample FLEURS utterances and write 16 kHz audio with reference transcripts."""
    build = config.evaluation.build
    raw = RAW_DIR / "fleurs"
    tsv = download(f"{FLEURS_URL}/test.tsv", raw / "test.tsv")
    archive = download(f"{FLEURS_URL}/audio/test.tar.gz", raw / "test.tar.gz")

    with tsv.open(encoding="utf-8", newline="") as handle:
        rows = sorted(csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE))
    # Columns: sentence id, file name, raw transcription, normalized transcription, ...
    rows = [row for row in rows if len(row) >= 3 and row[2].strip()]
    chosen = random.Random(build.seed).sample(rows, min(build.fleurs_utterances, len(rows)))

    clips = extract_files(archive, (row[1] for row in chosen), raw / "clips")
    for index, row in enumerate(chosen):
        item_id = f"fleurs_{index:03d}"
        convert_to_wav(clips[row[1]], test_set / "audio" / f"{item_id}.wav", SAMPLE_RATE, 1)
        (test_set / "transcripts" / f"{item_id}.txt").write_text(
            row[2].strip() + "\n", encoding="utf-8"
        )
    logger.info("Wrote %d FLEURS utterances", len(chosen))
    return len(chosen)


def _load_librivox_clips() -> dict[str, list[Clip]]:
    """Read Indonesian LibriVox utterances grouped by reader."""
    metadata = download(
        f"{LIBRIVOX_URL}/metadata_test.csv.gz", RAW_DIR / "librivox" / "test.csv.gz"
    )
    by_reader: dict[str, list[Clip]] = defaultdict(list)
    with gzip.open(metadata, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["language"] == "ind" and row["sentence"].strip():
                by_reader[row["reader"]].append(
                    Clip(PurePosixPath(row["path"]).name, row["sentence"].strip(), row["reader"])
                )
    return {
        reader: sorted(clips, key=lambda clip: clip.archive_name)
        for reader, clips in by_reader.items()
    }


def plan_meetings(by_reader: dict[str, list[Clip]], build: DatasetBuildConfig) -> list[list[Clip]]:
    """Choose speakers and turns for each meeting, never reusing an utterance."""
    rng = random.Random(build.seed)
    pools = {reader: list(clips) for reader, clips in by_reader.items()}
    for pool in pools.values():
        rng.shuffle(pool)

    meetings: list[list[Clip]] = []
    for _ in range(build.meetings):
        count = rng.randint(*build.speakers_per_meeting)
        needed = -(-build.turns_per_meeting // count) + 1
        eligible = sorted(reader for reader, pool in pools.items() if len(pool) >= needed)
        if len(eligible) < count:
            logger.warning("Not enough unused utterances for more meetings")
            break
        readers = rng.sample(eligible, count)

        # Everyone speaks once first, then turns alternate between different speakers
        # who still have unused utterances.
        meeting = [pools[reader].pop() for reader in readers]
        while len(meeting) < build.turns_per_meeting:
            options = [
                reader for reader in readers if reader != meeting[-1].speaker and pools[reader]
            ]
            if not options:
                break
            meeting.append(pools[rng.choice(options)].pop())
        meetings.append(meeting)
    return meetings


def build_meetings(config: AppConfig, test_set: Path) -> int:
    """Stitch LibriVox utterances into meetings with exact RTTM references."""
    build = config.evaluation.build
    meetings = plan_meetings(_load_librivox_clips(), build)
    archive = download(f"{LIBRIVOX_URL}/audio_test.tgz", RAW_DIR / "librivox" / "audio_test.tgz")
    wanted = {clip.archive_name for meeting in meetings for clip in meeting}
    files = extract_files(archive, wanted, RAW_DIR / "librivox" / "clips")

    rng = random.Random(build.seed + 1)
    for index, meeting in enumerate(meetings):
        item_id = f"rapat_{index:02d}"
        labels: dict[str, str] = {}
        pieces: list[np.ndarray] = []
        turns: list[SpeakerTurn] = []
        lines: list[str] = []
        position = 0.0
        for clip in meeting:
            wav = convert_to_wav(
                files[clip.archive_name], RAW_DIR / "librivox" / "wav" / f"{clip.archive_name}.wav"
            )
            samples, _ = load_wav(wav)
            regions = detect_speech(samples, SAMPLE_RATE, config.vad)
            if not regions:
                logger.warning("No speech found in %s, skipping", clip.archive_name)
                continue

            # Trim leading and trailing silence; label each speech region as the reader.
            start = int(regions[0].start * SAMPLE_RATE)
            end = int(regions[-1].end * SAMPLE_RATE)
            for region in regions:
                turns.append(
                    SpeakerTurn(
                        start=round(position + region.start - start / SAMPLE_RATE, 3),
                        end=round(position + region.end - start / SAMPLE_RATE, 3),
                        speaker=f"reader_{clip.speaker}",
                    )
                )
            label = labels.setdefault(clip.speaker, f"Pembicara {len(labels) + 1}")
            lines.append(f"{label}: {clip.text}")

            gap = rng.uniform(build.min_gap_seconds, build.max_gap_seconds)
            pieces += [samples[start:end], np.zeros(int(gap * SAMPLE_RATE), dtype=np.float32)]
            position += (end - start) / SAMPLE_RATE + int(gap * SAMPLE_RATE) / SAMPLE_RATE

        sf.write(
            test_set / "audio" / f"{item_id}.wav",
            np.concatenate(pieces),
            SAMPLE_RATE,
            subtype="PCM_16",
        )
        (test_set / "transcripts" / f"{item_id}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        write_rttm(turns, test_set / "rttm" / f"{item_id}.rttm", item_id)
        logger.info(
            "Wrote %s: %.1f s, %d speakers, %d turns", item_id, position, len(labels), len(lines)
        )
    return len(meetings)


def main() -> int:
    """Entry point."""
    config = load_config()
    setup_logging(config.logging.level)
    test_set = config.evaluation.test_set_dir
    for folder in ("audio", "transcripts", "rttm", "summaries"):
        (test_set / folder).mkdir(parents=True, exist_ok=True)

    clean_generated(test_set)
    utterances = build_fleurs(config, test_set)
    meetings = build_meetings(config, test_set)
    logger.info("Test set ready: %d FLEURS utterances, %d meetings", utterances, meetings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
