"""Discover and read items in data/test_set/.

Each item is identified by its file stem across these folders:

- audio/<id>.wav: recording (optional)
- transcripts/<id>.txt: reference transcript, one utterance per line,
  optionally prefixed with "Pembicara N: "
- rttm/<id>.rttm: reference speaker turns (optional)
- summaries/<id>.json: reference summary in the Summary schema (optional)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from notulen.schemas import Segment, Summary, Transcript, Word

AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a", ".ogg")
SPEAKER_PREFIX = re.compile(r"^(Pembicara \d+):\s*")
SECONDS_PER_WORD = 0.4
SECONDS_BETWEEN_LINES = 1.0


@dataclass(frozen=True)
class EvalItem:
    """Files available for one test set item."""

    item_id: str
    audio: Path | None
    transcript: Path | None
    rttm: Path | None
    summary: Path | None

    @property
    def subset(self) -> str:
        """Name of the group the item belongs to, taken from its id prefix."""
        return self.item_id.split("_", 1)[0]


def _find(folder: Path, item_id: str, extensions: tuple[str, ...]) -> Path | None:
    return next(
        (
            folder / f"{item_id}{ext}"
            for ext in extensions
            if (folder / f"{item_id}{ext}").is_file()
        ),
        None,
    )


def discover(test_set_dir: Path) -> list[EvalItem]:
    """List every item that has at least one reference file."""
    ids = {
        path.stem
        for folder, pattern in (
            ("transcripts", "*.txt"),
            ("rttm", "*.rttm"),
            ("summaries", "*.json"),
        )
        for path in (test_set_dir / folder).glob(pattern)
    }
    return [
        EvalItem(
            item_id=item_id,
            audio=_find(test_set_dir / "audio", item_id, AUDIO_EXTENSIONS),
            transcript=_find(test_set_dir / "transcripts", item_id, (".txt",)),
            rttm=_find(test_set_dir / "rttm", item_id, (".rttm",)),
            summary=_find(test_set_dir / "summaries", item_id, (".json",)),
        )
        for item_id in sorted(ids)
    ]


def read_lines(path: Path) -> list[tuple[str | None, str]]:
    """Read (speaker label, text) pairs from a reference transcript."""
    lines: list[tuple[str | None, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        match = SPEAKER_PREFIX.match(line)
        if match:
            lines.append((match.group(1), line[match.end() :].strip()))
        else:
            lines.append((None, line))
    return lines


def reference_text(path: Path) -> str:
    """Return the reference transcript as plain text, without speaker labels."""
    return " ".join(text for _, text in read_lines(path))


def transcript_from_text(path: Path) -> Transcript:
    """Build a transcript from a reference text file, with evenly spaced word times.

    Used to evaluate summarization on its own, without ASR errors.
    """
    segments: list[Segment] = []
    position = 0.0
    for speaker, text in read_lines(path):
        words: list[Word] = []
        for token in text.split():
            words.append(
                Word(
                    text=token,
                    start=round(position, 3),
                    end=round(position + SECONDS_PER_WORD * 0.9, 3),
                    probability=1.0,
                    speaker=speaker,
                )
            )
            position += SECONDS_PER_WORD
        if words:
            segments.append(
                Segment(
                    start=words[0].start,
                    end=words[-1].end,
                    text=" ".join(word.text for word in words),
                    words=tuple(words),
                    speaker=speaker,
                )
            )
        position += SECONDS_BETWEEN_LINES
    return Transcript(
        source=path.name,
        language="id",
        model="reference",
        duration=round(position, 3),
        segments=tuple(segments),
    )


def load_summary(path: Path) -> Summary:
    """Read a reference summary."""
    return Summary.model_validate_json(path.read_text(encoding="utf-8"))
