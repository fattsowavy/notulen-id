"""Assign speakers to transcribed words and group them into utterances."""

from __future__ import annotations

import logging
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import accumulate, groupby

from notulen.config import MergeConfig
from notulen.schemas import Segment, SpeakerTurn, Transcript, Word

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Item:
    """The smallest unit that receives a speaker: a word, or a segment without words."""

    speaker: str | None
    start: float
    end: float
    text: str
    words: tuple[Word, ...]


class _TurnIndex:
    """Find speaker turns near a time span, including turns that overlap each other."""

    def __init__(self, turns: Sequence[SpeakerTurn]) -> None:
        self.turns = sorted(turns, key=lambda turn: (turn.start, turn.end))
        self.starts = [turn.start for turn in self.turns]
        self.max_ends = list(accumulate((turn.end for turn in self.turns), max))

    def nearby(self, start: float, end: float, margin: float) -> list[SpeakerTurn]:
        """Return turns within margin seconds of the span, in time order."""
        found: list[SpeakerTurn] = []
        index = bisect_right(self.starts, end + margin) - 1
        while index >= 0 and self.max_ends[index] >= start - margin:
            turn = self.turns[index]
            if turn.end >= start - margin:
                found.append(turn)
            index -= 1
        found.reverse()
        return found


def _find_speaker(start: float, end: float, index: _TurnIndex, max_gap: float) -> str | None:
    """Pick the speaker with the most overlap, or the nearest turn within max_gap."""
    candidates = index.nearby(start, end, max_gap)

    overlaps: dict[str, float] = {}
    for turn in candidates:
        overlap = min(end, turn.end) - max(start, turn.start)
        if overlap > 0:
            overlaps[turn.speaker] = overlaps.get(turn.speaker, 0.0) + overlap
    if overlaps:
        return max(overlaps, key=lambda speaker: overlaps[speaker])

    if not candidates:
        return None
    nearest = min(candidates, key=lambda turn: max(turn.start - end, start - turn.end, 0.0))
    return nearest.speaker


def _build_items(transcript: Transcript, index: _TurnIndex, max_gap: float) -> list[_Item]:
    """Assign a speaker to every word, and to segments that have no word timestamps."""
    items: list[_Item] = []
    for segment in transcript.segments:
        if not segment.words:
            speaker = _find_speaker(segment.start, segment.end, index, max_gap)
            items.append(_Item(speaker, segment.start, segment.end, segment.text, ()))
            continue
        for word in segment.words:
            speaker = _find_speaker(word.start, word.end, index, max_gap)
            items.append(_Item(speaker, word.start, word.end, word.text, (word,)))
    return sorted(items, key=lambda item: item.start)


def _speaker_labels(items: Sequence[_Item], label_format: str) -> dict[str, str]:
    """Number speakers by the order in which they first speak."""
    labels: dict[str, str] = {}
    for item in items:
        if item.speaker is not None and item.speaker not in labels:
            labels[item.speaker] = label_format.format(index=len(labels) + 1)
    return labels


def align_transcript(
    transcript: Transcript, turns: Sequence[SpeakerTurn], config: MergeConfig
) -> Transcript:
    """Return a transcript whose segments are utterances from a single speaker.

    Consecutive words from the same speaker are merged. Words that cannot be
    matched to any speaker keep ``speaker=None``.
    """
    items = _build_items(transcript, _TurnIndex(turns), config.max_gap_seconds)
    labels = _speaker_labels(items, config.speaker_label_format)

    utterances: list[Segment] = []
    for speaker, group in groupby(items, key=lambda item: item.speaker):
        members = list(group)
        label = labels.get(speaker) if speaker is not None else None
        utterances.append(
            Segment(
                start=members[0].start,
                end=max(member.end for member in members),
                text=" ".join(member.text for member in members),
                words=tuple(
                    word.model_copy(update={"speaker": label})
                    for member in members
                    for word in member.words
                ),
                speaker=label,
            )
        )

    unassigned = sum(1 for item in items if item.speaker is None)
    if unassigned:
        logger.warning("%d of %d item(s) have no speaker", unassigned, len(items))
    logger.info("Aligned %d utterance(s) from %d speaker(s)", len(utterances), len(labels))
    return transcript.model_copy(update={"segments": tuple(utterances)})
