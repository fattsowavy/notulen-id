"""Diarization error rate."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate

from notulen.schemas import SpeakerTurn


@dataclass(frozen=True)
class DerScores:
    """Corpus-level DER with its components, in seconds of reference speech."""

    der: float
    confusion: float
    missed: float
    false_alarm: float
    total: float
    files: int


def to_annotation(turns: Sequence[SpeakerTurn], uri: str) -> Annotation:
    """Convert speaker turns to a pyannote annotation."""
    annotation = Annotation(uri=uri)
    for index, turn in enumerate(turns):
        if turn.end > turn.start:
            annotation[Segment(turn.start, turn.end), str(index)] = turn.speaker
    return annotation


def score_diarization(
    files: Sequence[tuple[str, Sequence[SpeakerTurn], Sequence[SpeakerTurn]]], collar: float
) -> DerScores:
    """Compute DER over (uri, reference, hypothesis) triples.

    Speaker labels do not need to match: the metric finds the best mapping.
    The scored region covers both reference and hypothesis, so speech detected
    outside the reference counts as false alarm.
    """
    if not files:
        raise ValueError("No files to score")

    metric = DiarizationErrorRate(collar=collar, skip_overlap=False)
    totals = {"confusion": 0.0, "missed detection": 0.0, "false alarm": 0.0, "total": 0.0}
    for uri, reference, hypothesis in files:
        end = max((turn.end for turn in [*reference, *hypothesis]), default=0.0)
        uem = Timeline([Segment(0.0, end)], uri=uri)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            detail = metric(
                to_annotation(reference, uri),
                to_annotation(hypothesis, uri),
                uem=uem,
                detailed=True,
            )
        for key in totals:
            totals[key] += float(detail[key])

    total = totals["total"]
    errors = totals["confusion"] + totals["missed detection"] + totals["false alarm"]
    return DerScores(
        der=errors / total if total > 0 else 0.0,
        confusion=totals["confusion"],
        missed=totals["missed detection"],
        false_alarm=totals["false alarm"],
        total=total,
        files=len(files),
    )
