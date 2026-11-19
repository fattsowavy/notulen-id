"""Word and character error rates."""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

import jiwer

logger = logging.getLogger(__name__)

NON_WORD = re.compile(r"[^\w\s]|_")


def normalize_text(text: str) -> str:
    """Lowercase, split hyphenated words, and drop punctuation before scoring.

    Hyphens become spaces so "orang-orang" and "orang orang" score the same.
    Numbers are not spelled out, so "60" against "enam puluh" counts as errors.
    """
    text = unicodedata.normalize("NFKC", text).lower().replace("-", " ")
    return " ".join(NON_WORD.sub(" ", text).split())


@dataclass(frozen=True)
class AsrScores:
    """Corpus-level ASR error rates."""

    wer: float
    cer: float
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int
    utterances: int


def score_asr(references: Sequence[str], hypotheses: Sequence[str]) -> AsrScores:
    """Compute corpus WER and CER over normalized text.

    Pairs with an empty reference are skipped because error rates are undefined.
    """
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")

    pairs = [
        (normalize_text(reference), normalize_text(hypothesis))
        for reference, hypothesis in zip(references, hypotheses, strict=True)
    ]
    kept = [(reference, hypothesis) for reference, hypothesis in pairs if reference]
    if len(kept) < len(pairs):
        logger.warning("Skipped %d pair(s) with an empty reference", len(pairs) - len(kept))
    if not kept:
        raise ValueError("No reference text to score")

    refs = [reference for reference, _ in kept]
    hyps = [hypothesis for _, hypothesis in kept]
    words = jiwer.process_words(refs, hyps)
    characters = jiwer.process_characters(refs, hyps)
    return AsrScores(
        wer=words.wer,
        cer=characters.cer,
        substitutions=words.substitutions,
        deletions=words.deletions,
        insertions=words.insertions,
        reference_words=sum(len(reference.split()) for reference in refs),
        utterances=len(kept),
    )
