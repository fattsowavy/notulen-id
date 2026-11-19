"""Precision, recall, and F1 of extracted action items."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from notulen.schemas import ActionItem
from notulen.summarize.summarizer import HONORIFICS, tokenize


def token_f1(first: str, second: str) -> float:
    """Token-overlap F1 between two texts, as used in extractive QA evaluation."""
    left, right = Counter(tokenize(first)), Counter(tokenize(second))
    common = sum((left & right).values())
    if common == 0:
        return 0.0
    precision = common / sum(left.values())
    recall = common / sum(right.values())
    return 2 * precision * recall / (precision + recall)


def item_similarity(predicted: ActionItem, reference: ActionItem) -> float:
    """Similarity of two items: the better of task overlap and quote overlap."""
    return max(
        token_f1(predicted.tugas, reference.tugas),
        token_f1(predicted.kutipan_sumber, reference.kutipan_sumber),
    )


def match_items(
    predicted: Sequence[ActionItem], reference: Sequence[ActionItem], threshold: float
) -> list[tuple[int, int]]:
    """Pair items one-to-one, most similar first, keeping pairs at or above threshold."""
    candidates = sorted(
        (
            (item_similarity(pred, ref), pred_index, ref_index)
            for pred_index, pred in enumerate(predicted)
            for ref_index, ref in enumerate(reference)
        ),
        reverse=True,
    )
    used_pred: set[int] = set()
    used_ref: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for score, pred_index, ref_index in candidates:
        if score < threshold:
            break
        if pred_index in used_pred or ref_index in used_ref:
            continue
        used_pred.add(pred_index)
        used_ref.add(ref_index)
        pairs.append((pred_index, ref_index))
    return pairs


def same_owner(predicted: str | None, reference: str | None) -> bool:
    """Compare owners by name tokens, ignoring honorifics such as "Bu" or "Pak"."""
    if predicted is None or reference is None:
        return predicted is reference
    left = set(tokenize(predicted)) - HONORIFICS
    right = set(tokenize(reference)) - HONORIFICS
    if not left or not right:
        # Labels such as "Pembicara 2" reduce to their number.
        left, right = set(tokenize(predicted)), set(tokenize(reference))
    return bool(left & right)


@dataclass(frozen=True)
class ActionScores:
    """Micro-averaged action item scores across documents."""

    precision: float
    recall: float
    f1: float
    predicted: int
    reference: int
    matched: int
    owner_accuracy: float | None


def score_action_items(
    documents: Sequence[tuple[Sequence[ActionItem], Sequence[ActionItem]]], threshold: float
) -> ActionScores:
    """Score (predicted, reference) item lists from several documents."""
    predicted = sum(len(pred) for pred, _ in documents)
    reference = sum(len(ref) for _, ref in documents)
    matched = 0
    owners_correct = 0
    for pred, ref in documents:
        pairs = match_items(pred, ref, threshold)
        matched += len(pairs)
        owners_correct += sum(
            same_owner(pred[i].penanggung_jawab, ref[j].penanggung_jawab) for i, j in pairs
        )

    # With nothing predicted (or nothing expected), a perfect score means both are empty.
    precision = matched / predicted if predicted else float(reference == 0)
    recall = matched / reference if reference else float(predicted == 0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return ActionScores(
        precision=precision,
        recall=recall,
        f1=f1,
        predicted=predicted,
        reference=reference,
        matched=matched,
        owner_accuracy=owners_correct / matched if matched else None,
    )
