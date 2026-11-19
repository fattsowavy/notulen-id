"""Summary quality: ROUGE-L and BERTScore against reference summaries."""

from __future__ import annotations

from collections.abc import Sequence

from rouge_score import rouge_scorer

from notulen.schemas import Summary


def summary_text(summary: Summary) -> str:
    """Join topics, decisions, and key points into text. Action items are scored separately."""
    return "\n".join([*summary.topik, *summary.keputusan, *summary.poin_penting])


def rouge_l(references: Sequence[str], hypotheses: Sequence[str]) -> list[float]:
    """Return the ROUGE-L F1 of each pair."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return [
        scorer.score(reference, hypothesis)["rougeL"].fmeasure
        for reference, hypothesis in zip(references, hypotheses, strict=True)
    ]


def bertscore_f1(references: Sequence[str], hypotheses: Sequence[str], model: str) -> list[float]:
    """Return the BERTScore F1 of each pair, using a multilingual model for Indonesian."""
    from bert_score import score

    _, _, f1 = score(
        list(hypotheses), list(references), model_type=model, verbose=False, batch_size=8
    )
    return [float(value) for value in f1]
