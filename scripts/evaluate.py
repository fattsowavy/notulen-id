"""Evaluate every pipeline stage on data/test_set/ and write reports/results.md.

- ASR: WER and CER on items with audio and a reference transcript.
- Diarization: DER on items with audio and a reference RTTM, both with the
  speaker count detected automatically and with the true count given.
- Summary: ROUGE-L and BERTScore on items with a reference summary. The LLM
  reads the reference transcript, so these scores measure summarization alone.
- Action items: precision, recall, and F1 on the same items.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --skip-asr --skip-der
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import shutil
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

from notulen.asr.transcriber import Transcriber
from notulen.audio.preprocess import preprocess_audio
from notulen.config import PROJECT_ROOT, AppConfig, load_config
from notulen.diarization.diarizer import Diarizer, read_rttm, write_rttm
from notulen.evaluation.action_metrics import ActionScores, score_action_items
from notulen.evaluation.asr_metrics import score_asr
from notulen.evaluation.diarization_metrics import DerScores, score_diarization
from notulen.evaluation.summary_metrics import bertscore_f1, rouge_l, summary_text
from notulen.evaluation.test_set import (
    EvalItem,
    discover,
    load_summary,
    reference_text,
    transcript_from_text,
)
from notulen.logging_utils import setup_logging
from notulen.schemas import Summary
from notulen.summarize.llm_client import create_llm_client
from notulen.summarize.summarizer import summarize_transcript

logger = logging.getLogger("evaluate")

SUBSET_NAMES = {
    "fleurs": "FLEURS (ucapan tunggal)",
    "rapat": "Rapat simulasi LibriVox",
}


def evaluate_asr(
    config: AppConfig, items: list[EvalItem], output_dir: Path
) -> dict[str, dict[str, Any]]:
    """Transcribe each item and score it per subset."""
    references: dict[str, list[str]] = defaultdict(list)
    hypotheses: dict[str, list[str]] = defaultdict(list)
    audio_seconds: dict[str, float] = defaultdict(float)
    elapsed: dict[str, float] = defaultdict(float)

    with Transcriber(config.asr, config.vad) as transcriber:
        transcriber.load()
        for number, item in enumerate(items, start=1):
            assert item.audio is not None and item.transcript is not None
            logger.info("ASR %d/%d: %s", number, len(items), item.item_id)
            work_dir = config.paths.temp_dir / f"eval-{item.item_id}"
            try:
                audio = preprocess_audio(item.audio, config.audio, config.vad, work_dir)
                started = time.perf_counter()
                transcript = transcriber.transcribe(audio)
                elapsed[item.subset] += time.perf_counter() - started
            finally:
                shutil.rmtree(work_dir, ignore_errors=True)
            audio_seconds[item.subset] += audio.duration

            transcript.save_json(output_dir / "asr" / f"{item.item_id}.json")
            references[item.subset].append(reference_text(item.transcript))
            hypotheses[item.subset].append(transcript.text)

    results: dict[str, dict[str, Any]] = {}
    for subset in sorted(references):
        scores = score_asr(references[subset], hypotheses[subset])
        results[subset] = {
            **asdict(scores),
            "audio_seconds": round(audio_seconds[subset], 1),
            "real_time_factor": elapsed[subset] / audio_seconds[subset],
        }
        logger.info("ASR %s: WER %.3f, CER %.3f", subset, scores.wer, scores.cer)
    return results


def evaluate_diarization(
    config: AppConfig, items: list[EvalItem], output_dir: Path
) -> dict[str, DerScores]:
    """Diarize each item with automatic and known speaker counts."""
    runs: dict[str, list[Any]] = {"auto": [], "oracle": []}
    with Diarizer(config.diarization, config.secrets.hf_token) as diarizer:
        for number, item in enumerate(items, start=1):
            assert item.audio is not None and item.rttm is not None
            logger.info("DER %d/%d: %s", number, len(items), item.item_id)
            reference = read_rttm(item.rttm)
            speakers = len({turn.speaker for turn in reference})
            for mode, count in (("auto", None), ("oracle", speakers)):
                hypothesis = diarizer.diarize(item.audio, num_speakers=count)
                write_rttm(
                    hypothesis, output_dir / "rttm" / f"{item.item_id}_{mode}.rttm", item.item_id
                )
                runs[mode].append((item.item_id, reference, hypothesis))

    collar = config.evaluation.der_collar_seconds
    results = {mode: score_diarization(files, collar) for mode, files in runs.items()}
    for mode, scores in results.items():
        logger.info("DER %s: %.3f", mode, scores.der)
    return results


def evaluate_summaries(
    config: AppConfig, items: list[EvalItem], output_dir: Path
) -> tuple[list[dict[str, Any]], ActionScores]:
    """Summarize each reference transcript and compare with the reference summary."""
    predictions: list[Summary] = []
    references: list[Summary] = []
    with create_llm_client(config.llm, config.secrets) as client:
        for number, item in enumerate(items, start=1):
            assert item.transcript is not None and item.summary is not None
            logger.info("Summary %d/%d: %s", number, len(items), item.item_id)
            summary = summarize_transcript(
                transcript_from_text(item.transcript), client, config.summarize
            )
            path = output_dir / "summaries" / f"{item.item_id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
            predictions.append(summary)
            references.append(load_summary(item.summary))

    reference_texts = [summary_text(summary) for summary in references]
    predicted_texts = [summary_text(summary) for summary in predictions]
    rouge = rouge_l(reference_texts, predicted_texts)
    bert: list[float | None] = [None] * len(items)
    if config.evaluation.bertscore_enabled:
        try:
            bert = list(
                bertscore_f1(reference_texts, predicted_texts, config.evaluation.bertscore_model)
            )
        except (OSError, RuntimeError, ValueError) as exc:
            # Usually a failed model download; the other metrics are still worth reporting.
            logger.error("BERTScore unavailable, reporting it as missing: %s", exc)

    rows = [
        {"item": item.item_id, "rouge_l": rouge[index], "bertscore": bert[index]}
        for index, item in enumerate(items)
    ]
    actions = score_action_items(
        [
            (prediction.tindak_lanjut, reference.tindak_lanjut)
            for prediction, reference in zip(predictions, references, strict=True)
        ],
        config.evaluation.action_match_threshold,
    )
    return rows, actions


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def _number(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _device_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return platform.processor() or "CPU"


def render_report(config: AppConfig, results: dict[str, Any]) -> str:
    """Render the results as a Markdown report."""
    lines = [
        "# Hasil evaluasi",
        "",
        f"Dibuat oleh `scripts/evaluate.py` pada {date.today().isoformat()} di {_device_name()}.",
        "",
        f"- ASR: faster-whisper `{config.asr.model}` ({config.asr.compute_type}, "
        f"{config.asr.device}), bahasa `{config.asr.language}`",
        f"- Diarization: `{config.diarization.pipeline}`",
        f"- LLM: `{config.llm.model}` lewat {config.llm.backend}",
        "",
    ]

    if "asr" in results:
        lines += [
            "## ASR",
            "",
            "Teks dinormalisasi sebelum dinilai: huruf kecil, tanpa tanda baca, dan tanda hubung "
            'menjadi spasi. Angka tidak dieja, jadi "60" melawan "enam puluh" dihitung salah.',
            "",
            "| Subset | Ucapan | Kata referensi | WER | CER | Substitusi | Hapus | Sisip | RTF |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for subset, row in results["asr"].items():
            lines.append(
                f"| {SUBSET_NAMES.get(subset, subset)} | {row['utterances']} "
                f"| {row['reference_words']} | {_percent(row['wer'])} | {_percent(row['cer'])} "
                f"| {row['substitutions']} | {row['deletions']} | {row['insertions']} "
                f"| {row['real_time_factor']:.3f} |"
            )
        lines.append("")

    if "der" in results:
        collar = config.evaluation.der_collar_seconds
        lines += [
            "## Diarization",
            "",
            f"DER dengan collar {collar} detik di tiap batas giliran, "
            "termasuk area tumpang tindih.",
            "",
            "| Jumlah pembicara | File | DER | Salah pembicara | Terlewat | Alarm palsu |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        labels = {"auto": "Dideteksi otomatis", "oracle": "Diberikan (benar)"}
        for mode, scores in results["der"].items():
            total = scores["total"] or 1.0
            lines.append(
                f"| {labels[mode]} | {scores['files']} | {_percent(scores['der'])} "
                f"| {_percent(scores['confusion'] / total)} | {_percent(scores['missed'] / total)} "
                f"| {_percent(scores['false_alarm'] / total)} |"
            )
        lines.append("")

    if "summary" in results:
        rows = results["summary"]
        lines += [
            "## Ringkasan",
            "",
            "LLM membaca transkrip referensi, jadi skor ini mengukur peringkasan saja tanpa "
            "kesalahan ASR. Teks yang dibandingkan adalah topik, keputusan, dan poin penting.",
            "",
            "| Dokumen | ROUGE-L F1 | BERTScore F1 |",
            "| --- | ---: | ---: |",
        ]
        for row in rows:
            lines.append(
                f"| {row['item']} | {_number(row['rouge_l'])} | {_number(row['bertscore'])} |"
            )
        bert_values = [row["bertscore"] for row in rows if row["bertscore"] is not None]
        mean_rouge = sum(row["rouge_l"] for row in rows) / len(rows)
        mean_bert = sum(bert_values) / len(bert_values) if bert_values else None
        lines += [f"| **Rata-rata** | **{_number(mean_rouge)}** | **{_number(mean_bert)}** |", ""]

        actions = results["actions"]
        lines += [
            "## Tindak lanjut",
            "",
            f"Item dianggap cocok jika token F1 tugas atau kutipannya minimal "
            f"{config.evaluation.action_match_threshold}. Pencocokan satu lawan satu.",
            "",
            "| Precision | Recall | F1 | Prediksi | Referensi | Cocok | Penanggung jawab benar |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| {_percent(actions['precision'])} | {_percent(actions['recall'])} "
            f"| {_percent(actions['f1'])} | {actions['predicted']} | {actions['reference']} "
            f"| {actions['matched']} | {_percent(actions['owner_accuracy'])} |",
            "",
        ]

    lines += [
        "## Catatan metodologi",
        "",
        "- FLEURS dan LibriVox berisi ucapan yang dibacakan, bukan percakapan spontan. "
        "Rapat sungguhan biasanya lebih sulit: ada jeda ragu, kata gaul, dan bising.",
        "- Rapat simulasi tidak memiliki ucapan yang tumpang tindih, sehingga DER di sini "
        "cenderung lebih rendah daripada rapat sungguhan.",
        "- Rapat simulasi LibriVox hanya dipakai untuk DER. Transkripnya memakai ejaan lama "
        '(misalnya "itoe" dan "djadi"), sehingga WER-nya akan menghitung perbedaan ejaan '
        "sebagai kesalahan.",
        "- Ringkasan referensi ditulis manual untuk naskah rapat buatan, jadi jumlah dokumennya "
        "kecil dan angka ringkasan hanya indikatif.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate the pipeline on the test set.")
    parser.add_argument("--skip-asr", action="store_true", help="Skip WER and CER.")
    parser.add_argument("--skip-der", action="store_true", help="Skip DER.")
    parser.add_argument("--skip-summary", action="store_true", help="Skip summary metrics.")
    parser.add_argument("--limit", type=int, help="Use at most this many items per stage.")
    return parser.parse_args()


def main() -> int:
    """Entry point."""
    args = parse_args()
    config = load_config()
    setup_logging(config.logging.level)
    evaluation = config.evaluation

    items = discover(evaluation.test_set_dir)
    if not items:
        logger.error(
            "No test items in %s. Run: python scripts/build_test_set.py", evaluation.test_set_dir
        )
        return 1
    output_dir = evaluation.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    def pick(condition: Any) -> list[EvalItem]:
        selected = [item for item in items if condition(item)]
        return selected[: args.limit] if args.limit else selected

    results: dict[str, Any] = {}
    asr_items = pick(
        lambda item: (
            item.audio and item.transcript and item.subset not in evaluation.wer_exclude_subsets
        )
    )
    if not args.skip_asr and asr_items:
        results["asr"] = evaluate_asr(config, asr_items, output_dir)

    der_items = pick(lambda item: item.audio and item.rttm)
    if not args.skip_der and der_items:
        results["der"] = {
            mode: asdict(scores)
            for mode, scores in evaluate_diarization(config, der_items, output_dir).items()
        }

    summary_items = pick(lambda item: item.transcript and item.summary)
    if not args.skip_summary and summary_items:
        rows, actions = evaluate_summaries(config, summary_items, output_dir)
        results["summary"] = rows
        results["actions"] = asdict(actions)

    if not results:
        logger.error("Nothing was evaluated. Check the test set and the --skip options.")
        return 1

    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    report = evaluation.results_path
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_report(config, results), encoding="utf-8")
    logger.info("Report written to %s", report.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
