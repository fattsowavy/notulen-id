"""Tests for evaluation metrics, test set loading, and the reference data."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from notulen.config import load_config
from notulen.evaluation.action_metrics import (
    match_items,
    same_owner,
    score_action_items,
    token_f1,
)
from notulen.evaluation.asr_metrics import normalize_text, score_asr
from notulen.evaluation.diarization_metrics import score_diarization
from notulen.evaluation.summary_metrics import rouge_l, summary_text
from notulen.evaluation.test_set import (
    discover,
    load_summary,
    read_lines,
    reference_text,
    transcript_from_text,
)
from notulen.schemas import ActionItem, SpeakerTurn, Summary
from notulen.summarize.summarizer import QuoteLocator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_SET = PROJECT_ROOT / "data" / "test_set"


def load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclasses look up their module in sys.modules while being defined.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def item(task: str, quote: str, owner: str | None = None) -> ActionItem:
    return ActionItem(tugas=task, penanggung_jawab=owner, kutipan_sumber=quote, timestamp=0.0)


def turn(speaker: str, start: float, end: float) -> SpeakerTurn:
    return SpeakerTurn(start=start, end=end, speaker=speaker)


def test_normalize_text() -> None:
    assert normalize_text("Orang-orang  datang, YA?") == "orang orang datang ya"
    assert normalize_text("Kafé 60%") == normalize_text("kafé 60")


def test_score_asr_counts_errors_after_normalization() -> None:
    scores = score_asr(["Saya pergi ke pasar.", "Halo"], ["saya pergi pasar", "halo dunia"])

    assert (scores.substitutions, scores.deletions, scores.insertions) == (0, 1, 1)
    assert scores.reference_words == 5
    assert scores.wer == pytest.approx(2 / 5)


def test_score_asr_skips_empty_references_and_checks_lengths() -> None:
    assert score_asr(["", "halo"], ["apa", "halo"]).utterances == 1
    with pytest.raises(ValueError):
        score_asr(["halo"], [])
    with pytest.raises(ValueError):
        score_asr([" "], ["halo"])


def test_der_is_zero_for_a_perfect_hypothesis_with_other_labels() -> None:
    reference = [turn("A", 0, 5), turn("B", 5, 10)]
    hypothesis = [turn("x", 0, 5), turn("y", 5, 10)]

    scores = score_diarization([("f", reference, hypothesis)], collar=0.0)

    assert scores.der == pytest.approx(0.0)
    assert scores.total == pytest.approx(10.0)


def test_der_components() -> None:
    reference = [turn("A", 0, 5), turn("B", 5, 10)]
    hypothesis = [turn("x", 0, 10), turn("y", 10, 12)]

    scores = score_diarization([("f", reference, hypothesis)], collar=0.0)

    assert scores.confusion == pytest.approx(5.0)
    assert scores.false_alarm == pytest.approx(2.0)
    assert scores.missed == pytest.approx(0.0)
    assert scores.der == pytest.approx(0.7)


def test_der_aggregates_over_files() -> None:
    perfect = ("a", [turn("A", 0, 10)], [turn("x", 0, 10)])
    half = ("b", [turn("A", 0, 10)], [turn("x", 0, 5)])

    scores = score_diarization([perfect, half], collar=0.0)

    assert scores.files == 2
    assert scores.der == pytest.approx(5 / 20)


def test_token_f1() -> None:
    assert token_f1("kirim laporan", "Kirim laporan!") == 1.0
    assert token_f1("kirim laporan", "pesan ruangan") == 0.0
    assert token_f1("kirim laporan keuangan", "kirim laporan") == pytest.approx(0.8)


def test_items_match_one_to_one() -> None:
    predicted = [item("Kirim laporan keuangan", "a"), item("Kirim laporan", "b")]
    reference = [item("Kirim laporan keuangan", "c")]

    assert match_items(predicted, reference, 0.5) == [(0, 0)]


def test_items_match_on_quotes_when_tasks_differ() -> None:
    predicted = [item("Urus alat", "tolong pinjam cangkul dari RT enam")]
    reference = [item("Meminjam peralatan", "Pak Joko, tolong pinjam cangkul dari RT enam")]

    assert match_items(predicted, reference, 0.5) == [(0, 0)]


def test_same_owner() -> None:
    assert same_owner("Bu Sari", "Sari")
    assert same_owner("Pembicara 2", "pembicara 2")
    assert not same_owner("Pembicara 1", "Pembicara 2")
    assert not same_owner("Sari", None)
    assert same_owner(None, None)


def test_action_scores() -> None:
    documents = [
        (
            [item("Kirim laporan", "a", "Rina"), item("Karangan", "b")],
            [item("Kirim laporan", "c", "Bu Rina"), item("Perbaiki bug", "d")],
        ),
        ([], []),
    ]

    scores = score_action_items(documents, 0.5)

    assert (scores.predicted, scores.reference, scores.matched) == (2, 2, 1)
    assert scores.precision == scores.recall == scores.f1 == 0.5
    assert scores.owner_accuracy == 1.0


@pytest.mark.parametrize(
    ("predicted", "reference", "expected"), [(0, 0, 1.0), (1, 0, 0.0), (0, 1, 0.0)]
)
def test_action_scores_with_empty_lists(predicted: int, reference: int, expected: float) -> None:
    documents = [([item("A", "a")] * predicted, [item("B", "b")] * reference)]

    assert score_action_items(documents, 0.5).f1 == expected


def test_summary_text_excludes_action_items() -> None:
    summary = Summary(
        topik=("Anggaran",),
        keputusan=("Setuju",),
        tindak_lanjut=(item("Kirim laporan", "kirim"),),
    )

    assert summary_text(summary) == "Anggaran\nSetuju"


def test_rouge_l() -> None:
    scores = rouge_l(["anggaran naik lima persen", "x"], ["anggaran naik lima persen", "y"])

    assert scores == [pytest.approx(1.0), 0.0]


def test_discover_and_read(tmp_path: Path) -> None:
    for folder in ("audio", "transcripts", "rttm", "summaries"):
        (tmp_path / folder).mkdir()
    (tmp_path / "audio" / "rapat_00.wav").write_bytes(b"")
    (tmp_path / "transcripts" / "rapat_00.txt").write_text(
        "Pembicara 1: Halo semua.\n\nPembicara 2: Pagi.\n", encoding="utf-8"
    )
    (tmp_path / "rttm" / "rapat_00.rttm").write_text("", encoding="utf-8")
    (tmp_path / "summaries" / "naskah_09.json").write_text("{}", encoding="utf-8")

    items = discover(tmp_path)

    assert [entry.item_id for entry in items] == ["naskah_09", "rapat_00"]
    meeting = items[1]
    assert meeting.subset == "rapat"
    assert meeting.audio is not None and meeting.rttm is not None and meeting.summary is None
    assert items[0].transcript is None
    assert meeting.transcript is not None
    assert read_lines(meeting.transcript) == [
        ("Pembicara 1", "Halo semua."),
        ("Pembicara 2", "Pagi."),
    ]
    assert reference_text(meeting.transcript) == "Halo semua. Pagi."


def test_transcript_from_text_keeps_speakers_and_order(tmp_path: Path) -> None:
    path = tmp_path / "naskah.txt"
    path.write_text("Pembicara 1: satu dua\nPembicara 2: tiga\n", encoding="utf-8")

    transcript = transcript_from_text(path)

    assert [segment.speaker for segment in transcript.segments] == ["Pembicara 1", "Pembicara 2"]
    assert transcript.segments[1].start > transcript.segments[0].end
    assert transcript.text == "satu dua tiga"


def test_reference_summaries_are_valid_and_quotes_are_verbatim() -> None:
    summaries = sorted((TEST_SET / "summaries").glob("*.json"))
    assert summaries, "reference summaries are missing"
    for path in summaries:
        summary = load_summary(path)
        transcript = TEST_SET / "transcripts" / f"{path.stem}.txt"
        locator = QuoteLocator(transcript_from_text(transcript).segments)
        for action in summary.tindak_lanjut:
            assert locator.locate(action.kutipan_sumber, 1.0) is not None, (
                f"{path.name}: quote not found: {action.kutipan_sumber}"
            )


def test_planned_meetings_alternate_speakers_without_reusing_utterances() -> None:
    builder = load_script("build_test_set")
    by_reader = {
        reader: [builder.Clip(f"{reader}_{index}.mp3", "teks", reader) for index in range(size)]
        for reader, size in (("a", 40), ("b", 7), ("c", 6), ("d", 5), ("e", 3))
    }
    build = load_config(env_path=Path("missing.env")).evaluation.build.model_copy(
        update={"meetings": 6, "turns_per_meeting": 12, "speakers_per_meeting": (2, 3)}
    )

    meetings = builder.plan_meetings(by_reader, build)

    clips = [clip for meeting in meetings for clip in meeting]
    assert meetings
    assert len(clips) == len({clip.archive_name for clip in clips})
    for meeting in meetings:
        speakers = [clip.speaker for clip in meeting]
        assert len(set(speakers)) >= 2
        assert all(first != second for first, second in zip(speakers, speakers[1:], strict=False))
        assert len(meeting) <= build.turns_per_meeting


def test_fix_mojibake_repairs_double_decoded_text_only() -> None:
    builder = load_script("build_test_set")

    assert builder.fix_mojibake("(1000â€“1300 M)") == "(1000–1300 M)"
    assert builder.fix_mojibake("AsunciÃ³n") == "Asunción"
    assert builder.fix_mojibake("Asunción berdiri") == "Asunción berdiri"
    assert builder.fix_mojibake("teks biasa") == "teks biasa"


def test_fleurs_references_have_no_mojibake() -> None:
    builder = load_script("build_test_set")
    for path in sorted((TEST_SET / "transcripts").glob("fleurs_*.txt")):
        text = path.read_text(encoding="utf-8")
        assert builder.fix_mojibake(text) == text, path.name


def test_report_renders_every_section(tmp_path: Path) -> None:
    evaluate = load_script("evaluate")
    config = load_config(env_path=tmp_path / "missing.env")
    results = {
        "asr": {
            "fleurs": {
                "wer": 0.1,
                "cer": 0.05,
                "substitutions": 3,
                "deletions": 1,
                "insertions": 0,
                "reference_words": 40,
                "utterances": 4,
                "audio_seconds": 30.0,
                "real_time_factor": 0.12,
            }
        },
        "der": {
            mode: {
                "der": 0.08,
                "confusion": 1.0,
                "missed": 2.0,
                "false_alarm": 1.0,
                "total": 50.0,
                "files": 2,
            }
            for mode in ("auto", "oracle")
        },
        "summary": [{"item": "naskah_01", "rouge_l": 0.4, "bertscore": None}],
        "actions": {
            "precision": 0.75,
            "recall": 1.0,
            "f1": 0.857,
            "predicted": 4,
            "reference": 3,
            "matched": 3,
            "owner_accuracy": None,
        },
    }

    report = evaluate.render_report(config, results)

    assert "| FLEURS (ucapan tunggal) | 4 | 40 | 10.0% | 5.0% |" in report
    assert "| Dideteksi otomatis | 2 | 8.0% | 2.0% | 4.0% | 2.0% |" in report
    assert "| naskah_01 | 0.400 | - |" in report
    assert "| 75.0% | 100.0% | 85.7% | 4 | 3 | 3 | - |" in report
