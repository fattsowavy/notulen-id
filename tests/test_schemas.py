"""Tests for the shared data models."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from notulen.schemas import ActionItem, Segment, Transcript, Word


def make_word(text: str, start: float, end: float) -> Word:
    return Word(text=text, start=start, end=end, probability=0.9)


def test_word_rejects_end_before_start() -> None:
    with pytest.raises(ValidationError):
        make_word("halo", 2.0, 1.0)


def test_word_rejects_blank_text() -> None:
    with pytest.raises(ValidationError):
        make_word("   ", 0.0, 1.0)


def test_transcript_exposes_text_and_words() -> None:
    first = make_word("selamat", 0.0, 0.5)
    second = make_word("pagi", 0.5, 1.0)
    transcript = Transcript(
        source="rapat.mp3",
        language="id",
        model="medium",
        duration=1.0,
        segments=(Segment(start=0.0, end=1.0, text="selamat pagi", words=(first, second)),),
    )

    assert transcript.text == "selamat pagi"
    assert transcript.words == (first, second)


def test_transcript_json_round_trip_keeps_unicode(tmp_path: Path) -> None:
    transcript = Transcript(
        source="rapat.mp3",
        language="id",
        model="medium",
        duration=1.0,
        segments=(Segment(start=0.0, end=1.0, text="kafé déjà vu"),),
    )

    path = transcript.save_json(tmp_path / "nested" / "transcript.json")

    assert "kafé" in path.read_text(encoding="utf-8")
    assert Transcript.load_json(path) == transcript


@pytest.mark.parametrize("quote", ["", "   "])
def test_action_item_requires_source_quote(quote: str) -> None:
    with pytest.raises(ValidationError):
        ActionItem(tugas="Kirim laporan", kutipan_sumber=quote, timestamp=12.0)


def test_action_item_allows_unknown_owner_and_deadline() -> None:
    item = ActionItem(tugas="Kirim laporan", kutipan_sumber="tolong kirim laporan", timestamp=12.0)

    assert item.penanggung_jawab is None
    assert item.tenggat is None
