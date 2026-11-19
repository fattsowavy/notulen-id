"""Tests for transcript and summary rendering."""

from __future__ import annotations

import pytest

from notulen.formatting import (
    format_timestamp,
    render_summary_markdown,
    render_transcript_markdown,
    transcript_to_prompt_text,
)
from notulen.schemas import ActionItem, Segment, Summary, Transcript


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0.0, "00:00:00"), (59.99, "00:00:59"), (192.4, "00:03:12"), (3725.0, "01:02:05")],
)
def test_format_timestamp(seconds: float, expected: str) -> None:
    assert format_timestamp(seconds) == expected


def make_transcript() -> Transcript:
    return Transcript(
        source="rapat.mp3",
        language="id",
        model="medium",
        duration=200.0,
        segments=(
            Segment(start=3.2, end=5.0, text="Selamat pagi.", speaker="Pembicara 1"),
            Segment(start=192.0, end=195.0, text="Baik, setuju."),
        ),
    )


def test_transcript_markdown_uses_speaker_and_timestamp() -> None:
    markdown = render_transcript_markdown(make_transcript())

    assert markdown.startswith("# Transkrip: rapat.mp3")
    assert "**Pembicara 1** [00:00:03]: Selamat pagi." in markdown
    assert "**Tidak diketahui** [00:03:12]: Baik, setuju." in markdown


def test_prompt_text_has_one_line_per_utterance() -> None:
    assert transcript_to_prompt_text(make_transcript()).splitlines() == [
        "[00:00:03] Pembicara 1: Selamat pagi.",
        "[00:03:12] Tidak diketahui: Baik, setuju.",
    ]


def test_summary_markdown_renders_every_section() -> None:
    summary = Summary(
        topik=("Anggaran",),
        tindak_lanjut=(
            ActionItem(
                tugas="Kirim laporan",
                penanggung_jawab="Budi",
                kutipan_sumber="tolong kirim | laporan",
                timestamp=75.0,
            ),
        ),
    )

    markdown = render_summary_markdown("rapat.mp3", summary)

    assert markdown.startswith("# Notulen: rapat.mp3")
    assert "Dibuat otomatis oleh LLM" in markdown
    assert "## Topik\n\n- Anggaran\n" in markdown
    assert "_Tidak ada keputusan yang jelas._" in markdown
    assert '| Kirim laporan | Budi | - | 00:01:15 | "tolong kirim \\| laporan" |' in markdown


def test_summary_markdown_without_action_items() -> None:
    markdown = render_summary_markdown("rapat.mp3", Summary())

    assert markdown.endswith("_Tidak ada tindak lanjut yang jelas._\n")
