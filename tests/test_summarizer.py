"""Tests for map-reduce summarization and action item grounding."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from notulen.config import SummarizeConfig
from notulen.schemas import ActionItem, Segment, Transcript, Word
from notulen.summarize.llm_client import LlmError
from notulen.summarize.summarizer import (
    QuoteLocator,
    SummaryError,
    _ActionPart,
    _RawActionItem,
    _SummaryPart,
    ask_json,
    chunk_segments,
    ground_action_items,
    merge_duplicate_items,
    render_prompt,
    summarize_transcript,
)

CONFIG = SummarizeConfig(
    chunk_max_chars=12000,
    chunk_overlap_chars=500,
    require_action_item_quote=True,
    quote_match_threshold=0.8,
    json_attempts=2,
)


def utterance(text: str, start: float, speaker: str = "Pembicara 1") -> Segment:
    """Build a segment whose words are 0.5 s apart."""
    words = tuple(
        Word(text=token, start=start + 0.5 * index, end=start + 0.5 * index + 0.4, probability=0.9)
        for index, token in enumerate(text.split())
    )
    return Segment(start=words[0].start, end=words[-1].end, text=text, words=words, speaker=speaker)


def make_transcript(*segments: Segment) -> Transcript:
    return Transcript(
        source="rapat.mp3", language="id", model="medium", duration=600.0, segments=segments
    )


MEETING = make_transcript(
    utterance("Selamat pagi, kita bahas anggaran pemasaran.", 0.0),
    utterance("Baik. Saya akan kirim laporan keuangan sebelum hari Jumat.", 10.0, "Pembicara 2"),
    utterance("Oke, kita sepakat rapat lagi hari Selasa.", 20.0),
)


class ScriptedClient:
    """Answer prompts with a function and record every call."""

    def __init__(self, answer: Callable[[str], str]) -> None:
        self.answer = answer
        self.prompts: list[str] = []
        self.schemas: list[Any] = []

    def complete(self, prompt: str, system: str | None = None, json_schema: Any = None) -> str:
        self.prompts.append(prompt)
        self.schemas.append(json_schema)
        return self.answer(prompt)


def summary_json(**values: list[str]) -> str:
    return json.dumps({"topik": [], "keputusan": [], "poin_penting": [], **values})


def actions_json(*items: dict[str, Any]) -> str:
    return json.dumps({"tindak_lanjut": list(items)})


def is_action_prompt(prompt: str) -> bool:
    return '"tindak_lanjut"' in prompt


def is_reduce_prompt(prompt: str) -> bool:
    return "Ringkasan parsial" in prompt


def test_locator_finds_exact_quote_with_timestamp_and_original_text() -> None:
    locator = QuoteLocator(MEETING.segments)

    match = locator.locate("kirim laporan keuangan sebelum hari jumat", 0.8)

    assert match == (11.5, "kirim laporan keuangan sebelum hari Jumat.")


def test_locator_tolerates_small_differences() -> None:
    match = QuoteLocator(MEETING.segments).locate("saya akan mengirim laporan keuangan", 0.8)

    assert match is not None
    assert match[0] == 10.5


def test_locator_rejects_invented_quotes() -> None:
    locator = QuoteLocator(MEETING.segments)

    assert locator.locate("tolong pesan ruang rapat besar untuk minggu depan", 0.8) is None
    assert locator.locate("", 0.8) is None


def test_locator_handles_segments_without_words() -> None:
    segment = Segment(start=30.0, end=35.0, text="Andi akan menyiapkan presentasi")

    assert QuoteLocator([segment]).locate("menyiapkan presentasi", 0.8) == (
        30.0,
        "menyiapkan presentasi",
    )


def test_short_transcript_fits_in_one_chunk() -> None:
    assert chunk_segments(MEETING.segments, 12000, 500) == [list(MEETING.segments)]


def test_chunks_respect_the_limit_and_repeat_context() -> None:
    segments = [
        utterance(f"kalimat nomor {index} tentang anggaran", index * 5.0) for index in range(30)
    ]

    chunks = chunk_segments(segments, max_chars=300, overlap_chars=100)

    assert len(chunks) > 1
    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert current[0] in previous
    covered = {segment for chunk in chunks for segment in chunk}
    assert covered == set(segments)


def test_long_utterance_is_split_across_chunks() -> None:
    monologue = utterance(" ".join(f"kata{index}" for index in range(400)), 0.0)

    chunks = chunk_segments([monologue], max_chars=500, overlap_chars=0)

    assert len(chunks) > 1
    words = [word for chunk in chunks for segment in chunk for word in segment.words]
    assert words == list(monologue.words)


def test_grounding_rejects_missing_and_invented_quotes(caplog: pytest.LogCaptureFixture) -> None:
    raw = [
        _RawActionItem(tugas="Kirim laporan", kutipan_sumber="kirim laporan keuangan"),
        _RawActionItem(tugas="Tanpa kutipan", kutipan_sumber=None),
        _RawActionItem(tugas="Karangan", kutipan_sumber="pesan tiket pesawat ke Bali"),
        _RawActionItem(tugas="", kutipan_sumber="kirim laporan"),
    ]

    items = ground_action_items(raw, QuoteLocator(MEETING.segments), CONFIG, 0.0)

    assert [item.tugas for item in items] == ["Kirim laporan"]
    assert "without a source quote" in caplog.text
    assert "not in the transcript" in caplog.text


def test_grounding_can_keep_unmatched_quotes_when_configured() -> None:
    config = CONFIG.model_copy(update={"require_action_item_quote": False})
    raw = [_RawActionItem(tugas="Karangan", kutipan_sumber="pesan tiket pesawat ke Bali")]

    items = ground_action_items(raw, QuoteLocator(MEETING.segments), config, 42.0)

    assert items[0].kutipan_sumber == "pesan tiket pesawat ke Bali"
    assert items[0].timestamp == 42.0


def test_grounding_normalizes_placeholder_owner_and_deadline() -> None:
    raw = [
        _RawActionItem(
            tugas="Kirim laporan",
            penanggung_jawab=" tidak disebutkan ",
            tenggat="-",
            kutipan_sumber="kirim laporan keuangan",
        )
    ]

    item = ground_action_items(raw, QuoteLocator(MEETING.segments), CONFIG, 0.0)[0]

    assert item.penanggung_jawab is None
    assert item.tenggat is None


def test_summarize_single_chunk_without_reduce() -> None:
    def answer(prompt: str) -> str:
        if is_action_prompt(prompt):
            return actions_json(
                {
                    "tugas": "Kirim laporan keuangan",
                    "penanggung_jawab": "Pembicara 2",
                    "tenggat": "Jumat",
                    "kutipan_sumber": "Saya akan kirim laporan keuangan sebelum hari Jumat",
                },
                {"tugas": "Pesan ruangan", "kutipan_sumber": "tolong pesan ruangan besar"},
            )
        return summary_json(
            topik=["Anggaran pemasaran", " "], keputusan=["Rapat lagi hari Selasa."]
        )

    client = ScriptedClient(answer)
    summary = summarize_transcript(MEETING, client, CONFIG)  # type: ignore[arg-type]

    assert len(client.prompts) == 2
    assert not any(is_reduce_prompt(prompt) for prompt in client.prompts)
    assert summary.topik == ("Anggaran pemasaran",)
    assert summary.keputusan == ("Rapat lagi hari Selasa.",)
    assert len(summary.tindak_lanjut) == 1
    item = summary.tindak_lanjut[0]
    assert item.penanggung_jawab == "Pembicara 2"
    assert item.timestamp == 10.5
    assert item.kutipan_sumber == "Saya akan kirim laporan keuangan sebelum hari Jumat."


def test_summarize_long_transcript_reduces_and_dedupes() -> None:
    segments = [
        utterance(f"pembahasan bagian {index} soal anggaran", index * 5.0) for index in range(20)
    ]
    segments.append(utterance("Saya akan kirim laporan keuangan besok.", 200.0, "Pembicara 2"))
    transcript = make_transcript(*segments)
    config = CONFIG.model_copy(update={"chunk_max_chars": 400, "chunk_overlap_chars": 150})

    def answer(prompt: str) -> str:
        if is_reduce_prompt(prompt):
            return summary_json(topik=["Anggaran"])
        if is_action_prompt(prompt):
            if "kirim laporan" not in prompt:
                return actions_json()
            return actions_json(
                {"tugas": "Kirim laporan", "kutipan_sumber": "kirim laporan keuangan besok"}
            )
        return summary_json(topik=["Anggaran", "Laporan"])

    client = ScriptedClient(answer)
    summary = summarize_transcript(transcript, client, config)  # type: ignore[arg-type]

    chunk_count = len(chunk_segments(transcript.segments, 400, 150))
    assert chunk_count > 1
    assert len(client.prompts) == chunk_count * 2 + 1
    assert summary.topik == ("Anggaran",)
    assert len(summary.tindak_lanjut) == 1


def test_failed_reduce_falls_back_to_concatenation() -> None:
    segments = [
        utterance(f"pembahasan bagian {index} soal anggaran", index * 5.0) for index in range(20)
    ]
    config = CONFIG.model_copy(update={"chunk_max_chars": 400, "chunk_overlap_chars": 0})

    def answer(prompt: str) -> str:
        if is_reduce_prompt(prompt):
            return "bukan json"
        if is_action_prompt(prompt):
            return actions_json()
        return summary_json(topik=["Anggaran"], poin_penting=["Bagian dibahas"])

    summary = summarize_transcript(make_transcript(*segments), ScriptedClient(answer), config)  # type: ignore[arg-type]

    assert summary.topik == ("Anggaran",)
    assert summary.poin_penting == ("Bagian dibahas",)


def test_deadline_not_in_the_transcript_is_dropped(caplog: pytest.LogCaptureFixture) -> None:
    raw = [
        _RawActionItem(tugas="Kirim", tenggat="Jumat sore", kutipan_sumber="kirim laporan"),
        _RawActionItem(tugas="Kirim", tenggat="hari Jumat", kutipan_sumber="laporan keuangan"),
    ]

    items = ground_action_items(raw, QuoteLocator(MEETING.segments), CONFIG, 0.0)

    assert [item.tenggat for item in items] == [None, "hari Jumat"]
    assert "Jumat sore" in caplog.text


def make_item(
    task: str,
    quote: str,
    timestamp: float,
    owner: str | None = None,
    deadline: str | None = None,
) -> ActionItem:
    return ActionItem(
        tugas=task,
        penanggung_jawab=owner,
        tenggat=deadline,
        kutipan_sumber=quote,
        timestamp=timestamp,
    )


def merge(items: list[ActionItem], reply: str) -> tuple[list[ActionItem], ScriptedClient]:
    client = ScriptedClient(lambda prompt: reply)
    return merge_duplicate_items(client, items, CONFIG), client  # type: ignore[arg-type]


def test_duplicates_with_the_same_owner_are_merged() -> None:
    items = [
        make_item("Kirim laporan", "tolong kirim laporan", 1.0, "Bu Rina", "hari Jumat"),
        make_item("Kirim laporan", "siap saya kirim", 2.0, "Rina"),
        make_item("Perbaiki bug", "saya perbaiki bug", 3.0, "Budi"),
    ]

    kept, client = merge(items, '{"duplikat": [[1, 2]]}')

    assert kept == [items[0], items[2]]
    assert (
        "1. Tugas: Kirim laporan | Penanggung jawab: Bu Rina | Tenggat: hari Jumat | "
        'Kutipan: "tolong' in client.prompts[0]
    )


def test_merge_keeps_the_most_complete_item() -> None:
    items = [
        make_item("Kirim laporan", "tolong kirim laporan", 1.0, "Rina"),
        make_item("Kirim laporan", "siap saya kirim Jumat", 2.0, "Rina", "Jumat"),
    ]

    kept, _ = merge(items, '{"duplikat": [[1, 2]]}')

    assert kept == [items[1]]


@pytest.mark.parametrize(
    "owners",
    [("Pembicara 1", "Pembicara 2"), ("Rina", None), (None, None), ("Bu Rina", "Pak Budi")],
)
def test_groups_with_different_or_unknown_owners_are_ignored(
    owners: tuple[str | None, str | None],
) -> None:
    items = [
        make_item("Tugas A", "satu", 1.0, owners[0]),
        make_item("Tugas B", "dua", 2.0, owners[1]),
    ]

    kept, _ = merge(items, '{"duplikat": [[1, 2]]}')

    assert kept == items


@pytest.mark.parametrize(
    "reply", ['{"duplikat": []}', '{"duplikat": [[1], [7, 9], [0, 2]]}', "bukan json"]
)
def test_unusable_answers_keep_every_item(reply: str) -> None:
    items = [make_item("A", "satu", 1.0, "Rina"), make_item("B", "dua", 2.0, "Rina")]

    kept, _ = merge(items, reply)

    assert kept == items


def test_duplicate_tasks_across_chunks_are_merged_by_the_llm() -> None:
    segments = [
        utterance(f"pembahasan bagian {index} soal anggaran", index * 5.0) for index in range(12)
    ]
    segments.append(utterance("Rina tolong kirim laporan keuangan.", 100.0))
    segments.append(utterance("Siap, saya kirim laporannya besok.", 110.0, "Pembicara 2"))
    config = CONFIG.model_copy(update={"chunk_max_chars": 400, "chunk_overlap_chars": 0})

    def answer(prompt: str) -> str:
        if '"duplikat"' in prompt:
            return '{"duplikat": [[1, 2]]}'
        if is_reduce_prompt(prompt):
            return summary_json()
        if is_action_prompt(prompt) and "kirim laporan" in prompt:
            return actions_json(
                {
                    "tugas": "Kirim laporan",
                    "penanggung_jawab": "Rina",
                    "kutipan_sumber": "Rina tolong kirim laporan keuangan",
                },
                {
                    "tugas": "Kirim laporan",
                    "penanggung_jawab": "Pembicara 2 (Rina)",
                    "kutipan_sumber": "saya kirim laporannya besok",
                },
            )
        if is_action_prompt(prompt):
            return actions_json()
        return summary_json()

    client = ScriptedClient(answer)
    summary = summarize_transcript(make_transcript(*segments), client, config)  # type: ignore[arg-type]

    assert [item.kutipan_sumber for item in summary.tindak_lanjut] == [
        "Rina tolong kirim laporan keuangan."
    ]
    assert sum('"duplikat"' in prompt for prompt in client.prompts) == 1


def test_quote_joining_two_phrases_is_located_by_its_clauses() -> None:
    raw = [
        _RawActionItem(
            tugas="Perbaiki bug",
            kutipan_sumber="saya butuh waktu seminggu untuk memperbaikinya, "
            "kirim laporan keuangan sebelum hari Jumat",
        )
    ]

    items = ground_action_items(raw, QuoteLocator(MEETING.segments), CONFIG, 0.0)

    assert items[0].kutipan_sumber == "kirim laporan keuangan sebelum hari Jumat."
    assert items[0].timestamp == 11.5


def test_invalid_json_is_requested_again() -> None:
    replies = iter(["maaf, ini ringkasannya", '```json\n{"topik": ["Anggaran"]}\n```'])
    client = ScriptedClient(lambda prompt: next(replies))

    part = ask_json(client, "prompt", _SummaryPart, CONFIG)  # type: ignore[arg-type]

    assert part.topik == ["Anggaran"]
    assert len(client.prompts) == 2


def test_persistently_invalid_json_raises_an_llm_error() -> None:
    client = ScriptedClient(lambda prompt: '{"topik": "bukan daftar"}')

    with pytest.raises(SummaryError) as raised:
        ask_json(client, "prompt", _SummaryPart, CONFIG)  # type: ignore[arg-type]

    assert isinstance(raised.value, LlmError)
    assert len(client.prompts) == CONFIG.json_attempts


def test_null_fields_are_accepted() -> None:
    client = ScriptedClient(
        lambda prompt: '{"tindak_lanjut": [{"tugas": "Kirim", "penanggung_jawab": null}], "x": 1}'
    )

    part = ask_json(client, "prompt", _ActionPart, CONFIG)  # type: ignore[arg-type]

    assert part.tindak_lanjut[0].penanggung_jawab is None
    assert ScriptedClient(lambda prompt: '{"topik": null}').answer("") == '{"topik": null}'
    assert _SummaryPart.model_validate_json('{"topik": null}').topik == []


def test_schema_sent_to_the_llm_has_no_references() -> None:
    client = ScriptedClient(lambda prompt: actions_json())

    ask_json(client, "prompt", _ActionPart, CONFIG)  # type: ignore[arg-type]

    assert "$ref" not in json.dumps(client.schemas[0])
    assert "$defs" not in client.schemas[0]


def test_empty_transcript_needs_no_llm_call() -> None:
    client = ScriptedClient(lambda prompt: pytest.fail("LLM should not be called"))

    summary = summarize_transcript(make_transcript(), client, CONFIG)  # type: ignore[arg-type]

    assert summary.tindak_lanjut == ()


def test_render_prompt_keeps_braces_from_the_transcript() -> None:
    prompt = render_prompt("summary.md", transcript="[00:00:01] Pembicara 1: pakai {format} ini")

    assert "pakai {format} ini" in prompt
    assert "{transcript}" not in prompt
    assert '{"topik": ["..."]' in prompt
