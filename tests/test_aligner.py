"""Tests for word-to-speaker alignment, using synthetic data."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from notulen.config import MergeConfig
from notulen.merge.aligner import align_transcript
from notulen.schemas import Segment, SpeakerTurn, Transcript, Word

CONFIG = MergeConfig(max_gap_seconds=1.0, speaker_label_format="Pembicara {index}")


def word(text: str, start: float, end: float) -> Word:
    return Word(text=text, start=start, end=end, probability=0.9)


def turn(speaker: str, start: float, end: float) -> SpeakerTurn:
    return SpeakerTurn(start=start, end=end, speaker=speaker)


def transcript(*words: Word) -> Transcript:
    segment = Segment(
        start=words[0].start,
        end=words[-1].end,
        text=" ".join(item.text for item in words),
        words=words,
    )
    return Transcript(
        source="rapat.mp3", language="id", model="medium", duration=60.0, segments=(segment,)
    )


def speakers_of(result: Transcript) -> list[str | None]:
    return [segment.speaker for segment in result.segments]


def test_word_goes_to_speaker_with_largest_overlap() -> None:
    result = align_transcript(
        transcript(word("halo", 0.0, 1.0)), [turn("A", 0.0, 0.3), turn("B", 0.3, 2.0)], CONFIG
    )

    assert result.words[0].speaker == "Pembicara 1"
    assert speakers_of(result) == ["Pembicara 1"]
    assert len(result.segments) == 1


def test_overlap_is_summed_across_turns_of_one_speaker() -> None:
    turns = [turn("A", 0.0, 1.5), turn("A", 1.5, 3.0), turn("B", 3.0, 5.0)]

    result = align_transcript(
        transcript(word("panjang", 0.0, 5.0), word("lagi", 5.5, 5.8)), turns, CONFIG
    )

    assert result.segments[0].text == "panjang"
    assert result.segments[0].speaker == "Pembicara 1"
    assert result.segments[1].text == "lagi"
    assert result.segments[1].speaker == "Pembicara 2"


def test_equal_overlap_goes_to_earlier_turn() -> None:
    result = align_transcript(
        transcript(word("sama", 0.0, 2.0), word("kedua", 2.0, 3.0)),
        [turn("A", 0.0, 1.0), turn("B", 1.0, 3.0)],
        CONFIG,
    )

    assert [segment.text for segment in result.segments] == ["sama", "kedua"]


def test_overlapping_turns_are_all_considered() -> None:
    turns = [turn("A", 0.0, 100.0), turn("B", 10.0, 11.0), turn("C", 50.0, 51.0)]

    result = align_transcript(transcript(word("akhir", 90.0, 91.0)), turns, CONFIG)

    assert result.segments[0].speaker == "Pembicara 1"
    assert result.segments[0].text == "akhir"


def test_word_in_a_gap_goes_to_nearest_turn_within_limit() -> None:
    result = align_transcript(
        transcript(word("dekat", 2.8, 3.0)), [turn("A", 0.0, 1.0), turn("B", 3.5, 5.0)], CONFIG
    )

    assert result.segments[0].speaker == "Pembicara 1"
    assert result.words[0].speaker == "Pembicara 1"


def test_nearest_turn_selection_prefers_the_closer_speaker() -> None:
    words = (word("awal", 3.2, 3.4), word("jeda", 1.5, 1.6))

    result = align_transcript(
        transcript(*sorted(words, key=lambda item: item.start)),
        [turn("A", 0.0, 1.0), turn("B", 3.5, 5.0)],
        CONFIG,
    )

    assert [segment.text for segment in result.segments] == ["jeda", "awal"]
    assert speakers_of(result) == ["Pembicara 1", "Pembicara 2"]


def test_word_beyond_the_limit_has_no_speaker() -> None:
    result = align_transcript(transcript(word("jauh", 10.0, 10.5)), [turn("A", 0.0, 1.0)], CONFIG)

    assert speakers_of(result) == [None]
    assert result.words[0].speaker is None


def test_zero_length_word_inside_a_turn_is_assigned() -> None:
    result = align_transcript(transcript(word("ya", 1.0, 1.0)), [turn("A", 0.0, 2.0)], CONFIG)

    assert speakers_of(result) == ["Pembicara 1"]


def test_consecutive_words_from_one_speaker_are_merged() -> None:
    result = align_transcript(
        transcript(word("selamat", 0.0, 0.5), word("pagi", 0.5, 1.0), word("semua", 1.0, 1.5)),
        [turn("A", 0.0, 2.0)],
        CONFIG,
    )

    assert len(result.segments) == 1
    utterance = result.segments[0]
    assert utterance.text == "selamat pagi semua"
    assert (utterance.start, utterance.end) == (0.0, 1.5)
    assert len(utterance.words) == 3


def test_utterances_merge_across_asr_segment_boundaries() -> None:
    first = Segment(start=0.0, end=0.5, text="satu", words=(word("satu", 0.0, 0.5),))
    second = Segment(start=0.6, end=1.0, text="dua", words=(word("dua", 0.6, 1.0),))
    source = Transcript(
        source="rapat.mp3", language="id", model="medium", duration=2.0, segments=(first, second)
    )

    result = align_transcript(source, [turn("A", 0.0, 2.0)], CONFIG)

    assert [segment.text for segment in result.segments] == ["satu dua"]


def test_speaker_change_splits_a_single_asr_segment() -> None:
    words = (word("apa", 0.0, 0.5), word("kabar", 0.5, 1.0), word("baik", 1.2, 1.6))

    result = align_transcript(
        transcript(*words), [turn("A", 0.0, 1.1), turn("B", 1.1, 2.0)], CONFIG
    )

    assert [segment.text for segment in result.segments] == ["apa kabar", "baik"]
    assert speakers_of(result) == ["Pembicara 1", "Pembicara 2"]


def test_returning_speaker_starts_a_new_utterance_with_the_same_label() -> None:
    words = (word("satu", 0.0, 0.5), word("dua", 1.0, 1.5), word("tiga", 2.0, 2.5))
    turns = [turn("A", 0.0, 0.8), turn("B", 0.8, 1.8), turn("A", 1.8, 3.0)]

    result = align_transcript(transcript(*words), turns, CONFIG)

    assert speakers_of(result) == ["Pembicara 1", "Pembicara 2", "Pembicara 1"]


def test_labels_follow_speaking_order_and_skip_silent_speakers() -> None:
    words = (word("satu", 10.0, 10.5), word("dua", 20.0, 20.5))
    turns = [turn("SILENT", 0.0, 1.0), turn("Z", 10.0, 11.0), turn("A", 20.0, 21.0)]

    result = align_transcript(transcript(*words), turns, CONFIG)

    assert speakers_of(result) == ["Pembicara 1", "Pembicara 2"]
    assert [segment.text for segment in result.segments] == ["satu", "dua"]


def test_unsorted_turns_are_handled() -> None:
    result = align_transcript(
        transcript(word("satu", 0.0, 0.5), word("dua", 5.0, 5.5)),
        [turn("B", 5.0, 6.0), turn("A", 0.0, 1.0)],
        CONFIG,
    )

    assert speakers_of(result) == ["Pembicara 1", "Pembicara 2"]


def test_segment_without_words_is_assigned_as_a_whole() -> None:
    source = Transcript(
        source="rapat.mp3",
        language="id",
        model="medium",
        duration=5.0,
        segments=(Segment(start=1.0, end=3.0, text="tanpa timestamp kata"),),
    )

    result = align_transcript(source, [turn("A", 0.0, 1.5), turn("B", 1.5, 4.0)], CONFIG)

    assert result.segments[0].text == "tanpa timestamp kata"
    assert result.segments[0].speaker == "Pembicara 1"
    assert result.segments[0].words == ()


def test_no_turns_leaves_words_unassigned() -> None:
    result = align_transcript(transcript(word("halo", 0.0, 0.5)), [], CONFIG)

    assert speakers_of(result) == [None]


def test_empty_transcript_stays_empty() -> None:
    source = Transcript(source="rapat.mp3", language="id", model="medium", duration=1.0)

    assert align_transcript(source, [turn("A", 0.0, 1.0)], CONFIG).segments == ()


def test_metadata_and_input_are_preserved() -> None:
    source = transcript(word("halo", 0.0, 0.5))

    result = align_transcript(source, [turn("A", 0.0, 1.0)], CONFIG)

    assert (result.source, result.language, result.model, result.duration) == (
        source.source,
        source.language,
        source.model,
        source.duration,
    )
    assert source.segments[0].speaker is None
    assert source.words[0].speaker is None


def test_label_format_must_contain_index() -> None:
    with pytest.raises(ValidationError):
        MergeConfig(max_gap_seconds=1.0, speaker_label_format="Pembicara")
