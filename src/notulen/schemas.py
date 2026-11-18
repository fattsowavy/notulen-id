"""Data models shared by every pipeline stage."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Schema(BaseModel):
    """Immutable base model that trims strings and rejects unknown fields."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class TimeSpan(Schema):
    """A time range in seconds from the start of the recording."""

    start: float = Field(ge=0)
    end: float = Field(ge=0)

    @model_validator(mode="after")
    def check_order(self) -> TimeSpan:
        """Reject spans that end before they start."""
        if self.end < self.start:
            raise ValueError(f"end ({self.end}) must not be before start ({self.start})")
        return self


class Word(TimeSpan):
    """A recognized word."""

    text: str = Field(min_length=1)
    probability: float = Field(ge=0, le=1)
    speaker: str | None = None


class Segment(TimeSpan):
    """A continuous utterance."""

    text: str = Field(min_length=1)
    words: tuple[Word, ...] = ()
    speaker: str | None = None
    avg_logprob: float | None = None
    no_speech_prob: float | None = None


class Transcript(Schema):
    """A complete transcript of one recording."""

    source: str
    language: str
    model: str
    duration: float = Field(ge=0)
    segments: tuple[Segment, ...] = ()

    @property
    def text(self) -> str:
        """Return the full transcript as plain text."""
        return " ".join(segment.text for segment in self.segments)

    @property
    def words(self) -> tuple[Word, ...]:
        """Return every word in time order."""
        return tuple(word for segment in self.segments for word in segment.words)

    def save_json(self, path: Path) -> Path:
        """Write the transcript as UTF-8 JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def load_json(cls, path: Path) -> Transcript:
        """Read a transcript from a JSON file."""
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class ActionItem(Schema):
    """A follow-up task backed by a quote from the transcript."""

    tugas: str = Field(min_length=1)
    penanggung_jawab: str | None = None
    tenggat: str | None = None
    kutipan_sumber: str = Field(min_length=1)
    timestamp: float = Field(ge=0)


class Summary(Schema):
    """Structured minutes of a recording."""

    topik: tuple[str, ...] = ()
    keputusan: tuple[str, ...] = ()
    poin_penting: tuple[str, ...] = ()
    tindak_lanjut: tuple[ActionItem, ...] = ()
