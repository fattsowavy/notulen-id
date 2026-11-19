"""Map-reduce summarization with action items grounded in the transcript.

Each chunk of the transcript is summarized and searched for action items
separately (map). Partial summaries are then merged by the LLM (reduce).
Action items are not passed through the LLM again: their quotes are matched
against the transcript, which rejects invented items and gives each one the
timestamp of the words it quotes.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError, ValidationInfo, field_validator

from notulen.config import SummarizeConfig
from notulen.formatting import segment_prompt_line
from notulen.schemas import ActionItem, Segment, Summary, Transcript
from notulen.summarize.llm_client import LlmClient, LlmError

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"
EMPTY_VALUES = {"", "-", "null", "none", "n/a", "tidak ada", "tidak disebutkan", "tidak diketahui"}
CODE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")
CLAUSE_BOUNDARY = re.compile(r"[.!?;,]+\s+|\s+-\s+|\.\.\.")
MIN_CLAUSE_TOKENS = 4
HONORIFICS = {"pak", "bapak", "bu", "ibu", "mas", "mbak", "kak", "bang", "pembicara"}

ModelT = TypeVar("ModelT", bound=BaseModel)


class SummaryError(LlmError):
    """Raised when the LLM output cannot be turned into a summary."""


class _LlmOutput(BaseModel):
    """Lenient base for parsing LLM replies: unknown keys are ignored, nulls become empty."""

    model_config = ConfigDict(extra="ignore")

    @field_validator("*", mode="before")
    @classmethod
    def null_to_default(cls, value: Any, info: ValidationInfo) -> Any:
        """Replace null with the field's default when the default is not None."""
        if value is None and info.field_name is not None:
            default = cls.model_fields[info.field_name].default
            if default is not None:
                return copy.copy(default)
        return value


class _SummaryPart(_LlmOutput):
    topik: list[str] = []
    keputusan: list[str] = []
    poin_penting: list[str] = []


class _RawActionItem(_LlmOutput):
    tugas: str = ""
    penanggung_jawab: str | None = None
    tenggat: str | None = None
    kutipan_sumber: str | None = None


class _ActionPart(_LlmOutput):
    tindak_lanjut: list[_RawActionItem] = []


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens without punctuation."""
    return re.findall(r"\w+", text.lower())


@dataclass(frozen=True)
class _Token:
    text: str
    word_index: int


class QuoteLocator:
    """Find where a quote occurs in a set of transcript segments, tolerating small differences."""

    def __init__(self, segments: Sequence[Segment]) -> None:
        self._words: list[tuple[str, float]] = []
        self._tokens: list[_Token] = []
        for segment in segments:
            units = (
                [(word.text, word.start) for word in segment.words]
                if segment.words
                else [(text, segment.start) for text in segment.text.split()]
            )
            for text, start in units:
                index = len(self._words)
                self._words.append((text, start))
                self._tokens.extend(_Token(token, index) for token in tokenize(text))
        self._texts = [token.text for token in self._tokens]
        self._vocabulary = set(self._texts)

    def supports(self, text: str, threshold: float) -> bool:
        """Return whether enough of the words in text occur in the transcript."""
        tokens = tokenize(text)
        if not tokens:
            return False
        found = sum(token in self._vocabulary for token in tokens)
        return found / len(tokens) >= threshold

    def locate(self, quote: str, threshold: float) -> tuple[float, str] | None:
        """Return the start time and transcript text of the best match, if similar enough."""
        target = tokenize(quote)
        if not target or not self._tokens:
            return None

        size = min(len(target), len(self._texts))
        target_text = " ".join(target)
        target_set = set(target)
        best_score, best_index = 0.0, -1
        for index in range(len(self._texts) - size + 1):
            window = self._texts[index : index + size]
            if window == target:
                best_score, best_index = 1.0, index
                break
            # Cheap filter before the more expensive similarity computation.
            if len(target_set.intersection(window)) < threshold * len(target_set) / 2:
                continue
            score = SequenceMatcher(None, " ".join(window), target_text, autojunk=False).ratio()
            if score > best_score:
                best_score, best_index = score, index

        if best_index < 0 or best_score < threshold:
            return None
        first = self._tokens[best_index].word_index
        last = self._tokens[best_index + size - 1].word_index
        text = " ".join(word for word, _ in self._words[first : last + 1])
        return self._words[first][1], text


def _split_long_segment(segment: Segment, max_chars: int) -> list[Segment]:
    """Split an utterance whose prompt line would not fit in one chunk."""
    if len(segment_prompt_line(segment)) <= max_chars:
        return [segment]

    budget = max(max_chars // 2, 1)
    pieces: list[Segment] = []
    if segment.words:
        group: list[Any] = []
        for word in segment.words:
            if group and len(" ".join(item.text for item in [*group, word])) > budget:
                pieces.append(_segment_from_words(segment, group))
                group = []
            group.append(word)
        pieces.append(_segment_from_words(segment, group))
        return pieces

    text: list[str] = []
    for token in segment.text.split():
        if text and len(" ".join([*text, token])) > budget:
            pieces.append(segment.model_copy(update={"text": " ".join(text)}))
            text = []
        text.append(token)
    pieces.append(segment.model_copy(update={"text": " ".join(text)}))
    return pieces


def _segment_from_words(segment: Segment, words: Sequence[Any]) -> Segment:
    return segment.model_copy(
        update={
            "start": words[0].start,
            "end": words[-1].end,
            "text": " ".join(word.text for word in words),
            "words": tuple(words),
        }
    )


def chunk_segments(
    segments: Sequence[Segment], max_chars: int, overlap_chars: int
) -> list[list[Segment]]:
    """Group utterances into chunks of prompt text, repeating a little context between chunks."""
    pieces = [piece for segment in segments for piece in _split_long_segment(segment, max_chars)]

    chunks: list[list[Segment]] = []
    current: list[Segment] = []
    size = 0
    for piece in pieces:
        length = len(segment_prompt_line(piece)) + 1
        if current and size + length > max_chars:
            chunks.append(current)
            carried: list[Segment] = []
            carried_size = 0
            for previous in reversed(current):
                previous_length = len(segment_prompt_line(previous)) + 1
                if carried_size + previous_length > overlap_chars:
                    break
                carried.insert(0, previous)
                carried_size += previous_length
            current, size = carried, carried_size
        current.append(piece)
        size += length
    if current:
        chunks.append(current)
    return chunks


def load_prompt(name: str) -> str:
    """Read a prompt template from the prompts folder."""
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def render_prompt(name: str, **values: str) -> str:
    """Fill {placeholders} in a prompt template without interpreting other braces."""
    prompt = load_prompt(name)
    for key, value in values.items():
        prompt = prompt.replace("{" + key + "}", value)
    return prompt


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Replace $ref pointers with their definitions, since some backends do not resolve them."""
    definitions = schema.get("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(definitions[node["$ref"].rsplit("/", 1)[-1]])
            return {key: resolve(value) for key, value in node.items() if key != "$defs"}
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    resolved: dict[str, Any] = resolve(schema)
    return resolved


def ask_json(
    client: LlmClient, prompt: str, model: type[ModelT], config: SummarizeConfig
) -> ModelT:
    """Ask for JSON matching model, asking again when the reply does not validate."""
    schema = _inline_refs(model.model_json_schema())
    error: Exception | None = None
    for attempt in range(1, config.json_attempts + 1):
        reply = client.complete(prompt, json_schema=schema)
        try:
            return model.model_validate_json(CODE_FENCE.sub("", reply.strip()))
        except ValidationError as exc:
            error = exc
            logger.warning(
                "Invalid JSON from LLM (attempt %d/%d): %s",
                attempt,
                config.json_attempts,
                exc.errors()[0]["msg"],
            )
    raise SummaryError(
        f"LLM did not return valid JSON for {model.__name__} "
        f"after {config.json_attempts} attempt(s): {error}"
    )


def _optional(value: str | None) -> str | None:
    """Normalize placeholder answers such as "-" or "tidak disebutkan" to None."""
    if value is None:
        return None
    cleaned = value.strip()
    return None if cleaned.lower() in EMPTY_VALUES else cleaned


def _locate_quote(locator: QuoteLocator, quote: str, threshold: float) -> tuple[float, str] | None:
    """Locate the whole quote, or else its longest clause that is found.

    LLMs often join two separate phrases into one quote; each phrase still
    proves that the task was spoken.
    """
    match = locator.locate(quote, threshold)
    if match is not None:
        return match
    clauses = [
        clause
        for clause in CLAUSE_BOUNDARY.split(quote)
        if len(tokenize(clause)) >= MIN_CLAUSE_TOKENS
    ]
    for clause in sorted(clauses, key=lambda text: len(tokenize(text)), reverse=True):
        match = locator.locate(clause, threshold)
        if match is not None:
            return match
    return None


def ground_action_items(
    raw_items: Sequence[_RawActionItem],
    locator: QuoteLocator,
    config: SummarizeConfig,
    fallback_timestamp: float,
) -> list[ActionItem]:
    """Keep action items whose quote is found in the transcript, using the transcript's words."""
    items: list[ActionItem] = []
    for raw in raw_items:
        task = raw.tugas.strip()
        quote = (raw.kutipan_sumber or "").strip()
        if not task:
            logger.warning("Rejected action item without a task: %r", raw)
            continue
        if not quote:
            logger.warning("Rejected action item without a source quote: %s", task)
            continue

        match = _locate_quote(locator, quote, config.quote_match_threshold)
        if match is not None:
            timestamp, quote = match
        elif config.require_action_item_quote:
            logger.warning(
                "Rejected action item whose quote is not in the transcript: %s (quote: %r)",
                task,
                quote[:200],
            )
            continue
        else:
            timestamp = fallback_timestamp

        deadline = _optional(raw.tenggat)
        if deadline is not None and not locator.supports(deadline, config.quote_match_threshold):
            logger.warning("Dropped deadline %r that is not in the transcript: %s", deadline, task)
            deadline = None

        items.append(
            ActionItem(
                tugas=task,
                penanggung_jawab=_optional(raw.penanggung_jawab),
                tenggat=deadline,
                kutipan_sumber=quote,
                timestamp=round(timestamp, 3),
            )
        )
    return items


def _dedupe_items(items: Sequence[ActionItem]) -> list[ActionItem]:
    """Drop items that quote the same words, which happens where chunks overlap."""
    seen: set[str] = set()
    unique: list[ActionItem] = []
    for item in sorted(items, key=lambda entry: entry.timestamp):
        key = " ".join(tokenize(item.kutipan_sumber))
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


class _Duplicates(_LlmOutput):
    duplikat: list[list[int]] = []


def _describe_item(number: int, item: ActionItem) -> str:
    return (
        f"{number}. Tugas: {item.tugas} | Penanggung jawab: {item.penanggung_jawab or '-'} "
        f'| Tenggat: {item.tenggat or "-"} | Kutipan: "{item.kutipan_sumber}"'
    )


def _owner_tokens(item: ActionItem) -> set[str]:
    """Owner name tokens without honorifics, so "Bu Rina" matches "Rina"."""
    return set(tokenize(item.penanggung_jawab or "")) - HONORIFICS


def _completeness(item: ActionItem) -> int:
    return (item.penanggung_jawab is not None) + (item.tenggat is not None)


def merge_duplicate_items(
    client: LlmClient, items: Sequence[ActionItem], config: SummarizeConfig
) -> list[ActionItem]:
    """Merge items the LLM marks as the same task.

    The LLM only names groups of duplicates. A group is merged only when every
    item has the same known owner, and items outside any group are always kept,
    so a wrong answer can leave a duplicate but cannot drop a distinct task.
    """
    listing = "\n".join(_describe_item(number, item) for number, item in enumerate(items, 1))
    try:
        answer = ask_json(
            client, render_prompt("dedupe_actions.md", items=listing), _Duplicates, config
        )
    except SummaryError as exc:
        logger.warning("Action item deduplication failed, keeping all items: %s", exc)
        return list(items)

    removed: set[int] = set()
    for group in answer.duplikat:
        members = sorted({number - 1 for number in group if 1 <= number <= len(items)} - removed)
        if len(members) < 2:
            continue
        owners = [_owner_tokens(items[index]) for index in members]
        if not all(owners) or not set.intersection(*owners):
            logger.info("Ignored duplicate group %s: owners differ or are unknown", group)
            continue
        keep = max(members, key=lambda index: (_completeness(items[index]), -index))
        removed.update(index for index in members if index != keep)

    if removed:
        logger.info("Merged %d duplicate action item(s)", len(removed))
    return [item for index, item in enumerate(items) if index not in removed]


def _clean_list(values: Sequence[str]) -> tuple[str, ...]:
    """Strip blanks and exact duplicates while keeping order."""
    seen: set[str] = set()
    cleaned: list[str] = []
    for value in values:
        text = value.strip()
        if text and text.lower() not in seen:
            seen.add(text.lower())
            cleaned.append(text)
    return tuple(cleaned)


def _reduce(
    client: LlmClient, parts: Sequence[_SummaryPart], config: SummarizeConfig
) -> _SummaryPart:
    """Merge partial summaries, falling back to concatenation if the LLM output is unusable."""
    partials = json.dumps([part.model_dump() for part in parts], ensure_ascii=False, indent=1)
    try:
        return ask_json(client, render_prompt("reduce.md", partials=partials), _SummaryPart, config)
    except SummaryError as exc:
        logger.warning("Reduce step failed, concatenating partial summaries: %s", exc)
        return _SummaryPart(
            topik=[item for part in parts for item in part.topik],
            keputusan=[item for part in parts for item in part.keputusan],
            poin_penting=[item for part in parts for item in part.poin_penting],
        )


def summarize_transcript(
    transcript: Transcript, client: LlmClient, config: SummarizeConfig
) -> Summary:
    """Produce structured minutes for a transcript of any length."""
    if not transcript.segments:
        return Summary()

    chunks = chunk_segments(transcript.segments, config.chunk_max_chars, config.chunk_overlap_chars)
    logger.info("Summarizing %d chunk(s)", len(chunks))

    parts: list[_SummaryPart] = []
    items: list[ActionItem] = []
    for number, chunk in enumerate(chunks, start=1):
        logger.info("Chunk %d/%d: summary and action items", number, len(chunks))
        text = "\n".join(segment_prompt_line(segment) for segment in chunk)
        parts.append(
            ask_json(client, render_prompt("summary.md", transcript=text), _SummaryPart, config)
        )

        actions = ask_json(
            client, render_prompt("action_items.md", transcript=text), _ActionPart, config
        )
        grounded = ground_action_items(
            actions.tindak_lanjut, QuoteLocator(chunk), config, chunk[0].start
        )
        logger.info(
            "Kept %d of %d action item(s) from chunk %d",
            len(grounded),
            len(actions.tindak_lanjut),
            number,
        )
        items.extend(grounded)

    unique = _dedupe_items(items)
    if len(chunks) == 1:
        merged = parts[0]
    else:
        merged = _reduce(client, parts, config)
        # Overlapping chunks often report one task through different quotes.
        if len(unique) > 1:
            unique = merge_duplicate_items(client, unique, config)

    return Summary(
        topik=_clean_list(merged.topik),
        keputusan=_clean_list(merged.keputusan),
        poin_penting=_clean_list(merged.poin_penting),
        tindak_lanjut=tuple(unique),
    )
