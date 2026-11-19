"""Summarize transcripts with an LLM."""

from __future__ import annotations

import logging
from pathlib import Path

from notulen.config import SummarizeConfig
from notulen.formatting import transcript_to_prompt_text
from notulen.schemas import Transcript
from notulen.summarize.llm_client import LlmClient

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"
TRANSCRIPT_PLACEHOLDER = "{transcript}"


def load_prompt(name: str) -> str:
    """Read a prompt template from the prompts folder."""
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def summarize_simple(transcript: Transcript, client: LlmClient, config: SummarizeConfig) -> str:
    """Summarize a transcript with one prompt.

    Transcripts longer than ``chunk_max_chars`` are truncated. Map-reduce
    summarization of long recordings replaces this in the next iteration.
    """
    text = transcript_to_prompt_text(transcript)
    if len(text) > config.chunk_max_chars:
        logger.warning(
            "Transcript has %d characters; summarizing only the first %d",
            len(text),
            config.chunk_max_chars,
        )
        text = text[: config.chunk_max_chars]

    prompt = load_prompt("summary.md").replace(TRANSCRIPT_PLACEHOLDER, text)
    return client.complete(prompt).strip()
