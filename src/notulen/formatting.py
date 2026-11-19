"""Render transcripts and summaries as text."""

from __future__ import annotations

from notulen.schemas import Segment, Transcript

UNKNOWN_SPEAKER = "Tidak diketahui"


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    total = max(int(seconds), 0)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def speaker_name(segment: Segment) -> str:
    """Return the segment's speaker label, or a placeholder when it is unknown."""
    return segment.speaker or UNKNOWN_SPEAKER


def render_transcript_markdown(transcript: Transcript) -> str:
    """Render a speaker-labelled transcript as Markdown."""
    lines = [
        f"# Transkrip: {transcript.source}",
        "",
        f"- Durasi: {format_timestamp(transcript.duration)}",
        f"- Model ASR: {transcript.model}",
        f"- Bahasa: {transcript.language}",
        "",
    ]
    for segment in transcript.segments:
        lines.append(
            f"**{speaker_name(segment)}** [{format_timestamp(segment.start)}]: {segment.text}"
        )
        lines.append("")
    return "\n".join(lines)


def transcript_to_prompt_text(transcript: Transcript) -> str:
    """Render a transcript as compact lines for an LLM prompt."""
    return "\n".join(
        f"[{format_timestamp(segment.start)}] {speaker_name(segment)}: {segment.text}"
        for segment in transcript.segments
    )


def render_summary_markdown(source: str, summary: str) -> str:
    """Wrap an LLM summary in a Markdown document."""
    return (
        f"# Ringkasan: {source}\n\n"
        "> Dibuat otomatis oleh LLM. Periksa kembali sebelum dibagikan.\n\n"
        f"{summary.strip()}\n"
    )
