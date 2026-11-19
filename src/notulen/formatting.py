"""Render transcripts and summaries as text."""

from __future__ import annotations

from notulen.schemas import Segment, Summary, Transcript

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


def segment_prompt_line(segment: Segment) -> str:
    """Render one utterance as a compact line for an LLM prompt."""
    return f"[{format_timestamp(segment.start)}] {speaker_name(segment)}: {segment.text}"


def transcript_to_prompt_text(transcript: Transcript) -> str:
    """Render a transcript as compact lines for an LLM prompt."""
    return "\n".join(segment_prompt_line(segment) for segment in transcript.segments)


def _bullets(values: tuple[str, ...], empty: str) -> list[str]:
    return [f"- {value}" for value in values] if values else [f"_{empty}_"]


def _table_cell(value: str | None) -> str:
    """Make a value safe for a Markdown table cell."""
    return (value or "-").replace("|", "\\|").replace("\n", " ")


def render_summary_markdown(source: str, summary: Summary) -> str:
    """Render structured minutes as Markdown."""
    lines = [
        f"# Notulen: {source}",
        "",
        "> Dibuat otomatis oleh LLM. Periksa kembali sebelum dibagikan.",
        "",
        "## Topik",
        "",
        *_bullets(summary.topik, "Tidak ada topik yang terdeteksi."),
        "",
        "## Keputusan",
        "",
        *_bullets(summary.keputusan, "Tidak ada keputusan yang jelas."),
        "",
        "## Poin penting",
        "",
        *_bullets(summary.poin_penting, "Tidak ada poin penting lain."),
        "",
        "## Tindak lanjut",
        "",
    ]
    if summary.tindak_lanjut:
        lines += [
            "| Tugas | Penanggung jawab | Tenggat | Waktu | Kutipan sumber |",
            "| --- | --- | --- | --- | --- |",
        ]
        lines += [
            f"| {_table_cell(item.tugas)} | {_table_cell(item.penanggung_jawab)} "
            f"| {_table_cell(item.tenggat)} | {format_timestamp(item.timestamp)} "
            f'| "{_table_cell(item.kutipan_sumber)}" |'
            for item in summary.tindak_lanjut
        ]
    else:
        lines.append("_Tidak ada tindak lanjut yang jelas._")
    return "\n".join(lines) + "\n"
