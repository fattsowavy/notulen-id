"""Gradio demo: upload a recording and get a transcript, minutes, and action items.

Usage:
    python app/gradio_app.py
    python app/gradio_app.py --port 7861 --share
"""

from __future__ import annotations

import argparse
import logging
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gradio as gr

from notulen.asr.transcriber import TranscriptionError
from notulen.audio.preprocess import AudioPreprocessError
from notulen.config import PROJECT_ROOT, AppConfig, load_config
from notulen.diarization.diarizer import DiarizationError
from notulen.formatting import format_timestamp, render_summary_markdown
from notulen.logging_utils import setup_logging
from notulen.pipeline import NotulenPipeline, PipelineResult
from notulen.schemas import Summary

logger = logging.getLogger("gradio_app")

SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".webm"}
ACTION_COLUMNS = ["Tugas", "Penanggung jawab", "Tenggat", "Waktu", "Kutipan sumber"]

# Progress shown when each stage starts, with an Indonesian label.
STAGE_PROGRESS = {
    "preprocess": (0.05, "Menyiapkan audio"),
    "asr": (0.15, "Mentranskripsi ucapan"),
    "diarization": (0.50, "Mengenali pembicara"),
    "alignment": (0.65, "Menggabungkan kata dan pembicara"),
    "summary": (0.70, "Membuat ringkasan dengan LLM"),
}

KNOWN_ERRORS = (FileNotFoundError, AudioPreprocessError, TranscriptionError, DiarizationError)

ProgressCallback = Callable[..., Any]
Outputs = tuple[str, str, list[list[str]], str | None, str]


def find_samples(samples_dir: Path = SAMPLES_DIR) -> list[str]:
    """List audio files in the samples folder."""
    if not samples_dir.is_dir():
        return []
    return sorted(
        str(path) for path in samples_dir.iterdir() if path.suffix.lower() in AUDIO_EXTENSIONS
    )


def action_rows(summary: Summary | None) -> list[list[str]]:
    """Turn action items into table rows."""
    if summary is None:
        return []
    return [
        [
            item.tugas,
            item.penanggung_jawab or "-",
            item.tenggat or "-",
            format_timestamp(item.timestamp),
            item.kutipan_sumber,
        ]
        for item in summary.tindak_lanjut
    ]


def summary_tab_markdown(result: PipelineResult) -> str:
    """Render the summary tab, or explain why there is no summary."""
    if result.summary_error:
        return f"**Ringkasan gagal dibuat.** {result.summary_error}"
    if result.summary is None:
        return "_Ringkasan tidak dibuat._"
    markdown = render_summary_markdown(result.transcript.source, result.summary)
    # Action items have their own tab.
    return markdown.split("## Tindak lanjut")[0].rstrip() + "\n"


def write_minutes(result: PipelineResult, output_dir: Path) -> Path:
    """Write one Markdown file with the summary followed by the transcript."""
    parts = []
    if "summary" in result.files:
        parts.append(result.files["summary"].read_text(encoding="utf-8"))
    parts.append(result.files["transcript_md"].read_text(encoding="utf-8"))
    path = output_dir / f"notulen_{Path(result.transcript.source).stem}.md"
    path.write_text("\n---\n\n".join(parts), encoding="utf-8")
    return path


def status_markdown(result: PipelineResult) -> str:
    """Summarize timings and any problems."""
    stages = ", ".join(
        f"{name} {seconds:.1f} dtk" for name, seconds in result.timings.items() if name != "total"
    )
    speakers = len({segment.speaker for segment in result.transcript.segments if segment.speaker})
    lines = [
        f"Selesai dalam **{result.timings.get('total', 0.0):.1f} detik** "
        f"untuk audio {format_timestamp(result.transcript.duration)}, "
        f"{speakers} pembicara terdeteksi.",
        "",
        f"Waktu per tahap: {stages}.",
    ]
    if result.summary_error:
        lines += ["", f"⚠️ Transkrip tersimpan, tetapi ringkasan gagal: {result.summary_error}"]
    return "\n".join(lines)


class DemoRunner:
    """Run the pipeline for the web UI, keeping one pipeline for the whole app."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.pipeline = NotulenPipeline(config)

    def run(
        self,
        audio_path: str | None,
        num_speakers: float | None,
        summarize: bool,
        progress: ProgressCallback,
    ) -> Outputs:
        """Process one upload and return values for every output component."""
        if not audio_path:
            raise gr.Error("Unggah atau rekam audio terlebih dahulu.")
        speakers = int(num_speakers) if num_speakers and num_speakers > 0 else None

        def on_stage(stage: str) -> None:
            fraction, label = STAGE_PROGRESS.get(stage, (0.0, stage))
            progress(fraction, desc=label)

        output_dir = self.config.paths.output_dir / "gradio" / uuid.uuid4().hex[:12]
        try:
            result = self.pipeline.run(
                Path(audio_path),
                output_dir=output_dir,
                num_speakers=speakers,
                summarize=summarize,
                on_stage=on_stage,
            )
        except KNOWN_ERRORS as exc:
            logger.exception("Pipeline failed")
            raise gr.Error(str(exc)) from exc
        progress(1.0, desc="Selesai")

        transcript = result.files["transcript_md"].read_text(encoding="utf-8")
        return (
            transcript,
            summary_tab_markdown(result),
            action_rows(result.summary),
            str(write_minutes(result, output_dir)),
            status_markdown(result),
        )


def build_demo(runner: DemoRunner, samples: list[str]) -> gr.Blocks:
    """Build the Gradio interface."""
    with gr.Blocks(title="Notulen Otomatis") as demo:
        gr.Markdown(
            "# Notulen Otomatis\n"
            "Unggah rekaman rapat atau podcast berbahasa Indonesia. Hasilnya berupa transkrip "
            "per pembicara, ringkasan, dan daftar tindak lanjut beserta kutipan sumbernya."
        )
        with gr.Row():
            with gr.Column(scale=1):
                audio = gr.Audio(
                    sources=["upload", "microphone"], type="filepath", label="Audio rapat"
                )
                num_speakers = gr.Number(
                    value=0,
                    precision=0,
                    minimum=0,
                    maximum=20,
                    label="Jumlah pembicara (0 = deteksi otomatis)",
                )
                summarize = gr.Checkbox(value=True, label="Buat ringkasan dan tindak lanjut")
                submit = gr.Button("Proses", variant="primary")
                if samples:
                    gr.Examples(examples=[[path] for path in samples], inputs=[audio])
            with gr.Column(scale=2):
                status = gr.Markdown()
                with gr.Tabs():
                    with gr.Tab("Transkrip"):
                        transcript = gr.Markdown()
                    with gr.Tab("Ringkasan"):
                        summary = gr.Markdown()
                    with gr.Tab("Tindak Lanjut"):
                        actions = gr.Dataframe(headers=ACTION_COLUMNS, interactive=False, wrap=True)
                download = gr.DownloadButton("Unduh notulen (.md)", visible=True)

        def process(
            audio_path: str | None,
            speakers: float | None,
            make_summary: bool,
            progress: gr.Progress = gr.Progress(),  # noqa: B008
        ) -> Outputs:
            return runner.run(audio_path, speakers, make_summary, progress)

        submit.click(
            process,
            inputs=[audio, num_speakers, summarize],
            outputs=[transcript, summary, actions, download, status],
        )
    return demo


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the Notulen Otomatis web demo.")
    parser.add_argument("--host", default="127.0.0.1", help="Address to listen on.")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link.")
    return parser.parse_args()


def main() -> None:
    """Start the web demo."""
    args = parse_args()
    config = load_config()
    setup_logging(config.logging.level)

    demo = build_demo(DemoRunner(config), find_samples())
    # One job at a time: the models share a single GPU.
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=[str(config.paths.output_dir)],
    )


if __name__ == "__main__":
    main()
