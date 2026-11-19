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

try:
    # Available on Hugging Face Spaces; ZeroGPU only lends a GPU inside @spaces.GPU.
    import spaces
except ImportError:
    spaces = None

from notulen.asr.transcriber import TranscriptionError
from notulen.audio.preprocess import AudioPreprocessError, probe_duration
from notulen.config import PROJECT_ROOT, AppConfig, load_config
from notulen.diarization.diarizer import DiarizationError
from notulen.formatting import format_timestamp, render_summary_markdown
from notulen.logging_utils import setup_logging
from notulen.pipeline import NotulenPipeline, PipelineResult
from notulen.schemas import Summary
from notulen.summarize.llm_client import llm_unavailable_reason

logger = logging.getLogger("gradio_app")

# spaces.GPU has no effect on other hardware, so the GPU path is safe wherever it is installed.
ON_ZEROGPU = spaces is not None

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

# GPU time requested per call on ZeroGPU, in seconds.
GPU_BASE_SECONDS = 40
GPU_SECONDS_PER_AUDIO_SECOND = 0.5
GPU_MAX_SECONDS = 150

ProgressCallback = Callable[..., Any]
Outputs = tuple[str, str, list[list[str]], str | None, str]

# The runner used by run_audio_stages, which must not take it as an argument.
_ACTIVE_RUNNER: DemoRunner | None = None


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


def estimate_gpu_seconds(audio_seconds: float | None) -> int:
    """GPU time to request on ZeroGPU: model loading plus about half the audio length.

    Shorter requests get better queue priority and use less of the visitor's quota.
    """
    if audio_seconds is None:
        return GPU_MAX_SECONDS
    estimate = GPU_BASE_SECONDS + GPU_SECONDS_PER_AUDIO_SECOND * audio_seconds
    return int(min(max(estimate, GPU_BASE_SECONDS), GPU_MAX_SECONDS))


def _gpu_duration(audio_path: str, output_dir: str, num_speakers: int | None) -> int:
    return estimate_gpu_seconds(probe_duration(Path(audio_path)))


def gpu(duration: Callable[..., int]) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Wrap a function with spaces.GPU on ZeroGPU, and leave it unchanged elsewhere."""
    if spaces is None:
        return lambda function: function
    decorator: Callable[[Callable[..., Any]], Callable[..., Any]] = spaces.GPU(duration=duration)
    return decorator


@gpu(duration=_gpu_duration)
def run_audio_stages(audio_path: str, output_dir: str, num_speakers: int | None) -> PipelineResult:
    """Run every GPU stage (ASR and diarization) and write the transcript.

    Defined at module level, with plain arguments, so ZeroGPU can run it in its
    GPU worker process.
    """
    if _ACTIVE_RUNNER is None:
        raise RuntimeError("The demo runner has not been created")
    return _ACTIVE_RUNNER.pipeline.run(
        Path(audio_path),
        output_dir=Path(output_dir),
        num_speakers=num_speakers,
        summarize=False,
    )


class DemoRunner:
    """Run the pipeline for the web UI, keeping one pipeline for the whole app."""

    def __init__(self, config: AppConfig) -> None:
        global _ACTIVE_RUNNER
        self.config = config
        self.pipeline = NotulenPipeline(config)
        self.llm_issue = llm_unavailable_reason(config.llm, config.secrets)
        _ACTIVE_RUNNER = self

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
        limit = self.config.demo.max_audio_seconds
        if limit is not None:
            duration = probe_duration(Path(audio_path))
            if duration is not None and duration > limit:
                raise gr.Error(
                    f"Audio berdurasi {format_timestamp(duration)}, melebihi batas "
                    f"{format_timestamp(limit)} di demo ini. Potong audionya atau jalankan lokal."
                )
        notes: list[str] = []
        if summarize and self.llm_issue:
            summarize = False
            notes.append(self.llm_issue)
        speakers = int(num_speakers) if num_speakers and num_speakers > 0 else None

        def on_stage(stage: str) -> None:
            fraction, label = STAGE_PROGRESS.get(stage, (0.0, stage))
            progress(fraction, desc=label)

        output_dir = self.config.paths.output_dir / "gradio" / uuid.uuid4().hex[:12]
        try:
            if ON_ZEROGPU:
                # Stage callbacks cannot cross into the GPU worker process.
                progress(0.1, desc="Memproses audio di GPU")
                result = run_audio_stages(str(audio_path), str(output_dir), speakers)
            else:
                result = self.pipeline.run(
                    Path(audio_path),
                    output_dir=output_dir,
                    num_speakers=speakers,
                    summarize=False,
                    on_stage=on_stage,
                )
        except KNOWN_ERRORS as exc:
            logger.exception("Pipeline failed")
            raise gr.Error(str(exc)) from exc

        # The LLM runs outside the GPU function so it does not spend GPU quota.
        if summarize:
            result = self.pipeline.summarize(result, on_stage)
        progress(1.0, desc="Selesai")

        transcript = result.files["transcript_md"].read_text(encoding="utf-8")
        return (
            transcript,
            summary_tab_markdown(result),
            action_rows(result.summary),
            str(write_minutes(result, output_dir)),
            "\n\n".join([status_markdown(result), *notes]),
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
                limit = runner.config.demo.max_audio_seconds
                audio = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Audio rapat"
                    + (f" (maksimal {format_timestamp(limit)})" if limit else ""),
                    max_length=int(limit) if limit else None,
                )
                num_speakers = gr.Number(
                    value=0,
                    precision=0,
                    minimum=0,
                    maximum=20,
                    label="Jumlah pembicara (0 = deteksi otomatis)",
                )
                summarize = gr.Checkbox(
                    value=runner.llm_issue is None,
                    interactive=runner.llm_issue is None,
                    label="Buat ringkasan dan tindak lanjut",
                    info=runner.llm_issue,
                )
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Host and port default to Gradio's own defaults, which honour GRADIO_SERVER_NAME
    and GRADIO_SERVER_PORT (set automatically on Hugging Face Spaces).
    """
    parser = argparse.ArgumentParser(description="Run the Notulen Otomatis web demo.")
    parser.add_argument("--host", help="Address to listen on (default 127.0.0.1).")
    parser.add_argument("--port", type=int, help="Port to listen on (default 7860).")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Start the web demo."""
    args = parse_args(argv)
    config = load_config()
    setup_logging(config.logging.level)

    demo = build_demo(DemoRunner(config), find_samples())
    # One job at a time: the models share one GPU, or two CPU cores on Spaces.
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=[str(config.paths.output_dir)],
    )


if __name__ == "__main__":
    main()
