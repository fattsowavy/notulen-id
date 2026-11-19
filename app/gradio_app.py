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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr
import httpx

try:
    # Available on Hugging Face Spaces; ZeroGPU only lends a GPU inside @spaces.GPU.
    import spaces
except ImportError:
    spaces = None

from notulen.asr.transcriber import TranscriptionError
from notulen.audio.preprocess import AudioPreprocessError, probe_duration
from notulen.config import PROJECT_ROOT, AppConfig, LlmConfig, LlmProviderConfig, load_config
from notulen.diarization.diarizer import DiarizationError
from notulen.formatting import format_timestamp, render_summary_markdown
from notulen.logging_utils import setup_logging
from notulen.pipeline import NotulenPipeline, PipelineResult
from notulen.schemas import Summary
from notulen.summarize.llm_client import same_endpoint, select_api_key

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

OLLAMA_TAGS_TIMEOUT_SECONDS = 2.0
# Used when the config lists no providers: offer only the LLM from the llm: section.
CONFIGURED_PROVIDER_ID = "config"
KEY_PRIVACY_NOTE = (
    "API key hanya dipakai untuk permintaan ini: tidak disimpan dan tidak ditulis ke log. "
    "Key milik server hanya dikirim ke penyedia yang diatur di file config."
)

ProgressCallback = Callable[..., Any]
Outputs = tuple[str, str, list[list[str]], str | None, str]


@dataclass(frozen=True)
class ProviderView:
    """What the LLM settings panel shows for one provider."""

    models: list[str]
    model: str
    base_url: str
    url_editable: bool
    key_visible: bool
    key_info: str


@dataclass(frozen=True)
class LlmChoice:
    """The LLM chosen for one request, or why it cannot be used."""

    config: LlmConfig | None
    label: str
    api_key: str | None = None
    issue: str | None = None


def list_ollama_models(host: str, timeout: float = OLLAMA_TAGS_TIMEOUT_SECONDS) -> list[str]:
    """Names of the models installed in Ollama, or an empty list if it cannot be reached."""
    try:
        response = httpx.get(f"{host.rstrip('/')}/api/tags", timeout=timeout)
        response.raise_for_status()
        return sorted(str(model["name"]) for model in response.json().get("models", []))
    except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError):
        return []


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
        self.providers = self._enabled_providers(config)
        self.default_provider = config.demo.default_llm_provider or next(iter(self.providers))
        uses_ollama = any(provider.backend == "ollama" for provider in self.providers.values())
        self._ollama_models = list_ollama_models(config.llm.ollama.host) if uses_ollama else []
        _ACTIVE_RUNNER = self

    @staticmethod
    def _enabled_providers(config: AppConfig) -> dict[str, LlmProviderConfig]:
        providers = {
            name: provider
            for name, provider in config.demo.llm_providers.items()
            if provider.enabled
        }
        if providers:
            return providers
        llm = config.llm
        return {
            CONFIGURED_PROVIDER_ID: LlmProviderConfig(
                label=f"{llm.backend} (config)",
                backend=llm.backend,
                base_url=llm.openai_compatible.base_url if llm.backend != "ollama" else "",
                models=(llm.model,),
            )
        }

    def _llm_config(self, provider: LlmProviderConfig, model: str, base_url: str) -> LlmConfig:
        """The configured LLM settings with the provider, model, and endpoint replaced."""
        if provider.backend == "ollama":
            return self.config.llm.model_copy(update={"backend": "ollama", "model": model})
        endpoint = self.config.llm.openai_compatible.model_copy(update={"base_url": base_url})
        return self.config.llm.model_copy(
            update={"backend": "openai_compatible", "model": model, "openai_compatible": endpoint}
        )

    def _is_configured(self, provider: LlmProviderConfig) -> bool:
        """Whether the provider is the LLM from the config file's llm: section."""
        if provider.backend == "ollama":
            return self.config.llm.backend == "ollama"
        candidate = self._llm_config(provider, self.config.llm.model, provider.base_url)
        return same_endpoint(candidate, self.config.llm)

    def provider_view(self, provider_id: str) -> ProviderView:
        """Describe how the settings panel should look for a provider."""
        provider = self.providers[provider_id]
        if provider.backend == "ollama":
            models = self._ollama_models or list(provider.models)
            base_url, editable = self.config.llm.ollama.host, False
        else:
            models, base_url, editable = (
                list(provider.models),
                provider.base_url,
                not provider.base_url,
            )

        configured = self._is_configured(provider)
        model = self.config.llm.model if configured else (models[0] if models else "")
        if model and model not in models:
            models = [model, *models]

        server_key = configured and (
            select_api_key(self.config.llm, self.config.llm, self.config.secrets) is not None
        )
        if provider.backend == "ollama":
            key_info = ""
        elif server_key:
            key_info = "Opsional. Jika kosong, dipakai key milik server."
        elif provider.requires_key:
            key_info = f"Wajib untuk {provider.label}." + (
                f" Buat di {provider.key_url}" if provider.key_url else ""
            )
        else:
            key_info = "Opsional, isi jika server LLM memintanya."
        return ProviderView(
            models=models,
            model=model,
            base_url=base_url,
            url_editable=editable,
            key_visible=provider.backend != "ollama",
            key_info=key_info,
        )

    def resolve_llm(
        self,
        provider_id: str | None,
        model: str | None,
        base_url: str | None,
        api_key: str | None,
    ) -> LlmChoice:
        """Turn the settings panel values into an LLM config, or explain what is missing."""
        provider_id = provider_id or self.default_provider
        provider = self.providers.get(provider_id)
        if provider is None:
            return LlmChoice(
                None, provider_id, issue=f"Penyedia LLM {provider_id!r} tidak dikenal."
            )
        if model is None:
            model = self.provider_view(provider_id).model
        model = model.strip()
        if not model:
            return LlmChoice(None, provider.label, issue="Ringkasan dilewati: isi nama model LLM.")

        # Preset endpoints are fixed; only an empty preset takes the URL typed by the user.
        url = provider.base_url or (base_url or "").strip()
        if provider.backend != "ollama" and not url.startswith(("http://", "https://")):
            return LlmChoice(
                None,
                provider.label,
                issue="Ringkasan dilewati: isi base URL yang diawali http:// atau https://.",
            )

        config = self._llm_config(provider, model, url)
        user_key = api_key.strip() if api_key and api_key.strip() else None
        if provider.backend != "ollama" and provider.requires_key:
            if select_api_key(config, self.config.llm, self.config.secrets, user_key) is None:
                hint = f" Buat di {provider.key_url}" if provider.key_url else ""
                return LlmChoice(
                    None,
                    provider.label,
                    issue=f"Ringkasan dilewati: isi API key untuk {provider.label}.{hint}",
                )
        return LlmChoice(config, provider.label, api_key=user_key)

    def run(
        self,
        audio_path: str | None,
        num_speakers: float | None,
        summarize: bool,
        progress: ProgressCallback,
        provider_id: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> Outputs:
        """Process one upload and return values for every output component.

        The LLM settings default to the provider selected when the page opens.
        """
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
        choice = self.resolve_llm(provider_id, model, base_url, api_key) if summarize else None
        if choice is not None and choice.issue:
            summarize = False
            notes.append(f"⚠️ {choice.issue}")
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
        if summarize and choice is not None and choice.config is not None:
            result = self.pipeline.summarize(
                result, on_stage, llm=choice.config, api_key=choice.api_key
            )
            if result.summary is not None:
                notes.append(f"Ringkasan dibuat dengan `{choice.config.model}` ({choice.label}).")
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
                summarize = gr.Checkbox(value=True, label="Buat ringkasan dan tindak lanjut")
                view = runner.provider_view(runner.default_provider)
                with gr.Accordion("Pengaturan LLM", open=False):
                    provider = gr.Dropdown(
                        choices=[(item.label, name) for name, item in runner.providers.items()],
                        value=runner.default_provider,
                        label="Penyedia",
                    )
                    model = gr.Dropdown(
                        choices=view.models,
                        value=view.model,
                        allow_custom_value=True,
                        label="Model",
                        info="Pilih dari daftar atau ketik nama model lain.",
                    )
                    base_url = gr.Textbox(
                        value=view.base_url,
                        interactive=view.url_editable,
                        label="Base URL",
                        placeholder="http://127.0.0.1:1234/v1",
                    )
                    api_key = gr.Textbox(
                        type="password",
                        label="API key",
                        info=view.key_info,
                        visible=view.key_visible,
                    )
                    gr.Markdown(KEY_PRIVACY_NOTE)
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

        def on_provider_change(provider_id: str) -> tuple[Any, Any, Any]:
            selected = runner.provider_view(provider_id)
            return (
                gr.update(choices=selected.models, value=selected.model),
                gr.update(value=selected.base_url, interactive=selected.url_editable),
                # Clear the key so it is not sent to a provider it was not meant for.
                gr.update(value="", visible=selected.key_visible, info=selected.key_info),
            )

        provider.change(on_provider_change, inputs=provider, outputs=[model, base_url, api_key])

        def process(
            audio_path: str | None,
            speakers: float | None,
            make_summary: bool,
            provider_id: str,
            model_name: str | None,
            endpoint: str | None,
            key: str | None,
            progress: gr.Progress = gr.Progress(),  # noqa: B008
        ) -> Outputs:
            return runner.run(
                audio_path,
                speakers,
                make_summary,
                progress,
                provider_id=provider_id,
                model=model_name,
                base_url=endpoint,
                api_key=key,
            )

        submit.click(
            process,
            inputs=[audio, num_speakers, summarize, provider, model, base_url, api_key],
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
