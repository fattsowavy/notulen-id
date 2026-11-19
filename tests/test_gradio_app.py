"""Tests for the Gradio demo, with the pipeline replaced by a fake."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from notulen.config import AppConfig, PathsConfig, load_config
from notulen.pipeline import PipelineResult
from notulen.schemas import ActionItem, Segment, Summary, Transcript

gr = pytest.importorskip("gradio")

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "gradio_app.py"


def load_app() -> ModuleType:
    spec = importlib.util.spec_from_file_location("gradio_app", APP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


app = load_app()

SUMMARY = Summary(
    topik=("Anggaran",),
    tindak_lanjut=(
        ActionItem(
            tugas="Kirim laporan",
            penanggung_jawab="Rina",
            kutipan_sumber="tolong kirim laporan",
            timestamp=65.0,
        ),
    ),
)


class FakePipeline:
    """Write minimal output files and report each stage."""

    def __init__(self, summary_error: str | None = None) -> None:
        self.summary_error = summary_error
        self.calls: list[dict[str, Any]] = []
        self.summarized = 0

    def run(self, audio_path: Path, **kwargs: Any) -> PipelineResult:
        self.calls.append(kwargs)
        if kwargs.get("on_stage"):
            for stage in ("preprocess", "asr", "diarization", "alignment"):
                kwargs["on_stage"](stage)

        output_dir: Path = kwargs["output_dir"]
        output_dir.mkdir(parents=True)
        transcript_md = output_dir / "transcript.md"
        transcript_md.write_text("**Pembicara 1** [00:00:00]: halo", encoding="utf-8")
        return PipelineResult(
            transcript=Transcript(
                source=audio_path.name,
                language="id",
                model="medium",
                duration=90.0,
                segments=(Segment(start=0.0, end=1.0, text="halo", speaker="Pembicara 1"),),
            ),
            turns=(),
            output_dir=output_dir,
            timings={"asr": 2.0, "total": 3.5},
            files={"transcript_md": transcript_md},
        )

    def summarize(self, result: PipelineResult, on_stage: Any = None) -> PipelineResult:
        self.summarized += 1
        if on_stage:
            on_stage("summary")
        if self.summary_error:
            result.summary_error = self.summary_error
        else:
            result.summary = SUMMARY
            result.files["summary"] = result.output_dir / "summary.md"
            result.files["summary"].write_text("# Notulen", encoding="utf-8")
        return result


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppConfig:
    for name in ("HF_TOKEN", "LLM_API_KEY", "OLLAMA_HOST"):
        monkeypatch.delenv(name, raising=False)
    base = load_config(env_path=tmp_path / "missing.env")
    return base.model_copy(
        update={"paths": PathsConfig(output_dir=tmp_path / "out", temp_dir=tmp_path / "tmp")}
    )


def make_runner(config: AppConfig, pipeline: FakePipeline) -> Any:
    runner = app.DemoRunner(config)
    runner.pipeline = pipeline
    return runner


def test_find_samples_lists_audio_only(tmp_path: Path) -> None:
    for name in ("b.mp3", "a.WAV", "notes.txt"):
        (tmp_path / name).write_bytes(b"")

    assert [Path(path).name for path in app.find_samples(tmp_path)] == ["a.WAV", "b.mp3"]
    assert app.find_samples(tmp_path / "missing") == []


def test_action_rows() -> None:
    assert app.action_rows(SUMMARY) == [
        ["Kirim laporan", "Rina", "-", "00:01:05", "tolong kirim laporan"]
    ]
    assert app.action_rows(None) == []


def test_run_returns_every_output(config: AppConfig) -> None:
    pipeline = FakePipeline()
    progress: list[tuple[float, str]] = []

    transcript, summary, rows, download, status = make_runner(config, pipeline).run(
        "C:/rekaman/rapat.mp3", 3.0, True, lambda value, desc: progress.append((value, desc))
    )

    assert "Pembicara 1" in transcript
    assert "## Topik" in summary
    assert "Tindak lanjut" not in summary
    assert rows[0][0] == "Kirim laporan"
    minutes = Path(download)
    assert minutes.name == "notulen_rapat.md"
    assert (
        minutes.read_text(encoding="utf-8") == "# Notulen\n---\n\n**Pembicara 1** [00:00:00]: halo"
    )
    assert "3.5 detik" in status
    assert pipeline.calls[0]["num_speakers"] == 3
    assert [desc for _, desc in progress][-1] == "Selesai"
    assert [value for value, _ in progress] == sorted(value for value, _ in progress)


def test_zero_speakers_means_automatic(config: AppConfig) -> None:
    pipeline = FakePipeline()

    make_runner(config, pipeline).run("rapat.mp3", 0, False, lambda *args, **kwargs: None)

    assert pipeline.calls[0]["num_speakers"] is None
    assert pipeline.summarized == 0


def test_audio_stages_never_include_the_summary(config: AppConfig) -> None:
    pipeline = FakePipeline()

    make_runner(config, pipeline).run("rapat.mp3", 0, True, lambda *args, **kwargs: None)

    assert pipeline.calls[0]["summarize"] is False
    assert pipeline.summarized == 1


def test_zerogpu_path_runs_audio_stages_in_the_gpu_function(
    config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = FakePipeline()
    runner = make_runner(config, pipeline)
    monkeypatch.setattr(app, "ON_ZEROGPU", True)
    progress: list[str] = []

    outputs = runner.run("rapat.mp3", 2, True, lambda value, desc: progress.append(desc))

    # The real function, as on Spaces without a GPU decorator, reaches the shared runner.
    assert pipeline.calls[0]["num_speakers"] == 2
    assert "on_stage" not in pipeline.calls[0]
    assert progress[0] == "Memproses audio di GPU"
    assert "Membuat ringkasan dengan LLM" in progress
    assert "Pembicara 1" in outputs[0]


@pytest.mark.parametrize(
    ("audio_seconds", "expected"), [(None, 150), (0.0, 40), (60.0, 70), (120.0, 100), (900.0, 150)]
)
def test_gpu_time_estimate(audio_seconds: float | None, expected: int) -> None:
    assert app.estimate_gpu_seconds(audio_seconds) == expected


def test_gpu_decorator_is_a_no_op_without_spaces(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "spaces", None)

    def work() -> str:
        return "ok"

    assert app.gpu(duration=lambda: 1)(work) is work


def test_failed_summary_is_reported(config: AppConfig) -> None:
    outputs = make_runner(config, FakePipeline("Cannot reach Ollama")).run(
        "rapat.mp3", None, True, lambda *args, **kwargs: None
    )

    assert "Ringkasan gagal" in outputs[1]
    assert outputs[2] == []
    assert "Cannot reach Ollama" in outputs[4]
    assert Path(outputs[3]).read_text(encoding="utf-8").startswith("**Pembicara 1**")


def test_missing_audio_is_rejected(config: AppConfig) -> None:
    with pytest.raises(gr.Error):
        make_runner(config, FakePipeline()).run(None, 0, True, lambda *args, **kwargs: None)


def test_pipeline_errors_are_shown_to_the_user(config: AppConfig) -> None:
    pipeline = FakePipeline()

    def fail(*args: Any, **kwargs: Any) -> None:
        raise FileNotFoundError("Audio file not found: rapat.mp3")

    pipeline.run = fail  # type: ignore[method-assign]

    with pytest.raises(gr.Error, match="not found"):
        make_runner(config, pipeline).run("rapat.mp3", 0, True, lambda *args, **kwargs: None)


def test_build_demo_with_and_without_samples(config: AppConfig, tmp_path: Path) -> None:
    sample = tmp_path / "contoh.wav"
    sample.write_bytes(b"")
    runner = make_runner(config, FakePipeline())

    assert isinstance(app.build_demo(runner, []), gr.Blocks)
    assert isinstance(app.build_demo(runner, [str(sample)]), gr.Blocks)
