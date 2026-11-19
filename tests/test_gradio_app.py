"""Tests for the Gradio demo, with the pipeline replaced by a fake."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from notulen.config import PROJECT_ROOT, AppConfig, DemoConfig, PathsConfig, Secrets, load_config
from notulen.pipeline import PipelineResult
from notulen.schemas import ActionItem, Segment, Summary, Transcript

gr = pytest.importorskip("gradio")

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "gradio_app.py"


def load_app() -> ModuleType:
    spec = importlib.util.spec_from_file_location("gradio_app", APP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclasses look their module up in sys.modules while the module is loading.
    sys.modules["gradio_app"] = module
    spec.loader.exec_module(module)
    return module


app = load_app()
# Kept before the autouse fixture replaces it, so it can be tested on its own.
REAL_LIST_OLLAMA_MODELS = app.list_ollama_models

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
        self.llm_calls: list[tuple[Any, str | None]] = []

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

    def summarize(
        self,
        result: PipelineResult,
        on_stage: Any = None,
        llm: Any = None,
        api_key: str | None = None,
    ) -> PipelineResult:
        self.summarized += 1
        self.llm_calls.append((llm, api_key))
        if on_stage:
            on_stage("summary")
        if self.summary_error:
            result.summary_error = self.summary_error
        else:
            result.summary = SUMMARY
            result.files["summary"] = result.output_dir / "summary.md"
            result.files["summary"].write_text("# Notulen", encoding="utf-8")
        return result


@pytest.fixture(autouse=True)
def no_ollama_server(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HF_TOKEN", "LLM_API_KEY", "OLLAMA_HOST", "NOTULEN_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(app, "list_ollama_models", lambda host: ["qwen3:8b", "gemma3:4b"])


def with_paths(config: AppConfig, tmp_path: Path) -> AppConfig:
    return config.model_copy(
        update={"paths": PathsConfig(output_dir=tmp_path / "out", temp_dir=tmp_path / "tmp")}
    )


@pytest.fixture
def config(tmp_path: Path) -> AppConfig:
    return with_paths(load_config(env_path=tmp_path / "missing.env"), tmp_path)


@pytest.fixture
def spaces_config(tmp_path: Path) -> AppConfig:
    spaces = load_config(PROJECT_ROOT / "configs" / "spaces.yaml", env_path=tmp_path / "x.env")
    return with_paths(
        spaces.model_copy(update={"secrets": Secrets(hf_token=SecretStr("hf_server"))}), tmp_path
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


def test_ollama_view_lists_installed_models_and_starts_on_the_configured_one(
    config: AppConfig,
) -> None:
    runner = make_runner(config, FakePipeline())
    view = runner.provider_view("ollama")

    assert runner.default_provider == "ollama"
    assert view.model == config.llm.model == "gemma3:4b"
    assert view.models == ["qwen3:8b", "gemma3:4b"]
    assert not view.key_visible
    assert not view.url_editable


def test_default_run_uses_the_configured_llm_without_a_key(config: AppConfig) -> None:
    pipeline = FakePipeline()

    outputs = make_runner(config, pipeline).run("rapat.mp3", 0, True, lambda *a, **k: None)

    llm, key = pipeline.llm_calls[0]
    assert (llm.backend, llm.model, key) == ("ollama", "gemma3:4b", None)
    assert "gemma3:4b" in outputs[4]


def test_other_provider_needs_the_users_key(config: AppConfig) -> None:
    pipeline = FakePipeline()
    runner = make_runner(config, pipeline)

    outputs = runner.run(
        "rapat.mp3", 0, True, lambda *a, **k: None, provider_id="groq", model="llama-3.3-70b"
    )
    assert pipeline.summarized == 0
    assert "API key untuk Groq" in outputs[4]
    assert "console.groq.com" in outputs[4]

    runner.run(
        "rapat.mp3",
        0,
        True,
        lambda *a, **k: None,
        provider_id="groq",
        model=" llama-3.3-70b ",
        base_url="https://attacker.test/v1",
        api_key=" gsk_user ",
    )
    llm, key = pipeline.llm_calls[0]
    assert llm.model == "llama-3.3-70b"
    # A preset's endpoint cannot be replaced from the page.
    assert llm.openai_compatible.base_url == "https://api.groq.com/openai/v1"
    assert key == "gsk_user"


def test_custom_provider_takes_the_typed_endpoint(config: AppConfig) -> None:
    pipeline = FakePipeline()
    runner = make_runner(config, pipeline)
    assert runner.provider_view("custom").url_editable

    outputs = runner.run(
        "rapat.mp3", 0, True, lambda *a, **k: None, provider_id="custom", model="m", base_url="lm"
    )
    assert "base URL" in outputs[4]

    runner.run(
        "rapat.mp3",
        0,
        True,
        lambda *a, **k: None,
        provider_id="custom",
        model="local-model",
        base_url="http://127.0.0.1:1234/v1",
    )
    llm, key = pipeline.llm_calls[0]
    assert llm.openai_compatible.base_url == "http://127.0.0.1:1234/v1"
    assert key is None


def test_spaces_hides_ollama_and_offers_the_server_key(spaces_config: AppConfig) -> None:
    runner = make_runner(spaces_config, FakePipeline())

    assert runner.default_provider == "hf_router"
    assert "ollama" not in runner.providers
    assert "key milik server" in runner.provider_view("hf_router").key_info
    assert "Wajib" in runner.provider_view("groq").key_info
    assert runner.resolve_llm("hf_router", None, None, None).issue is None
    assert runner.resolve_llm("groq", "llama", None, None).issue is not None


def test_empty_model_is_reported(config: AppConfig) -> None:
    choice = make_runner(config, FakePipeline()).resolve_llm("ollama", "  ", None, None)

    assert choice.config is None
    assert "nama model" in (choice.issue or "")


def test_config_without_providers_offers_the_llm_section(config: AppConfig) -> None:
    bare = config.model_copy(update={"demo": DemoConfig()})
    runner = make_runner(bare, FakePipeline())

    assert list(runner.providers) == [app.CONFIGURED_PROVIDER_ID]
    assert runner.provider_view(app.CONFIGURED_PROVIDER_ID).model == "gemma3:4b"


def test_unknown_default_provider_is_rejected() -> None:
    with pytest.raises(ValidationError, match="default_llm_provider"):
        DemoConfig(default_llm_provider="nope")


def test_list_ollama_models(monkeypatch: pytest.MonkeyPatch) -> None:
    original = REAL_LIST_OLLAMA_MODELS

    def fake_get(url: str, timeout: float) -> httpx.Response:
        assert url == "http://ollama.test:11434/api/tags"
        return httpx.Response(
            200,
            json={"models": [{"name": "b:1"}, {"name": "a:2"}]},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(httpx, "get", fake_get)
    assert original("http://ollama.test:11434/") == ["a:2", "b:1"]

    def broken(url: str, timeout: float) -> httpx.Response:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", broken)
    assert original("http://ollama.test:11434") == []
