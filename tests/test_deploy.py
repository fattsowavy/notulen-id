"""Tests for configuration layering and the Hugging Face Spaces setup."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest
from pydantic import SecretStr

from notulen.audio.preprocess import probe_duration
from notulen.config import PROJECT_ROOT, Secrets, load_config
from notulen.summarize.llm_client import (
    OpenAICompatibleClient,
    create_llm_client,
    llm_unavailable_reason,
    resolve_api_key,
)

SPACES_CONFIG = PROJECT_ROOT / "configs" / "spaces.yaml"


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HF_TOKEN", "LLM_API_KEY", "OLLAMA_HOST", "NOTULEN_CONFIG"):
        monkeypatch.delenv(name, raising=False)


def load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_spaces_config_overrides_only_what_it_lists(tmp_path: Path) -> None:
    default = load_config(env_path=tmp_path / "missing.env")
    spaces = load_config(SPACES_CONFIG, env_path=tmp_path / "missing.env")

    assert (spaces.asr.device, spaces.asr.compute_type) == ("cuda", "float16")
    assert spaces.asr.cpu_fallback
    assert spaces.diarization.device == "cuda"
    assert spaces.llm.backend == "openai_compatible"
    assert spaces.llm.openai_compatible.use_hf_token
    assert spaces.demo.max_audio_seconds == 120
    assert not default.asr.cpu_fallback
    # Inherited unchanged from default.yaml.
    assert spaces.asr.language == default.asr.language == "id"
    assert spaces.llm.ollama == default.llm.ollama
    assert spaces.paths.output_dir == default.paths.output_dir
    assert default.demo.max_audio_seconds is None


def test_config_path_can_come_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NOTULEN_CONFIG", "configs/spaces.yaml")

    assert load_config(env_path=tmp_path / "missing.env").llm.backend == "openai_compatible"


def test_extends_merges_nested_sections(tmp_path: Path) -> None:
    shutil.copy(PROJECT_ROOT / "configs" / "default.yaml", tmp_path / "base.yaml")
    child = tmp_path / "child.yaml"
    child.write_text("extends: base.yaml\nvad:\n  threshold: 0.3\n", encoding="utf-8")

    config = load_config(child, env_path=tmp_path / "missing.env")

    assert config.vad.threshold == 0.3
    assert config.vad.speech_pad_ms == 200


def test_extends_loop_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="loop"):
        load_config(tmp_path / "a.yaml", env_path=tmp_path / "missing.env")


def test_hf_token_is_used_only_when_allowed(tmp_path: Path) -> None:
    llm = load_config(SPACES_CONFIG, env_path=tmp_path / "missing.env").llm
    no_hf = llm.model_copy(
        update={
            "openai_compatible": llm.openai_compatible.model_copy(update={"use_hf_token": False})
        }
    )
    hf_only = Secrets(hf_token=SecretStr("hf_x"))
    both = Secrets(hf_token=SecretStr("hf_x"), llm_api_key=SecretStr("sk_y"))

    assert resolve_api_key(llm, hf_only) == "hf_x"
    assert resolve_api_key(llm, both) == "sk_y"
    assert resolve_api_key(no_hf, hf_only) is None
    assert isinstance(create_llm_client(llm, hf_only), OpenAICompatibleClient)


def test_missing_credentials_are_reported(tmp_path: Path) -> None:
    spaces = load_config(SPACES_CONFIG, env_path=tmp_path / "missing.env")
    default = load_config(env_path=tmp_path / "missing.env")

    assert "HF_TOKEN" in (llm_unavailable_reason(spaces.llm, Secrets()) or "")
    assert llm_unavailable_reason(spaces.llm, Secrets(hf_token=SecretStr("hf_x"))) is None
    assert llm_unavailable_reason(default.llm, Secrets()) is None


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe not on PATH")
def test_probe_duration(tmp_path: Path) -> None:
    import numpy as np
    import soundfile as sf

    path = tmp_path / "tone.wav"
    sf.write(path, np.zeros(24000, dtype=np.float32), 16000)
    broken = tmp_path / "broken.mp3"
    broken.write_bytes(b"not audio")

    assert probe_duration(path) == pytest.approx(1.5, abs=0.01)
    assert probe_duration(broken) is None


def test_requirements_match_zerogpu_and_skip_dev_tools() -> None:
    lines = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    packages = [line for line in lines if line and not line.startswith(("#", "--"))]

    # ZeroGPU supports PyTorch 2.8 and newer only.
    assert "torch==2.8.0" in packages
    assert "spaces" in packages
    assert not any(name in line for line in packages for name in ("gradio", "pytest", "jiwer"))
    assert (PROJECT_ROOT / "packages.txt").read_text(encoding="utf-8").split() == ["ffmpeg"]
    front_matter = (PROJECT_ROOT / "deploy" / "spaces" / "README.md").read_text(encoding="utf-8")
    assert 'python_version: "3.12"' in front_matter


def test_prepare_space_keeps_git_folder_and_replaces_managed_files(tmp_path: Path) -> None:
    prepare_space = load_script("prepare_space")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref", encoding="utf-8")
    (tmp_path / "src" / "notulen").mkdir(parents=True)
    (tmp_path / "src" / "notulen" / "stale.py").write_text("old", encoding="utf-8")

    files = {path.relative_to(tmp_path).as_posix() for path in prepare_space.prepare(tmp_path)}

    assert {"app.py", "README.md", "requirements.txt", "packages.txt"} <= files
    assert "configs/spaces.yaml" in files
    assert "src/notulen/pipeline.py" in files
    assert "src/notulen/stale.py" not in files
    assert not any("__pycache__" in name for name in files)
    assert (tmp_path / ".git" / "HEAD").is_file()
    assert (tmp_path / "README.md").read_text(encoding="utf-8").startswith("---\ntitle:")
