"""Regression test for the PyTorch and CTranslate2 DLL conflict on Windows."""

from __future__ import annotations

import subprocess
import sys

import pytest

SCRIPT = """
import numpy as np

from notulen.audio.preprocess import detect_speech
from notulen.config import VadConfig

vad = VadConfig(
    enabled=True,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=200,
)
detect_speech(np.zeros(16000, dtype=np.float32), 16000, vad)

import torch

layer = torch.nn.Conv1d(1, 4, 9).cuda()
layer(torch.randn(1, 1, 16000).cuda())
print("ok")
"""


def test_torch_gpu_inference_works_after_vad() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("faster_whisper")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    # A separate process is required: the crash depends on which DLL loads first.
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT], capture_output=True, text=True, timeout=120, check=False
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")
