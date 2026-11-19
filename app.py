"""Hugging Face Spaces entry point.

Spaces runs this file with `python app.py`. It uses configs/spaces.yaml
(ZeroGPU for Whisper and pyannote, LLM through the Hugging Face router) unless
NOTULEN_CONFIG points elsewhere. For local use, run app/gradio_app.py instead.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    # ZeroGPU must be imported before anything initializes CUDA.
    import spaces  # noqa: F401
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent
# The package is not pip-installed on Spaces, so import it from the source tree.
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "app")]
os.environ.setdefault("NOTULEN_CONFIG", str(ROOT / "configs" / "spaces.yaml"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from gradio_app import main  # noqa: E402

if __name__ == "__main__":
    main([])
