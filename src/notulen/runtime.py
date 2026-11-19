"""Process-wide runtime safeguards."""

from __future__ import annotations

import importlib
import sys


def load_torch_first() -> None:
    """Import PyTorch before anything that imports CTranslate2.

    On Windows, CTranslate2 bundles its own cudnn64_9.dll. A process can hold only
    one copy, and PyTorch crashes natively if that copy is loaded first. CTranslate2
    works correctly with the copy shipped by PyTorch.
    """
    if sys.platform == "win32":
        importlib.import_module("torch")
