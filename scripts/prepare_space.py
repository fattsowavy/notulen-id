"""Assemble the files for a Hugging Face Space into one folder.

The Space needs its own README with YAML front matter, so the files are
copied into dist/space/ instead of pushing the whole repository.

Usage:
    python scripts/prepare_space.py
    python scripts/prepare_space.py --output C:/path/to/space-repo
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from notulen.config import PROJECT_ROOT
from notulen.logging_utils import setup_logging

logger = logging.getLogger("prepare_space")

FILES = {
    "app.py": "app.py",
    "requirements.txt": "requirements.txt",
    "packages.txt": "packages.txt",
    "deploy/spaces/README.md": "README.md",
    "app/gradio_app.py": "app/gradio_app.py",
    "configs/default.yaml": "configs/default.yaml",
    "configs/spaces.yaml": "configs/spaces.yaml",
    "LICENSE": "LICENSE",
}
FOLDERS = {
    "src/notulen": "src/notulen",
    "data/samples": "data/samples",
}
IGNORED = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info", "README.md")


def prepare(output: Path) -> list[Path]:
    """Copy the Space files into output.

    Only the paths this script manages are replaced, so a cloned Space
    repository keeps its .git folder and any other files.
    """
    output.mkdir(parents=True, exist_ok=True)
    for target in [*FILES.values(), *FOLDERS.values()]:
        existing = output / target
        if existing.is_dir():
            shutil.rmtree(existing)
        elif existing.exists():
            existing.unlink()

    for source, target in FILES.items():
        path = PROJECT_ROOT / source
        if not path.is_file():
            logger.warning("Skipping missing file %s", source)
            continue
        (output / target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, output / target)

    for source, target in FOLDERS.items():
        path = PROJECT_ROOT / source
        if path.is_dir():
            shutil.copytree(path, output / target, ignore=IGNORED)

    return sorted(path for path in output.rglob("*") if path.is_file() and ".git" not in path.parts)


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Assemble the Hugging Face Space files.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "dist" / "space")
    args = parser.parse_args()
    setup_logging()

    if args.output.resolve() == PROJECT_ROOT.resolve():
        parser.error("--output must not be the project folder")
    files = prepare(args.output)
    size = sum(path.stat().st_size for path in files)
    logger.info("Wrote %d files (%.1f MB) to %s", len(files), size / 2**20, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
