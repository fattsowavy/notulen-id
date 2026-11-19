"""Compare Whisper configurations on speed, GPU memory, and accuracy.

Runs each configuration on the same FLEURS items from data/test_set/ and
writes a table to reports/latency.md.

Usage:
    python scripts/benchmark_latency.py --models small medium --limit 30
    python scripts/benchmark_latency.py --models medium --compute-types int8_float16 float16
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from notulen.asr.transcriber import Transcriber
from notulen.audio.preprocess import PreprocessedAudio, preprocess_audio
from notulen.config import PROJECT_ROOT, AppConfig, load_config
from notulen.evaluation.asr_metrics import score_asr
from notulen.evaluation.test_set import discover, reference_text
from notulen.logging_utils import setup_logging

logger = logging.getLogger("benchmark_latency")

SAMPLE_INTERVAL_SECONDS = 0.2


class GpuMemorySampler:
    """Record the peak GPU memory in use, sampled with nvidia-smi."""

    def __init__(self) -> None:
        self.peak_mib: int | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def read_mib() -> int | None:
        try:
            output = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        values = [int(line) for line in output.split() if line.strip().isdigit()]
        return max(values) if values else None

    def _run(self) -> None:
        while not self._stop.is_set():
            value = self.read_mib()
            if value is not None:
                self.peak_mib = max(self.peak_mib or 0, value)
            time.sleep(SAMPLE_INTERVAL_SECONDS)

    def __enter__(self) -> GpuMemorySampler:
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._stop.set()
        self._thread.join()


@dataclass(frozen=True)
class BenchmarkRow:
    """Result of one configuration."""

    model: str
    compute_type: str
    device: str
    load_seconds: float
    real_time_factor: float
    wer: float
    cer: float
    peak_vram_mib: int | None


def run_configuration(
    config: AppConfig,
    model: str,
    compute_type: str,
    audios: list[PreprocessedAudio],
    references: list[str],
) -> BenchmarkRow:
    """Load one model configuration and transcribe every item."""
    asr = config.asr.model_copy(update={"model": model, "compute_type": compute_type})
    baseline = GpuMemorySampler.read_mib()
    with GpuMemorySampler() as sampler, Transcriber(asr, config.vad) as transcriber:
        started = time.perf_counter()
        transcriber.load()
        load_seconds = time.perf_counter() - started

        started = time.perf_counter()
        hypotheses = [transcriber.transcribe(audio).text for audio in audios]
        elapsed = time.perf_counter() - started

    scores = score_asr(references, hypotheses)
    duration = sum(audio.duration for audio in audios)
    peak = None
    if sampler.peak_mib is not None and baseline is not None:
        peak = max(sampler.peak_mib - baseline, 0)
    row = BenchmarkRow(
        model=model,
        compute_type=compute_type,
        device=asr.device,
        load_seconds=load_seconds,
        real_time_factor=elapsed / duration,
        wer=scores.wer,
        cer=scores.cer,
        peak_vram_mib=peak,
    )
    logger.info("%s", row)
    return row


def render(rows: list[BenchmarkRow], items: int, seconds: float) -> str:
    """Render the benchmark as Markdown."""
    lines = [
        "# Perbandingan konfigurasi ASR",
        "",
        f"Dibuat oleh `scripts/benchmark_latency.py` pada {date.today().isoformat()} "
        f"dengan {items} ucapan FLEURS ({seconds / 60:.1f} menit audio).",
        "",
        "RTF adalah waktu transkripsi dibagi durasi audio; di bawah 1 berarti lebih cepat dari "
        "waktu nyata. VRAM adalah tambahan memori GPU di atas kondisi sebelum model dimuat.",
        "",
        "| Model | compute_type | Device | Muat model (dtk) | RTF | WER | CER | VRAM puncak (MB) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        vram = "-" if row.peak_vram_mib is None else str(row.peak_vram_mib)
        lines.append(
            f"| {row.model} | {row.compute_type} | {row.device} | {row.load_seconds:.1f} "
            f"| {row.real_time_factor:.3f} | {row.wer * 100:.1f}% | {row.cer * 100:.1f}% | {vram} |"
        )
    return "\n".join(lines) + "\n"


def parse_args(config: AppConfig) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compare Whisper configurations.")
    parser.add_argument("--models", nargs="+", default=["small", config.asr.model])
    parser.add_argument("--compute-types", nargs="+", default=[config.asr.compute_type])
    parser.add_argument("--limit", type=int, default=30, help="Number of FLEURS items.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "latency.md")
    return parser.parse_args()


def main() -> int:
    """Entry point."""
    config = load_config()
    setup_logging(config.logging.level)
    args = parse_args(config)

    items = [
        item
        for item in discover(config.evaluation.test_set_dir)
        if item.subset == "fleurs" and item.audio and item.transcript
    ][: args.limit]
    if not items:
        logger.error("No FLEURS items found. Run: python scripts/build_test_set.py")
        return 1

    work_dir = config.paths.temp_dir / "benchmark"
    try:
        audios = [
            preprocess_audio(item.audio, config.audio, config.vad, work_dir / item.item_id)
            for item in items
            if item.audio is not None
        ]
        references = [reference_text(item.transcript) for item in items if item.transcript]
        rows = [
            run_configuration(config, model, compute_type, audios, references)
            for model in dict.fromkeys(args.models)
            for compute_type in dict.fromkeys(args.compute_types)
        ]
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render(rows, len(audios), sum(audio.duration for audio in audios)), encoding="utf-8"
    )
    logger.info("Benchmark written to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
