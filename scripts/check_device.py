"""Detect local hardware and tools, then recommend pipeline defaults.

Uses only the standard library so it can run before the conda env exists.
PyTorch is inspected when it happens to be installed.

Usage:
    python scripts/check_device.py
    python scripts/check_device.py --json outputs/device.json
"""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("check_device")

# Cards report slightly less than their marketed size (an "8 GB" card is often
# ~7900-8188 MiB), so the tiers use tolerant thresholds.
VRAM_HIGH_MIB = 7600
VRAM_MID_MIB = 3800
COMMAND_TIMEOUT_S = 20
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"


@dataclass
class GpuInfo:
    """A single NVIDIA GPU as reported by nvidia-smi."""

    name: str
    vram_mib: int
    driver: str


@dataclass
class TorchInfo:
    """PyTorch status in the current interpreter."""

    installed: bool
    version: str | None = None
    cuda_available: bool = False
    cuda_version: str | None = None
    vram_mib: int | None = None


@dataclass
class OllamaInfo:
    """Ollama availability and installed models."""

    binary: str | None = None
    server_running: bool = False
    models: list[str] = field(default_factory=list)


@dataclass
class Recommendation:
    """Pipeline defaults derived from the detected hardware."""

    whisper_model: str
    compute_type: str
    device: str
    llm_size_class: str


@dataclass
class DeviceReport:
    """Everything detected on this machine."""

    os: str
    python: str
    cpu_physical_cores: int | None
    cpu_logical_cores: int | None
    ram_total_gb: float | None
    gpus: list[GpuInfo]
    torch: TorchInfo
    ffmpeg: str | None
    ollama: OllamaInfo
    recommendation: Recommendation


def run_command(args: list[str]) -> str | None:
    """Run a command and return stdout, or None if it is missing or fails."""
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=COMMAND_TIMEOUT_S,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("Command %s failed: %s", args, exc)
        return None
    return result.stdout.strip()


def detect_gpus() -> list[GpuInfo]:
    """Query NVIDIA GPUs through nvidia-smi."""
    output = run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return []

    gpus: list[GpuInfo] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            gpus.append(GpuInfo(name=parts[0], vram_mib=int(float(parts[1])), driver=parts[2]))
        except ValueError:
            logger.debug("Could not parse nvidia-smi line: %s", line)
    return gpus


def detect_torch() -> TorchInfo:
    """Inspect PyTorch and CUDA support if PyTorch is installed."""
    try:
        import torch
    except ImportError:
        return TorchInfo(installed=False)

    info = TorchInfo(
        installed=True,
        version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        cuda_version=torch.version.cuda,
    )
    if info.cuda_available:
        info.vram_mib = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
    return info


def detect_ram_gb() -> float | None:
    """Return total physical RAM in GiB."""
    if sys.platform == "win32":

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return round(status.ullTotalPhys / 1024**3, 1)
        return None

    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1)
    except (AttributeError, OSError, ValueError):
        return None


def detect_physical_cores() -> int | None:
    """Return the physical CPU core count, or None if it cannot be determined."""
    if sys.platform == "win32":
        output = run_command(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor | Measure-Object NumberOfCores -Sum).Sum",
            ]
        )
        return int(output) if output and output.isdigit() else None

    output = run_command(["lscpu", "-p=CORE,SOCKET"])
    if not output:
        return None
    return len({line for line in output.splitlines() if not line.startswith("#")}) or None


def find_ollama_binary() -> str | None:
    """Locate the Ollama binary on PATH or in the default Windows install folders."""
    found = shutil.which("ollama")
    if found:
        return found

    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Ollama" / "ollama.exe",
    ]
    return next((str(path) for path in candidates if path.is_file()), None)


def fetch_ollama_tags() -> list[str] | None:
    """Return model names from the Ollama HTTP API, or None if the server is down."""
    host = os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST
    if "://" not in host:
        host = f"http://{host}"
    try:
        with urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=3) as response:
            payload: dict[str, Any] = json.load(response)
    except (OSError, urllib.error.URLError, ValueError) as exc:
        logger.debug("Ollama API unreachable at %s: %s", host, exc)
        return None
    return [model["name"] for model in payload.get("models", []) if "name" in model]


def detect_ollama() -> OllamaInfo:
    """Detect the Ollama binary, server state, and installed models."""
    info = OllamaInfo(binary=find_ollama_binary())
    tags = fetch_ollama_tags()
    if tags is not None:
        info.server_running = True
        info.models = tags
    return info


def recommend(gpus: list[GpuInfo], ram_gb: float | None) -> Recommendation:
    """Map detected hardware to Whisper and LLM defaults."""
    vram_mib = max((gpu.vram_mib for gpu in gpus), default=0)

    if vram_mib >= VRAM_HIGH_MIB:
        return Recommendation("large-v3-turbo", "float16", "cuda", "7-8B")
    if vram_mib >= VRAM_MID_MIB:
        return Recommendation("medium", "int8_float16", "cuda", "3-4B")

    llm_size = "7-8B" if ram_gb is not None and ram_gb >= 32 else "3-4B"
    return Recommendation("small", "int8", "cpu", llm_size)


def collect_report() -> DeviceReport:
    """Run every detection step and build the report."""
    gpus = detect_gpus()
    ram_gb = detect_ram_gb()
    return DeviceReport(
        os=f"{platform.system()} {platform.release()} ({platform.machine()})",
        python=platform.python_version(),
        cpu_physical_cores=detect_physical_cores(),
        cpu_logical_cores=os.cpu_count(),
        ram_total_gb=ram_gb,
        gpus=gpus,
        torch=detect_torch(),
        ffmpeg=shutil.which("ffmpeg"),
        ollama=detect_ollama(),
        recommendation=recommend(gpus, ram_gb),
    )


def log_report(report: DeviceReport) -> None:
    """Log a readable summary of the report."""
    logger.info("OS            : %s", report.os)
    logger.info("Python        : %s", report.python)
    logger.info(
        "CPU cores     : %s physical / %s logical",
        report.cpu_physical_cores,
        report.cpu_logical_cores,
    )
    logger.info("RAM total     : %s GB", report.ram_total_gb)

    if report.gpus:
        for gpu in report.gpus:
            logger.info(
                "NVIDIA GPU    : %s, %d MiB VRAM, driver %s", gpu.name, gpu.vram_mib, gpu.driver
            )
    else:
        logger.warning("NVIDIA GPU    : not detected")

    if report.torch.installed:
        logger.info(
            "PyTorch       : %s (CUDA available=%s, CUDA=%s, VRAM=%s MiB)",
            report.torch.version,
            report.torch.cuda_available,
            report.torch.cuda_version,
            report.torch.vram_mib,
        )
    else:
        logger.info("PyTorch       : not installed in this interpreter")

    if report.ffmpeg:
        logger.info("ffmpeg        : %s", report.ffmpeg)
    else:
        logger.warning("ffmpeg        : not found on PATH")

    if report.ollama.binary or report.ollama.server_running:
        logger.info("Ollama binary : %s", report.ollama.binary or "not found (server reachable)")
        logger.info("Ollama server : %s", "running" if report.ollama.server_running else "stopped")
        logger.info("Ollama models : %s", ", ".join(report.ollama.models) or "(none listed)")
    else:
        logger.warning("Ollama        : not found (no binary, server unreachable)")

    rec = report.recommendation
    logger.info(
        "Recommended   : whisper=%s, compute_type=%s, device=%s, llm_size=%s",
        rec.whisper_model,
        rec.compute_type,
        rec.device,
        rec.llm_size_class,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, help="Optional path to write the report as JSON.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> int:
    """Entry point."""
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(message)s",
        stream=sys.stdout,
    )

    report = collect_report()
    log_report(report)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
        logger.info("Report written to %s", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
