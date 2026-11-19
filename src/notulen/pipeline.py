"""Run every stage, from audio file to speaker-labelled transcript and summary."""

from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType

from notulen.asr.transcriber import Transcriber
from notulen.audio.preprocess import preprocess_audio
from notulen.config import AppConfig, LlmConfig
from notulen.diarization.diarizer import Diarizer, write_rttm
from notulen.formatting import render_summary_markdown, render_transcript_markdown
from notulen.merge.aligner import align_transcript
from notulen.schemas import SpeakerTurn, Summary, Transcript
from notulen.summarize.llm_client import LlmError, create_llm_client
from notulen.summarize.summarizer import summarize_transcript

logger = logging.getLogger(__name__)

StageCallback = Callable[[str], None]


@dataclass
class PipelineResult:
    """Everything produced by one pipeline run."""

    transcript: Transcript
    turns: tuple[SpeakerTurn, ...]
    audio_seconds: float = 0.0
    output_dir: Path = Path()
    summary: Summary | None = None
    summary_error: str | None = None
    timings: dict[str, float] = field(default_factory=dict)
    files: dict[str, Path] = field(default_factory=dict)


class NotulenPipeline:
    """Orchestrate preprocessing, ASR, diarization, alignment, and summarization."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._transcriber: Transcriber | None = None
        self._diarizer: Diarizer | None = None

    def __enter__(self) -> NotulenPipeline:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Release every loaded model."""
        if self._transcriber is not None:
            self._transcriber.close()
            self._transcriber = None
        if self._diarizer is not None:
            self._diarizer.close()
            self._diarizer = None

    def _release_if_configured(self) -> None:
        """Free models between stages when memory is limited."""
        if self.config.runtime.release_models_between_stages:
            self.close()

    def _get_transcriber(self) -> Transcriber:
        if self._transcriber is None:
            self._transcriber = Transcriber(self.config.asr, self.config.vad)
        return self._transcriber

    def _get_diarizer(self) -> Diarizer:
        if self._diarizer is None:
            self._diarizer = Diarizer(self.config.diarization, self.config.secrets.hf_token)
        return self._diarizer

    def run(
        self,
        audio_path: Path,
        output_dir: Path | None = None,
        num_speakers: int | None = None,
        diarize: bool = True,
        summarize: bool = True,
        on_stage: StageCallback | None = None,
    ) -> PipelineResult:
        """Process one audio file and write the results to output_dir."""
        output_dir = output_dir or self.config.paths.output_dir / audio_path.stem
        work_dir = self.config.paths.temp_dir / f"{audio_path.stem}-{uuid.uuid4().hex[:8]}"
        timings: dict[str, float] = {}

        @contextmanager
        def stage(name: str) -> Iterator[None]:
            if on_stage is not None:
                on_stage(name)
            logger.info("Stage %s started", name)
            started = time.perf_counter()
            yield
            timings[name] = round(time.perf_counter() - started, 2)
            logger.info("Stage %s finished in %.1f s", name, timings[name])

        try:
            with stage("preprocess"):
                audio = preprocess_audio(audio_path, self.config.audio, self.config.vad, work_dir)

            with stage("asr"):
                transcript = self._get_transcriber().transcribe(audio)
                self._release_if_configured()

            turns: tuple[SpeakerTurn, ...] = ()
            if diarize:
                with stage("diarization"):
                    turns = tuple(self._get_diarizer().diarize(audio.wav_path, num_speakers))
                    self._release_if_configured()

            with stage("alignment"):
                aligned = align_transcript(transcript, turns, self.config.merge)

            result = PipelineResult(
                transcript=aligned,
                turns=turns,
                timings=timings,
                audio_seconds=audio.duration,
                output_dir=output_dir,
            )
            self._write_transcript(result, output_dir)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

        if summarize:
            return self.summarize(result, on_stage)
        self._finish(result)
        return result

    def summarize(
        self,
        result: PipelineResult,
        on_stage: StageCallback | None = None,
        llm: LlmConfig | None = None,
        api_key: str | None = None,
    ) -> PipelineResult:
        """Add an LLM summary to a finished transcript and update the timings.

        Runs separately from the audio stages so callers can keep it off the GPU.
        ``llm`` and ``api_key`` override the configured LLM for this call only;
        the key is used for the request and not stored.
        """
        if result.transcript.segments:
            if on_stage is not None:
                on_stage("summary")
            logger.info("Stage summary started")
            started = time.perf_counter()
            self._summarize(result, llm or self.config.llm, api_key)
            result.timings["summary"] = round(time.perf_counter() - started, 2)
            logger.info("Stage summary finished in %.1f s", result.timings["summary"])
            if result.summary is not None:
                self._write_summary(result, result.output_dir)
        self._finish(result)
        return result

    def _finish(self, result: PipelineResult) -> None:
        """Total the stage timings and write them next to the other outputs."""
        stages = {name: value for name, value in result.timings.items() if name != "total"}
        result.timings["total"] = round(sum(stages.values()), 2)
        result.files["timings"] = self._write_timings(
            result.timings, result.audio_seconds, result.output_dir
        )
        logger.info(
            "Processed %.1f s of audio in %.1f s (RTF %.2f)",
            result.audio_seconds,
            result.timings["total"],
            result.timings["total"] / result.audio_seconds if result.audio_seconds > 0 else 0.0,
        )

    def _summarize(self, result: PipelineResult, llm: LlmConfig, api_key: str | None) -> None:
        """Summarize the transcript, recording a failure instead of losing the transcript."""
        logger.info("Summarizing with %s model %s", llm.backend, llm.model)
        try:
            with create_llm_client(
                llm, self.config.secrets, user_api_key=api_key, configured=self.config.llm
            ) as client:
                result.summary = summarize_transcript(
                    result.transcript, client, self.config.summarize
                )
        except LlmError as exc:
            result.summary_error = str(exc)
            logger.error("Summary failed: %s", exc)

    @staticmethod
    def _write_summary(result: PipelineResult, output_dir: Path) -> None:
        """Write the structured summary as JSON and Markdown."""
        assert result.summary is not None
        summary_json = output_dir / "summary.json"
        summary_json.write_text(result.summary.model_dump_json(indent=2), encoding="utf-8")
        result.files["summary_json"] = summary_json

        summary_md = output_dir / "summary.md"
        summary_md.write_text(
            render_summary_markdown(result.transcript.source, result.summary), encoding="utf-8"
        )
        result.files["summary"] = summary_md

    @staticmethod
    def _write_transcript(result: PipelineResult, output_dir: Path) -> None:
        """Write the transcript as JSON and Markdown, plus the diarization RTTM."""
        output_dir.mkdir(parents=True, exist_ok=True)
        result.files["transcript_json"] = result.transcript.save_json(
            output_dir / "transcript.json"
        )

        markdown = output_dir / "transcript.md"
        markdown.write_text(render_transcript_markdown(result.transcript), encoding="utf-8")
        result.files["transcript_md"] = markdown

        if result.turns:
            result.files["rttm"] = write_rttm(
                list(result.turns),
                output_dir / "diarization.rttm",
                Path(result.transcript.source).stem,
            )

    @staticmethod
    def _write_timings(timings: dict[str, float], duration: float, output_dir: Path) -> Path:
        """Write per-stage timings as JSON."""
        total = timings["total"]
        payload = {
            "audio_seconds": round(duration, 2),
            "stages_seconds": {name: value for name, value in timings.items() if name != "total"},
            "total_seconds": total,
            # Processing time per second of audio; below 1 is faster than real time.
            "real_time_factor": round(total / duration, 3) if duration > 0 else None,
        }
        path = output_dir / "timings.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path
