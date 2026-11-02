"""In-memory background jobs so processing survives page reloads (lost on server restart)."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, BinaryIO, Literal

from transcriber import engine
from transcriber.core import (
    Segment,
    TranscriptionError,
    Word,
    transcript_text,
    words_text,
)

log = logging.getLogger("transcriber")
KEEP_FINISHED_SECONDS = 12 * 60 * 60
TRANSCRIPTS_DIR = Path(__file__).resolve().parent.parent / "transkripte"
SAVED_VERSION = 1
JOBS: dict[str, Job] = {}
_LOCK = threading.Lock()
_JOB_ID = re.compile(r"[0-9a-f]{32}")


def _write_atomic(destination: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=destination.parent, prefix=".transkript-")
    try:
        with os.fdopen(fd, "wb") as target:
            target.write(data)
        os.replace(tmp, destination)
    finally:
        Path(tmp).unlink(missing_ok=True)


def _segment(item: dict[str, Any]) -> Segment:
    words = tuple(
        Word(
            float(word["start"]),
            float(word["end"]),
            str(word["text"]),
            None if word["probability"] is None else float(word["probability"]),
        )
        for word in item["words"]
    )
    speaker = item["speaker"]
    return Segment(
        float(item["start"]),
        float(item["end"]),
        str(item["text"]),
        None if speaker is None else str(speaker),
        words,
    )


def _read_job(path: Path, load_audio: bool) -> Job:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["version"] != SAVED_VERSION or data["language"] not in ("de", "en"):
            raise ValueError("Unbekannte Version oder Sprache")
        segments = tuple(_segment(item) for item in data["segments"])
        job = Job(
            language=data["language"],
            detect_speakers=bool(data["detect_speakers"]),
            download_name=str(data["download_name"]),
            id=str(data["id"]),
            started=float(data["started"]),
            state="done",
            finished=None if data["finished"] is None else float(data["finished"]),
            segments=segments,
            reviewed={(int(si), int(wi)) for si, wi in data["reviewed"]},
        )
        job.result = transcript_text(segments)
        text_file = path.with_suffix(".txt")
        job.saved_path = text_file if text_file.exists() else None
        if load_audio:
            audio_file = path.with_suffix(".wav")
            job.audio = audio_file.read_bytes() if audio_file.exists() else None
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise TranscriptionError(
            "Das gespeicherte Transkript ist unvollständig oder beschädigt."
        ) from error
    return job


@dataclass
class Job:
    language: engine.Language
    detect_speakers: bool
    download_name: str
    vocabulary: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started: float = field(default_factory=time.time)
    phase: str = "Wird gestartet …"
    fraction: float | None = None
    state: Literal["running", "done", "error"] = "running"
    result: str | None = None
    error: str | None = None
    finished: float | None = None
    phase_started: float = field(default_factory=time.time)
    segments: tuple[Segment, ...] = ()
    audio: bytes | None = field(default=None, repr=False)
    reviewed: set[tuple[int, int]] = field(default_factory=set)
    saved_path: Path | None = None
    save_error: str | None = None

    def _saved_file(self, suffix: str) -> Path:
        name = Path(self.download_name.replace("\\", "/")).stem
        return TRANSCRIPTS_DIR / f"{name}-{self.id}{suffix}"

    def _record(self) -> dict[str, object]:
        return {
            "version": SAVED_VERSION,
            "id": self.id,
            "download_name": self.download_name,
            "language": self.language,
            "detect_speakers": self.detect_speakers,
            "started": self.started,
            "finished": self.finished,
            "segments": [asdict(segment) for segment in self.segments],
            "reviewed": [list(key) for key in sorted(self.reviewed)],
        }

    def save_transcript(self) -> None:
        if not self.result:
            return
        try:
            TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
            _write_atomic(self._saved_file(".txt"), self.result.encode("utf-8"))
            if self.audio is not None and not self._saved_file(".wav").exists():
                _write_atomic(self._saved_file(".wav"), self.audio)
            record = json.dumps(self._record(), ensure_ascii=False)
            _write_atomic(self._saved_file(".json"), record.encode("utf-8"))
        except OSError as error:
            log.warning(
                "Speicherung für Auftrag %s fehlgeschlagen: %s",
                self.id[:8],
                type(error).__name__,
            )
            self.saved_path = None
            self.save_error = (
                "Das aktuelle Transkript konnte nicht in transkripte/ gespeichert werden. "
                "Bitte als TXT herunterladen und Speicherplatz sowie Ordnerrechte prüfen."
            )
        else:
            self.saved_path = self._saved_file(".txt")
            self.save_error = None

    def retain_audio(self, audio: bytes) -> None:
        self.audio = audio

    def correct_word(self, segment_index: int, word_index: int, text: str) -> None:
        if self.state != "done":
            raise TranscriptionError("Korrekturen sind erst nach der Verarbeitung möglich.")
        if not text.strip():
            raise TranscriptionError("Bitte ein Wort oder eine korrigierte Formulierung eingeben.")
        segment = self.segments[segment_index]
        word = segment.words[word_index]
        words = list(segment.words)
        prefix = " " if word.text.startswith(" ") else ""
        words[word_index] = replace(word, text=prefix + text.strip())
        segments = list(self.segments)
        segments[segment_index] = replace(segment, words=tuple(words), text=words_text(words))
        self.segments = tuple(segments)
        self.result = transcript_text(self.segments)
        self.reviewed.add((segment_index, word_index))
        self.save_transcript()

    @property
    def elapsed(self) -> float:
        return (self.finished or time.time()) - self.started

    @property
    def remaining(self) -> float | None:
        if self.fraction is None or self.fraction < 0.03:
            return None
        spent = time.time() - self.phase_started
        return spent * (1 - self.fraction) / self.fraction

    def report(self, message: str, fraction: float | None = None) -> None:
        if message != self.phase:
            self.phase_started = time.time()
        self.phase = message
        self.fraction = fraction


def spawn(target: Callable[[], None]) -> None:
    threading.Thread(target=target, name="transcription", daemon=True).start()


def start(
    source: BinaryIO,
    language: engine.Language,
    detect_speakers: bool,
    download_name: str,
    max_bytes: int,
    vocabulary: str = "",
) -> Job:
    job = Job(language, detect_speakers, download_name, vocabulary)
    with _LOCK:
        cutoff = time.time() - KEEP_FINISHED_SECONDS
        for old in [j for j in JOBS.values() if j.finished and j.finished < cutoff]:
            JOBS.pop(old.id, None)
        JOBS[job.id] = job
    spawn(lambda: _run(job, source, max_bytes))
    return job


def _run(job: Job, source: BinaryIO, max_bytes: int) -> None:
    short = job.id[:8]
    log.info("═ Auftrag %s gestartet", short)
    try:
        segments = engine.transcribe(
            source,
            job.language,
            job.detect_speakers,
            job.report,
            max_bytes,
            vocabulary=job.vocabulary,
            retain_audio=job.retain_audio,
        )
        job.segments = segments
        job.result = transcript_text(segments)
        job.finished = time.time()
        job.save_transcript()
        job.state = "done"
    except TranscriptionError as error:
        job.error = str(error)
        job.state = "error"
    except Exception:
        log.exception("Unerwarteter Fehler in Auftrag %s", short)
        job.error = "Unerwarteter Fehler bei der Verarbeitung. Details stehen im Log."
        job.state = "error"
    finally:
        if job.state != "done":
            job.audio = None
        job.finished = job.finished or time.time()
        job.fraction = None
        if job.state == "running":
            job.state = "error"
            job.error = "Die Verarbeitung wurde abgebrochen."
        close = getattr(source, "close", None)
        if close is not None:
            close()
    if job.state == "done":
        log.info("═ Auftrag %s fertig nach %s", short, engine.clock(job.elapsed))
    else:
        log.warning(
            "═ Auftrag %s fehlgeschlagen nach %s: %s",
            short,
            engine.clock(job.elapsed),
            job.error,
        )


def get(job_id: str | None) -> Job | None:
    return JOBS.get(job_id) if job_id else None


def load_saved(job_id: str | None) -> Job | None:
    """Return a job from memory or from its saved files; None if nothing was saved."""
    if not job_id or not _JOB_ID.fullmatch(job_id):
        return None
    with _LOCK:
        if job_id in JOBS:
            return JOBS[job_id]
    paths = sorted(TRANSCRIPTS_DIR.glob(f"*-{job_id}.json"))
    if not paths:
        return None
    job = _read_job(paths[0], load_audio=True)
    if job.id != job_id:
        raise TranscriptionError("Das gespeicherte Transkript ist unvollständig oder beschädigt.")
    with _LOCK:
        JOBS[job.id] = job
    return job


def list_saved() -> list[Job]:
    """Finished jobs found in the transcripts folder, newest first; audio is not loaded."""
    if not TRANSCRIPTS_DIR.is_dir():
        return []
    found: list[Job] = []
    for path in TRANSCRIPTS_DIR.glob("*.json"):
        try:
            found.append(_read_job(path, load_audio=False))
        except TranscriptionError as error:
            log.warning(
                "Gespeichertes Transkript übersprungen: %s",
                type(error.__cause__).__name__,
            )
    return sorted(found, key=lambda job: job.started, reverse=True)


def discard(job_id: str | None) -> None:
    job = get(job_id)
    if job is not None and job.state != "running":
        JOBS.pop(job.id, None)
