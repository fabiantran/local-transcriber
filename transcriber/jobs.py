"""In-memory background jobs so processing survives page reloads (lost on server restart)."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import BinaryIO, Literal

from transcriber import engine
from transcriber.core import TranscriptionError, transcript_text

log = logging.getLogger("transcriber")
KEEP_FINISHED_SECONDS = 12 * 60 * 60
JOBS: dict[str, Job] = {}
_LOCK = threading.Lock()


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
        )
        job.result = transcript_text(segments)
        job.state = "done"
    except TranscriptionError as error:
        job.error = str(error)
        job.state = "error"
    except Exception:
        log.exception("Unerwarteter Fehler in Auftrag %s", short)
        job.error = "Unerwarteter Fehler bei der Verarbeitung. Details stehen im Log."
        job.state = "error"
    finally:
        job.finished = time.time()
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


def discard(job_id: str | None) -> None:
    job = get(job_id)
    if job is not None and job.state != "running":
        JOBS.pop(job.id, None)
