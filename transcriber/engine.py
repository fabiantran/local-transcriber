from __future__ import annotations

import logging
import sys
import threading
import time
import wave
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Literal, NotRequired, Protocol, TypedDict

from transcriber.core import (
    Segment,
    SpeakerTurn,
    TranscriptionError,
    Word,
    assign_speakers,
)
from transcriber.media import prepared_audio

if TYPE_CHECKING:
    from pyannote.audio import Pipeline

MODEL_ROOT = Path(__file__).resolve().parent.parent / ".models"
WHISPER_REPO = "mlx-community/whisper-large-v3-turbo"
WHISPER_LABEL = "Whisper large-v3-turbo"
# The previously installed small model keeps the app usable until the upgrade is downloaded.
LEGACY_WHISPER = ("whisper", "Whisper small (weniger genau)")
MODEL_DIRS = {"whisper": "whisper-large-v3-turbo", "speakers": "speakers"}
SPEAKER_REPO = "pyannote/speaker-diarization-community-1"
INFERENCE_LOCK = threading.Lock()
Language = Literal["de", "en"]
log = logging.getLogger("transcriber")


class Progress(Protocol):
    def __call__(self, message: str, fraction: float | None = None, /) -> None: ...


def clock(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"


class FractionReporter:
    """Forwards fractional progress and logs it in 10 % steps."""

    def __init__(self, phase: str, progress: Progress):
        self.phase = phase
        self.progress = progress
        self.logged = 0

    def __call__(self, fraction: float) -> None:
        fraction = min(max(fraction, 0.0), 1.0)
        self.progress(self.phase, fraction)
        step = int(fraction * 10)
        if step > self.logged:
            self.logged = step
            log.info("  %s %d %%", self.phase.rstrip(" …"), step * 10)


@contextmanager
def phase(name: str) -> Iterator[None]:
    log.info("▶ %s", name)
    started = time.monotonic()
    try:
        yield
    except BaseException as error:
        log.error(
            "✖ %s nach %s abgebrochen (%s)",
            name,
            clock(time.monotonic() - started),
            type(error).__name__,
        )
        raise
    log.info("✔ %s fertig in %s", name, clock(time.monotonic() - started))


class _WhisperBar:
    """Stand-in for mlx_whisper's tqdm bar that reports progress instead of printing."""

    report: FractionReporter | None = None

    def __init__(self, total: int = 0, **_: Any):
        self.total = max(total, 1)
        self.done = 0

    def __enter__(self) -> _WhisperBar:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def update(self, amount: int) -> None:
        self.done += amount
        if self.report is not None:
            self.report(self.done / self.total)


class WhisperWord(TypedDict):
    start: float
    end: float
    word: str
    probability: NotRequired[float]


class WhisperSegment(TypedDict):
    text: str
    words: list[WhisperWord]


class WhisperOutput(TypedDict):
    segments: list[WhisperSegment]


def audio_seconds(audio: Path) -> float:
    try:
        with wave.open(str(audio), "rb") as wav:
            return wav.getnframes() / wav.getframerate()
    except (OSError, wave.Error, EOFError):
        return 0.0


def whisper_model() -> tuple[Path, str] | None:
    preferred = MODEL_ROOT / MODEL_DIRS["whisper"]
    if (preferred / ".ready").is_file():
        return preferred, WHISPER_LABEL
    legacy = MODEL_ROOT / LEGACY_WHISPER[0]
    if (legacy / ".ready").is_file():
        return legacy, LEGACY_WHISPER[1]
    return None


def model_path(name: Literal["whisper", "speakers"]) -> Path:
    if name == "whisper" and (found := whisper_model()) is not None:
        return found[0]
    path = MODEL_ROOT / MODEL_DIRS[name]
    if not (path / ".ready").is_file():
        command = "uv run python -m transcriber.setup_models"
        if name == "speakers":
            command += " --speakers"
        raise TranscriptionError(
            f"Das lokale {'Sprachmodell' if name == 'whisper' else 'Sprechermodell'} fehlt. "
            f"Bitte einmalig im Terminal ausführen: `{command}`"
        )
    return path


MAX_VOCABULARY_CHARS = 500


def vocabulary_prompt(vocabulary: str) -> str | None:
    terms = [t.strip() for t in vocabulary.replace("\n", ",").split(",") if t.strip()]
    if not terms:
        return None
    return ", ".join(dict.fromkeys(terms))[:MAX_VOCABULARY_CHARS] + "."


class _CarryPrompt:
    """Reuses the first window's prompt (the vocabulary) for every later 30 s window.

    mlx_whisper drops the initial prompt after the first window when
    condition_on_previous_text is off, which would make the vocabulary useless for
    long recordings.
    """

    def __init__(self, original: Any):
        self.original = original
        self.prompt: list[int] | None = None

    def __call__(self, **kwargs: Any) -> Any:
        if self.prompt is None:
            self.prompt = list(kwargs.get("prompt") or [])
        elif not kwargs.get("prompt"):
            kwargs["prompt"] = self.prompt
        return self.original(**kwargs)


def whisper_words(
    audio: Path,
    model: Path,
    language: Language,
    report: FractionReporter | None = None,
    vocabulary: str = "",
) -> list[Word]:
    import mlx_whisper

    prompt = vocabulary_prompt(vocabulary)
    # mlx_whisper exposes neither progress nor prompt carrying, so module globals are
    # swapped while the inference lock guarantees a single concurrent transcription.
    module = sys.modules.get("mlx_whisper.transcribe")
    patches: dict[str, Any] = {}
    if module is not None:
        if report is not None and hasattr(module, "tqdm"):
            _WhisperBar.report = report
            patches["tqdm"] = type("tqdm", (), {"tqdm": _WhisperBar})
        if prompt is not None and hasattr(module, "DecodingOptions"):
            patches["DecodingOptions"] = _CarryPrompt(module.DecodingOptions)
    originals = {name: getattr(module, name) for name in patches}
    for name, value in patches.items():
        setattr(module, name, value)
    try:
        result: WhisperOutput = mlx_whisper.transcribe(
            str(audio),
            path_or_hf_repo=str(model),
            language=language,
            task="transcribe",
            word_timestamps=True,
            condition_on_previous_text=False,
            hallucination_silence_threshold=2.0,
            initial_prompt=prompt,
            verbose=False if report is not None else None,
        )
    finally:
        for name, value in originals.items():
            setattr(module, name, value)
        _WhisperBar.report = None
    words = []
    for segment in result["segments"]:
        if segment["text"].strip() and not segment.get("words"):
            raise TranscriptionError("Das Sprachmodell hat Text ohne Wortzeitstempel geliefert.")
        for item in segment.get("words", []):
            probability = item.get("probability")
            words.append(
                Word(
                    float(item["start"]),
                    float(item["end"]),
                    item["word"],
                    float(probability) if probability is not None else None,
                )
            )
    return words


@lru_cache(maxsize=1)
def speaker_pipeline(model: Path) -> Pipeline:
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(str(model), token=False)
    if pipeline is None:
        raise TranscriptionError(
            "Das lokale Sprechermodell konnte nicht geladen werden. Bitte Einrichtung wiederholen."
        )
    # The default CPU device is intentional; MPS support varies between operators.
    return pipeline


SPEAKER_STEPS = {"segmentation": (0.0, 0.3), "embeddings": (0.3, 0.95)}


def speaker_turns(
    audio: Path, model: Path, report: FractionReporter | None = None
) -> list[SpeakerTurn]:
    def hook(
        step: str,
        _artifact: object,
        file: object = None,
        total: int | None = None,
        completed: int | None = None,
    ) -> None:
        if report is None or step not in SPEAKER_STEPS:
            return
        start, end = SPEAKER_STEPS[step]
        done = 1.0 if not total or completed is None else min(completed / total, 1.0)
        report(start + (end - start) * done)

    output = speaker_pipeline(model)(str(audio), hook=hook)
    return [
        SpeakerTurn(float(turn.start), float(turn.end), str(speaker))
        for turn, _, speaker in output.exclusive_speaker_diarization.itertracks(yield_label=True)
    ]


def transcribe(
    source: BinaryIO,
    language: Language,
    detect_speakers: bool,
    progress: Progress,
    max_bytes: int,
    vocabulary: str = "",
    retain_audio: Callable[[bytes], None] | None = None,
) -> tuple[Segment, ...]:
    if language not in ("de", "en"):
        raise TranscriptionError("Bitte Deutsch oder Englisch auswählen.")
    whisper = model_path("whisper")
    speakers = model_path("speakers") if detect_speakers else None
    prompt = vocabulary_prompt(vocabulary)
    log.info(
        "Auftrag: Sprache=%s, Sprechererkennung=%s, Modell=%s, Fachbegriffe=%s",
        language,
        "an" if speakers else "aus",
        whisper.name,
        "ja" if prompt else "nein",
    )
    log.info("▶ Datei prüfen und Audio aufbereiten")
    started = time.monotonic()
    with prepared_audio(source, progress, max_bytes) as audio:
        log.info(
            "✔ Audio aufbereitet in %s, Länge %s",
            clock(time.monotonic() - started),
            clock(audio_seconds(audio)),
        )
        progress("Auf lokale Verarbeitung warten …")
        if INFERENCE_LOCK.locked():
            log.info("Warte, bis der vorherige Auftrag fertig ist …")
        with INFERENCE_LOCK:
            progress("Transkribieren …", 0.0)
            try:
                with phase("Transkribieren"):
                    words = whisper_words(
                        audio,
                        whisper,
                        language,
                        FractionReporter("Transkribieren …", progress),
                        vocabulary,
                    )
            except (
                RuntimeError,
                OSError,
                ValueError,
                ImportError,
                KeyError,
                TypeError,
                MemoryError,
            ) as error:
                raise TranscriptionError(
                    "Die lokale Transkription ist fehlgeschlagen. Bitte Modellinstallation, "
                    "freien Arbeitsspeicher und Python-Abhängigkeiten prüfen."
                ) from error
            turns = None
            if speakers is not None and words:
                progress("Sprecher erkennen …", 0.0)
                try:
                    with phase("Sprecher erkennen"):
                        turns = speaker_turns(
                            audio, speakers, FractionReporter("Sprecher erkennen …", progress)
                        )
                except (
                    RuntimeError,
                    OSError,
                    ValueError,
                    ImportError,
                    KeyError,
                    TypeError,
                    MemoryError,
                ) as error:
                    raise TranscriptionError(
                        "Die Sprechererkennung ist fehlgeschlagen. Der Auftrag wurde nicht "
                        "vollständig abgeschlossen. Bitte Sprechermodell und Installation prüfen."
                    ) from error
                if not turns:
                    raise TranscriptionError(
                        "Es wurde Text erkannt, aber keine Sprecher. Bitte die Aufnahme prüfen "
                        "oder bewusst ohne Sprechererkennung erneut starten."
                    )
            progress("Ergebnis erstellen …")
            segments = assign_speakers(words, turns)
            if retain_audio is not None and segments:
                try:
                    retain_audio(audio.read_bytes())
                except (OSError, MemoryError) as error:
                    raise TranscriptionError(
                        "Die Tonspur konnte nicht im Arbeitsspeicher behalten werden. "
                        "Bitte freien Speicherplatz und Arbeitsspeicher prüfen."
                    ) from error
            return segments
