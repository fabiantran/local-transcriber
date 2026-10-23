from __future__ import annotations

import io
import json
import math
import shutil
import subprocess
import tempfile
import wave
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

from transcriber.core import TranscriptionError

FORMATS = ("wav", "mp3", "m4a", "flac", "ogg", "mp4", "mov", "mkv", "webm")


def audio_clip(audio: bytes, start: float, end: float, context: float = 2.0) -> bytes:
    """Return a bounded WAV excerpt without creating another file on disk."""
    if not all(math.isfinite(value) for value in (start, end, context)):
        raise TranscriptionError("Der Audioausschnitt hat ungültige Zeitstempel.")
    if start < 0 or end < start or context < 0:
        raise TranscriptionError("Der Audioausschnitt hat ungültige Zeitstempel.")
    try:
        with wave.open(io.BytesIO(audio), "rb") as source:
            rate = source.getframerate()
            first = max(0, int((start - context) * rate))
            last = min(source.getnframes(), math.ceil((end + context) * rate))
            if first >= last:
                raise TranscriptionError("Die Wortstelle liegt außerhalb der Tonspur.")
            source.setpos(first)
            frames = source.readframes(last - first)
            output = io.BytesIO()
            with wave.open(output, "wb") as target:
                target.setparams(source.getparams())
                target.writeframes(frames)
            return output.getvalue()
    except (OSError, EOFError, wave.Error) as error:
        raise TranscriptionError("Der Audioausschnitt konnte nicht gelesen werden.") from error


def require_tools() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise TranscriptionError(
            f"Es fehlt: {', '.join(missing)}. Bitte mit `brew install ffmpeg` installieren."
        )


@contextmanager
def prepared_audio(
    source: BinaryIO, progress: Callable[[str], None], max_bytes: int
) -> Iterator[Path]:
    require_tools()
    with tempfile.TemporaryDirectory(prefix="local-transcriber-") as directory:
        original = Path(directory) / "input"
        audio = Path(directory) / "audio.wav"
        progress("Datei prüfen …")
        try:
            source.seek(0)
            size = 0
            with original.open("wb") as target:
                while chunk := source.read(1024 * 1024):
                    size += len(chunk)
                    if size > max_bytes:
                        raise TranscriptionError(
                            "Die Datei überschreitet das konfigurierte Upload-Limit."
                        )
                    target.write(chunk)
            if size == 0:
                raise TranscriptionError("Die Datei ist leer.")
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-protocol_whitelist",
                    "file,pipe",
                    "-select_streams",
                    "a:0",
                    "-show_entries",
                    "stream=codec_type",
                    "-of",
                    "json",
                    str(original),
                ],
                capture_output=True,
                check=False,
                timeout=60,
            )
            if probe.returncode != 0:
                raise TranscriptionError(
                    "Die Datei ist beschädigt oder das Format wird nicht erkannt."
                )
            metadata = json.loads(probe.stdout)
            if not metadata.get("streams"):
                raise TranscriptionError("Die Datei enthält keine lesbare Tonspur.")
            progress("Audio aufbereiten …")
            conversion = subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-xerror",
                    "-protocol_whitelist",
                    "file,pipe",
                    "-i",
                    str(original),
                    "-map",
                    "0:a:0",
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    str(audio),
                ],
                capture_output=True,
                check=False,
            )
            if conversion.returncode != 0:
                raise TranscriptionError("Die Tonspur konnte nicht vollständig gelesen werden.")
            with wave.open(str(audio), "rb") as wav:
                if wav.getnframes() == 0:
                    raise TranscriptionError("Die Tonspur ist leer.")
        except subprocess.TimeoutExpired as error:
            raise TranscriptionError("Die Dateiprüfung hat zu lange gedauert.") from error
        except (OSError, ValueError, wave.Error, MemoryError) as error:
            raise TranscriptionError(
                "Die Tonspur konnte nicht aufbereitet werden. "
                "Bitte Datei, freien Speicherplatz und Arbeitsspeicher prüfen."
            ) from error
        yield audio
