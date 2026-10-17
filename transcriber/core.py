from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass


class TranscriptionError(Exception):
    """An actionable error that can be displayed without exposing private data."""


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.start) or not math.isfinite(self.end):
            raise TranscriptionError("Das Modell hat ungültige Zeitstempel geliefert.")
        if self.start < 0 or self.end < self.start:
            raise TranscriptionError("Das Modell hat ungültige Zeitstempel geliefert.")


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.start)
            or not math.isfinite(self.end)
            or self.start < 0
            or self.end <= self.start
            or not self.speaker
        ):
            raise TranscriptionError("Die Sprechererkennung hat ungültige Abschnitte geliefert.")


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None = None


def assign_speakers(
    words: Sequence[Word], turns: Sequence[SpeakerTurn] | None
) -> tuple[Segment, ...]:
    """Assign by total overlap; retain unknown words rather than guessing."""
    ordered = sorted(turns or (), key=lambda turn: (turn.start, turn.end, turn.speaker))
    labels: dict[str, str] = {}
    for turn in ordered:
        if turn.speaker not in labels:
            labels[turn.speaker] = f"Sprecher {len(labels) + 1}"

    segments: list[Segment] = []
    active: list[SpeakerTurn] = []
    turn_index = 0
    for word in sorted(words, key=lambda word: (word.start, word.end)):
        if not word.text.strip():
            continue
        label = None
        if turns is not None:
            while turn_index < len(ordered) and ordered[turn_index].start < word.end:
                active.append(ordered[turn_index])
                turn_index += 1
            active = [turn for turn in active if turn.end > word.start]
            overlaps: dict[str, float] = defaultdict(float)
            for turn in active:
                overlap = min(word.end, turn.end) - max(word.start, turn.start)
                if overlap > 0:
                    overlaps[turn.speaker] += overlap
            label = (
                labels[max(overlaps, key=lambda key: overlaps[key])]
                if overlaps
                else ("Sprecher unbekannt")
            )
        if (
            segments
            and segments[-1].speaker == label
            and word.start - segments[-1].end <= 1.5
            and word.end - segments[-1].start <= 25
        ):
            previous = segments[-1]
            # Whisper words normally include leading spaces, but not always.
            separator = "" if word.text.startswith(" ") or word.text[0] in ".,!?;:" else " "
            segments[-1] = Segment(
                previous.start,
                max(previous.end, word.end),
                previous.text + separator + word.text,
                label,
            )
        else:
            segments.append(Segment(word.start, word.end, word.text.strip(), label))
    return tuple(segments)


def timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def transcript_text(segments: Sequence[Segment]) -> str:
    lines = []
    for segment in segments:
        label = f" {segment.speaker}:" if segment.speaker is not None else ""
        lines.append(
            f"[{timestamp(segment.start)} – {timestamp(segment.end)}]{label} {segment.text}"
        )
    return "\n\n".join(lines) + ("\n" if lines else "")
