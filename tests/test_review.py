import io
import math
import wave

import pytest

from transcriber import engine, jobs
from transcriber.core import SpeakerTurn, TranscriptionError, Word, assign_speakers, transcript_text
from transcriber.media import audio_clip


def wav_bytes(seconds=5):
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16000)
        target.writeframes(b"\x01\x00" * (seconds * 16000))
    return output.getvalue()


@pytest.mark.parametrize("probability", [-0.1, 1.1, math.nan, math.inf])
def test_invalid_confidence_is_rejected(probability):
    with pytest.raises(TranscriptionError, match="Konfidenz"):
        Word(0, 1, "Hi", probability)


@pytest.mark.parametrize(
    "start,end,frames", [(0, 0.5, 40000), (4.5, 5, 40000), (2, 2.5, 72000), (2, 2, 64000)]
)
def test_audio_excerpt_has_context_and_clamps_to_recording(start, end, frames):
    result = audio_clip(wav_bytes(), start, end)
    with wave.open(io.BytesIO(result), "rb") as excerpt:
        assert excerpt.getframerate() == 16000
        assert excerpt.getnchannels() == 1
        assert excerpt.getsampwidth() == 2
        assert excerpt.getnframes() == frames
        assert excerpt.readframes(frames) == b"\x01\x00" * frames


@pytest.mark.parametrize("start,end", [(9, 10), (-1, 1), (2, 1), (math.nan, 1)])
def test_invalid_audio_location_is_reported(start, end):
    with pytest.raises(TranscriptionError):
        audio_clip(wav_bytes(), start, end)


def test_corrupt_audio_is_reported():
    with pytest.raises(TranscriptionError, match="nicht gelesen"):
        audio_clip(b"invalid", 0, 1)


def test_correction_changes_only_selected_word_and_export():
    words = [Word(0, 1, " Hallo", 0.9), Word(1, 2, " Weld.", 0.3), Word(2, 3, " Weld.", 0.4)]
    job = jobs.Job("de", False, "out.txt", state="done")
    job.segments = assign_speakers(words, None)
    job.correct_word(0, 1, "Welt.")
    assert job.segments[0].text == "Hallo Welt. Weld."
    corrected = job.segments[0].words[1]
    assert (corrected.start, corrected.end, corrected.probability) == (1, 2, 0.3)
    assert job.result == transcript_text(job.segments)
    assert job.reviewed == {(0, 1)}
    with pytest.raises(TranscriptionError, match="eingeben"):
        job.correct_word(0, 1, " ")
    assert job.segments[0].text == "Hallo Welt. Weld."


def test_review_keeps_confidence_timestamps_and_speaker_assignment():
    words = [Word(0, 1, "Hi", 0.2), Word(1, 2, "Welt", 0.8)]
    turns = [SpeakerTurn(0, 1, "A"), SpeakerTurn(1, 2, "B")]
    job = jobs.Job("de", True, "out.txt", state="done")
    job.segments = assign_speakers(words, turns)
    job.correct_word(1, 0, "Welt!")
    assert job.segments[0].words == (words[0],)
    corrected = job.segments[1]
    assert corrected.speaker == "Sprecher 2"
    assert (corrected.start, corrected.end, corrected.words[0].probability) == (1, 2, 0.8)
    assert "Sprecher 2: Welt!" in job.result


def test_job_retains_review_data_and_discards_it(monkeypatch):
    monkeypatch.setattr(jobs, "JOBS", {})
    monkeypatch.setattr(jobs, "spawn", lambda target: target())
    audio = wav_bytes()
    words = [Word(0, 1, " Hallo", 0.4)]

    def transcribe(*args, retain_audio, **kwargs):
        retain_audio(audio)
        return assign_speakers(words, None)

    monkeypatch.setattr(engine, "transcribe", transcribe)
    job = jobs.start(io.BytesIO(b"source"), "de", False, "out.txt", 100)
    assert job.state == "done"
    assert job.audio == audio
    assert job.segments[0].words == tuple(words)
    jobs.discard(job.id)
    assert jobs.get(job.id) is None


def test_failed_job_releases_retained_audio(monkeypatch):
    def fail(*args, retain_audio, **kwargs):
        retain_audio(wav_bytes())
        raise TranscriptionError("Testfehler")

    monkeypatch.setattr(engine, "transcribe", fail)
    job = jobs.Job("de", False, "out.txt")
    jobs._run(job, io.BytesIO(), 100)
    assert job.state == "error"
    assert job.audio is None
    assert job.error == "Testfehler"
