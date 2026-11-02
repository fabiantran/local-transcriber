import io
from unittest.mock import Mock

import pytest

from transcriber import engine, jobs
from transcriber.core import Segment, TranscriptionError, Word, assign_speakers, transcript_text


def test_finished_transcripts_are_unique_utf8_and_survive_discard(
    monkeypatch, isolated_transcripts
):
    monkeypatch.setattr(jobs, "JOBS", {})
    monkeypatch.setattr(jobs, "spawn", lambda target: target())
    monkeypatch.setattr(
        engine, "transcribe", Mock(return_value=(Segment(0, 1, "Grüße!", "Sprecher 1"),))
    )
    first = jobs.start(io.BytesIO(), "de", True, "aufnahme-transkript.txt", 100)
    second = jobs.start(io.BytesIO(), "de", True, "aufnahme-transkript.txt", 100)
    assert first.state == second.state == "done"
    assert first.saved_path != second.saved_path
    for job in (first, second):
        assert job.saved_path.parent == isolated_transcripts
        assert job.saved_path.name == f"aufnahme-transkript-{job.id}.txt"
        assert job.saved_path.read_bytes() == job.result.encode("utf-8")
        jobs.discard(job.id)
        assert job.saved_path.exists()
    jobs.JOBS.clear()
    assert sorted(p.suffix for p in isolated_transcripts.iterdir()) == [
        ".json",
        ".json",
        ".txt",
        ".txt",
    ]


def test_corrections_update_same_file(isolated_transcripts):
    job = jobs.Job("de", False, "../aufnahme.txt", state="done")
    job.segments = assign_speakers([Word(0, 1, "Grüße", 0.4)], None)
    job.correct_word(0, 0, "Hallo")
    path = job.saved_path
    assert path.parent == isolated_transcripts
    job.correct_word(0, 0, "Tschüss")
    assert job.saved_path == path
    assert path.read_text(encoding="utf-8") == job.result
    assert "Tschüss" in job.result
    assert sorted(isolated_transcripts.iterdir()) == sorted([path, path.with_suffix(".json")])


def test_failed_write_preserves_previous_file_and_current_result(
    monkeypatch, isolated_transcripts, caplog
):
    job = jobs.Job("de", False, "private.txt", state="done", result="Alter Text")
    job.save_transcript()
    path = job.saved_path
    job.result = "Neuer privater Text"
    with monkeypatch.context() as patch:
        patch.setattr(jobs.os, "replace", Mock(side_effect=PermissionError("private.txt")))
        job.save_transcript()
    assert job.result == "Neuer privater Text"
    assert job.save_error
    assert job.saved_path is None
    assert path.read_text(encoding="utf-8") == "Alter Text"
    assert sorted(isolated_transcripts.iterdir()) == sorted([path, path.with_suffix(".json")])
    assert "PermissionError" in caplog.text
    assert "private.txt" not in caplog.text
    assert job.result not in caplog.text
    job.save_transcript()
    assert job.save_error is None
    assert job.saved_path == path
    assert path.read_text(encoding="utf-8") == job.result


def test_unwritable_directory_does_not_lose_transcript(monkeypatch, isolated_transcripts):
    isolated_transcripts.write_text("not a directory")
    monkeypatch.setattr(engine, "transcribe", Mock(return_value=(Segment(0, 1, "Hallo!", None),)))
    job = jobs.Job("de", False, "out.txt")
    source = io.BytesIO()
    jobs._run(job, source, 100)
    assert source.closed
    assert job.state == "done"
    assert "Hallo!" in job.result
    assert job.save_error
    assert job.saved_path is None


def test_no_speech_creates_no_file(monkeypatch, isolated_transcripts):
    monkeypatch.setattr(engine, "transcribe", Mock(return_value=()))
    job = jobs.Job("de", False, "out.txt")
    jobs._run(job, io.BytesIO(), 100)
    assert job.state == "done"
    assert not job.result
    assert job.save_error is None
    assert not isolated_transcripts.exists()


def test_saved_job_resumes_after_restart(monkeypatch, isolated_transcripts):
    monkeypatch.setattr(jobs, "JOBS", {})
    job = jobs.Job("de", True, "aufnahme.wav", state="done")
    job.segments = assign_speakers([Word(0, 1, "Grüße", 0.4), Word(1, 2, " Welt", 0.9)], None)
    job.retain_audio(b"audio-bytes")
    job.correct_word(0, 0, "Hallo")
    jobs.JOBS.clear()

    restored = jobs.load_saved(job.id)

    assert restored is not None
    assert restored.state == "done"
    assert restored.result == job.result
    assert restored.segments == job.segments
    assert restored.reviewed == {(0, 0)}
    assert restored.audio == b"audio-bytes"
    assert restored.saved_path == job.saved_path
    assert jobs.get(job.id) is restored
    restored.correct_word(0, 1, "Welt")
    assert restored.reviewed == {(0, 0), (0, 1)}
    assert "Hallo Welt" in restored.result


def test_list_saved_skips_corrupt_files_and_sorts_newest_first(isolated_transcripts, caplog):
    older = jobs.Job("de", False, "eins.wav", state="done", started=1.0)
    newer = jobs.Job("en", False, "zwei.wav", state="done", started=2.0)
    older.segments = assign_speakers([Word(0, 1, "Eins", None)], None)
    newer.segments = assign_speakers([Word(0, 1, "Zwei", None)], None)
    older.result = transcript_text(older.segments)
    newer.result = transcript_text(newer.segments)
    older.save_transcript()
    newer.save_transcript()
    broken = isolated_transcripts / f"kaputt-{'a' * 32}.json"
    broken.write_text('{"version": 1, "id": ', encoding="utf-8")

    listed = jobs.list_saved()

    assert [job.id for job in listed] == [newer.id, older.id]
    assert listed[0].language == "en"
    assert listed[0].audio is None
    assert "JSONDecodeError" in caplog.text
    assert broken.name not in caplog.text


def test_load_saved_rejects_invalid_or_incomplete_files(monkeypatch, isolated_transcripts):
    monkeypatch.setattr(jobs, "JOBS", {})
    isolated_transcripts.mkdir()
    job_id = "b" * 32
    (isolated_transcripts / f"aufnahme-{job_id}.json").write_text("{}", encoding="utf-8")

    assert jobs.load_saved("../" + job_id) is None
    assert jobs.load_saved("0" * 32) is None
    with pytest.raises(TranscriptionError):
        jobs.load_saved(job_id)


def test_list_saved_without_directory_is_empty(isolated_transcripts):
    assert not isolated_transcripts.exists()
    assert jobs.list_saved() == []
