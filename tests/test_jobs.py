import io
from unittest.mock import Mock

from transcriber import engine, jobs
from transcriber.core import Segment, Word, assign_speakers


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
    assert len(list(isolated_transcripts.iterdir())) == 2


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
    assert list(isolated_transcripts.iterdir()) == [path]


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
    assert list(isolated_transcripts.iterdir()) == [path]
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
