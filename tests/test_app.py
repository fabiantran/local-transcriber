import io
import tomllib
from pathlib import Path
from unittest.mock import Mock

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from transcriber import jobs
from transcriber.core import Segment, TranscriptionError

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def synchronous_jobs(monkeypatch):
    monkeypatch.setattr(jobs, "spawn", lambda target: target())
    monkeypatch.setattr(jobs, "JOBS", {})


def test_initial_ui_and_private_config():
    app = AppTest.from_file(str(ROOT / "app.py")).run()
    assert not app.exception
    assert app.selectbox[0].value == "Deutsch"
    assert app.toggle[0].value is False
    assert app.button[0].disabled
    config = tomllib.loads((ROOT / ".streamlit/config.toml").read_text())
    assert config["server"]["address"] == "127.0.0.1"
    assert config["server"]["maxUploadSize"] == 2048
    assert config["browser"]["gatherUsageStats"] is False


def test_explicit_start_result_invalidation_and_no_repeat(monkeypatch):
    upload = io.BytesIO(b"test")
    upload.name = "aufnahme.wav"
    monkeypatch.setattr(st, "file_uploader", lambda *_, **__: upload)
    runner = Mock(return_value=(Segment(0, 1, "Hallo!", "Sprecher 1"),))
    monkeypatch.setattr("transcriber.engine.transcribe", runner)
    app = AppTest.from_file(str(ROOT / "app.py")).run()
    runner.assert_not_called()
    app.button[0].click().run()
    assert not app.exception
    assert "Hallo!" in app.text[0].value
    assert app.get("download_button")[0].proto.label == "Als TXT herunterladen"
    assert jobs.get(app.query_params["job"]).download_name == "aufnahme-transkript.txt"
    app.run()
    assert runner.call_count == 1
    app.selectbox[0].select("Englisch").run()
    assert not app.text
    assert not app.get("download_button")
    app.button[0].click().run()
    assert runner.call_args.args[1] == "en"
    app.toggle[0].set_value(True).run()
    assert not app.text
    app.button[0].click().run()
    assert runner.call_args.args[2] is True
    app.text_area[0].input("Yilmaz, Kubernetes").run()
    app.button[0].click().run()
    assert runner.call_args.kwargs["vocabulary"] == "Yilmaz, Kubernetes"
    app.button[1].click().run()
    assert not app.text
    assert app.session_state["upload_version"] == 1


def test_processing_error_shown_without_download(monkeypatch):
    upload = io.BytesIO(b"test")
    upload.name = "aufnahme.wav"
    monkeypatch.setattr(st, "file_uploader", lambda *_, **__: upload)
    monkeypatch.setattr(
        "transcriber.engine.transcribe",
        Mock(side_effect=TranscriptionError("Das lokale Sprechermodell fehlt.")),
    )
    app = AppTest.from_file(str(ROOT / "app.py")).run()
    app.button[0].click().run()
    assert not app.exception
    assert app.error[0].value == "Das lokale Sprechermodell fehlt."
    assert not app.get("download_button")


def test_result_survives_reload_and_running_job_shows_progress(monkeypatch):
    upload = io.BytesIO(b"test")
    upload.name = "aufnahme.wav"
    monkeypatch.setattr(st, "file_uploader", lambda *_, **__: upload)
    monkeypatch.setattr(
        "transcriber.engine.transcribe",
        Mock(return_value=(Segment(0, 1, "Hallo!", None),)),
    )
    app = AppTest.from_file(str(ROOT / "app.py")).run()
    app.button[0].click().run()
    job_id = app.query_params["job"]

    reloaded = AppTest.from_file(str(ROOT / "app.py"))
    reloaded.query_params["job"] = job_id
    reloaded.run()
    assert "Hallo!" in reloaded.text[0].value

    running = jobs.Job("de", True, "x.txt")
    running.report("Sprecher erkennen …", 0.42)
    jobs.JOBS[running.id] = running
    reloaded = AppTest.from_file(str(ROOT / "app.py"))
    reloaded.query_params["job"] = running.id
    reloaded.run()
    assert not reloaded.exception
    assert any("Sprecher erkennen" in item.value for item in reloaded.markdown)
    assert reloaded.get("progress")[0].proto.value == 42
    assert reloaded.button[0].disabled


def test_unknown_job_after_restart_shows_hint():
    app = AppTest.from_file(str(ROOT / "app.py"))
    app.query_params["job"] = "verschwunden"
    app.run()
    assert not app.exception
    assert "nicht mehr verfügbar" in app.warning[0].value
