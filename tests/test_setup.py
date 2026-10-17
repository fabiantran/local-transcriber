import sys
from unittest.mock import Mock

import httpx
import pytest

from transcriber import setup_models

WHISPER_DIR = setup_models.MODEL_DIRS["whisper"]


@pytest.fixture
def model_setup(monkeypatch, tmp_path):
    monkeypatch.setattr(setup_models, "MODEL_ROOT", tmp_path)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(sys, "argv", ["setup_models"])
    return tmp_path


def test_existing_model_skips_download(model_setup, monkeypatch):
    (model_setup / WHISPER_DIR).mkdir()
    (model_setup / WHISPER_DIR / ".ready").touch()
    download = Mock(side_effect=AssertionError("should not download"))
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    assert setup_models.main() == 0
    download.assert_not_called()


def test_network_error_is_actionable_and_does_not_mark_ready(model_setup, monkeypatch, capsys):
    download = Mock(side_effect=httpx.ConnectError("private network details"))
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    assert setup_models.main() == 1
    error = capsys.readouterr().err
    assert "Internetverbindung prüfen" in error
    assert "private network details" not in error
    assert not (model_setup / WHISPER_DIR / ".ready").exists()


def test_force_removes_stale_ready_before_download(model_setup, monkeypatch):
    (model_setup / WHISPER_DIR).mkdir()
    marker = model_setup / WHISPER_DIR / ".ready"
    marker.touch()
    monkeypatch.setattr(sys, "argv", ["setup_models", "--force"])

    def fail_download(*_, **kwargs):
        assert not marker.exists()
        assert kwargs["force_download"] is True
        raise httpx.ConnectError("network unavailable")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fail_download)
    assert setup_models.main() == 1
    assert not marker.exists()


def test_installs_large_turbo_model_into_separate_folder():
    assert setup_models.WHISPER_REPO == "mlx-community/whisper-large-v3-turbo"
    assert WHISPER_DIR != "whisper"
