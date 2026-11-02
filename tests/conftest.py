import logging

import pytest

from transcriber import jobs, vocabulary

# Keep test runs out of the real log file; configure_logging() skips configured loggers.
logging.getLogger("transcriber").addHandler(logging.NullHandler())


@pytest.fixture(autouse=True)
def isolated_transcripts(tmp_path, monkeypatch):
    path = tmp_path / "transkripte"
    monkeypatch.setattr(jobs, "TRANSCRIPTS_DIR", path)
    return path


@pytest.fixture(autouse=True)
def isolated_vocabulary(tmp_path, monkeypatch):
    path = tmp_path / "fachbegriffe.txt"
    monkeypatch.setattr(vocabulary, "VOCABULARY_FILE", path)
    return path
