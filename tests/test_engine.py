import io
import os
import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from transcriber import engine
from transcriber.core import SpeakerTurn, TranscriptionError, Word


@pytest.fixture
def configured_engine(monkeypatch, tmp_path):
    monkeypatch.setattr(engine, "MODEL_ROOT", tmp_path)
    for name in engine.MODEL_DIRS.values():
        (tmp_path / name).mkdir()
        (tmp_path / name / ".ready").touch()

    @contextmanager
    def audio(*_):
        yield tmp_path / "audio.wav"

    monkeypatch.setattr(engine, "prepared_audio", audio)
    return tmp_path


@pytest.mark.parametrize("language", ["de", "en"])
def test_whisper_api_uses_local_model_and_explicit_language(monkeypatch, tmp_path, language):
    api = Mock(
        return_value={
            "segments": [{"text": " Hi", "words": [{"start": 0, "end": 1, "word": " Hi"}]}]
        }
    )
    monkeypatch.setitem(sys.modules, "mlx_whisper", SimpleNamespace(transcribe=api))
    assert engine.whisper_words(tmp_path / "a.wav", tmp_path, language) == [Word(0, 1, " Hi")]
    assert api.call_args.kwargs["language"] == language
    assert api.call_args.kwargs["task"] == "transcribe"
    assert api.call_args.kwargs["path_or_hf_repo"] == str(tmp_path)
    assert api.call_args.kwargs["word_timestamps"] is True


def test_speakers_off_does_not_require_model_or_call_pyannote(configured_engine, monkeypatch):
    (configured_engine / "speakers" / ".ready").unlink()
    monkeypatch.setattr(engine, "whisper_words", lambda *_: [Word(0, 1, " Hi")])
    diarize = Mock(side_effect=AssertionError("must not call diarization"))
    monkeypatch.setattr(engine, "speaker_turns", diarize)
    result = engine.transcribe(io.BytesIO(), "de", False, lambda *_: None, 100)
    assert result[0].speaker is None
    diarize.assert_not_called()


def test_missing_model_gives_setup_command(monkeypatch, tmp_path):
    monkeypatch.setattr(engine, "MODEL_ROOT", tmp_path)
    with pytest.raises(TranscriptionError, match="setup_models --speakers"):
        engine.model_path("speakers")


def test_speaker_failure_is_not_a_success_without_labels(configured_engine, monkeypatch):
    monkeypatch.setattr(engine, "whisper_words", lambda *_: [Word(0, 1, " Hi")])
    monkeypatch.setattr(engine, "speaker_turns", Mock(side_effect=RuntimeError("private details")))
    with pytest.raises(TranscriptionError, match="Sprechererkennung ist fehlgeschlagen") as error:
        engine.transcribe(io.BytesIO(), "en", True, lambda *_: None, 100)
    assert "private details" not in str(error.value)


def test_empty_diarization_with_text_is_an_error(configured_engine, monkeypatch):
    monkeypatch.setattr(engine, "whisper_words", lambda *_: [Word(0, 1, " Hi")])
    monkeypatch.setattr(engine, "speaker_turns", lambda *_: [])
    with pytest.raises(TranscriptionError, match="aber keine Sprecher"):
        engine.transcribe(io.BytesIO(), "de", True, lambda *_: None, 100)


def test_silence_produces_empty_result(configured_engine, monkeypatch):
    monkeypatch.setattr(engine, "whisper_words", lambda *_: [])
    assert engine.transcribe(io.BytesIO(), "de", True, lambda *_: None, 100) == ()


def test_transcription_error_is_actionable(configured_engine, monkeypatch):
    monkeypatch.setattr(engine, "whisper_words", Mock(side_effect=MemoryError("out of memory")))
    # Python allocation failures should not expose raw exception details either.
    with pytest.raises(TranscriptionError, match="Arbeitsspeicher"):
        engine.transcribe(io.BytesIO(), "de", False, lambda *_: None, 100)


def test_uses_exclusive_diarization(monkeypatch, tmp_path):
    annotation = Mock()
    annotation.itertracks.return_value = [(SimpleNamespace(start=0, end=1), 0, "A")]
    pipeline = Mock(return_value=SimpleNamespace(exclusive_speaker_diarization=annotation))
    monkeypatch.setattr(engine, "speaker_pipeline", lambda _: pipeline)
    assert engine.speaker_turns(tmp_path / "a.wav", tmp_path) == [SpeakerTurn(0, 1, "A")]
    annotation.itertracks.assert_called_once_with(yield_label=True)


def test_privacy_defaults():
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert os.environ["PYANNOTE_METRICS_ENABLED"] == "0"
    assert os.environ["OTEL_SDK_DISABLED"] == "true"


def test_progress_is_reported_and_logged_without_private_content(
    configured_engine, monkeypatch, caplog
):
    def fake_whisper(audio, model, language, report, vocabulary):
        report(0.5)
        report(1.0)
        return [Word(0, 1, " Geheim")]

    def fake_speakers(audio, model, report):
        report(0.3)
        return [SpeakerTurn(0, 1, "A")]

    monkeypatch.setattr(engine, "whisper_words", fake_whisper)
    monkeypatch.setattr(engine, "speaker_turns", fake_speakers)
    updates = []
    with caplog.at_level("INFO", logger="transcriber"):
        engine.transcribe(io.BytesIO(), "de", True, lambda *a: updates.append(a), 100)
    assert ("Transkribieren …", 0.5) in updates
    assert ("Sprecher erkennen …", 0.3) in updates
    assert "Transkribieren 50 %" in caplog.text
    assert "Sprecher erkennen fertig" in caplog.text
    assert "Geheim" not in caplog.text


def test_speaker_hook_maps_pyannote_steps_to_fraction(monkeypatch, tmp_path):
    annotation = Mock()
    annotation.itertracks.return_value = []

    def pipeline(_audio, hook):
        hook("segmentation", None, total=10, completed=5)
        hook("embeddings", None, total=4, completed=4)
        hook("discrete_diarization", object())
        return SimpleNamespace(exclusive_speaker_diarization=annotation)

    monkeypatch.setattr(engine, "speaker_pipeline", lambda _: pipeline)
    seen = []
    engine.speaker_turns(tmp_path / "a.wav", tmp_path, seen.append)
    assert seen == [0.15, 0.95]


def test_whisper_progress_bar_is_swapped_and_restored(monkeypatch, tmp_path):
    original = SimpleNamespace(tqdm=object())
    module = SimpleNamespace(tqdm=original)

    def api(*_, **kwargs):
        assert kwargs["verbose"] is False
        with module.tqdm.tqdm(total=200, unit="frames", disable=False) as bar:
            bar.update(100)
        return {"segments": []}

    monkeypatch.setitem(sys.modules, "mlx_whisper", SimpleNamespace(transcribe=api))
    monkeypatch.setitem(sys.modules, "mlx_whisper.transcribe", module)
    seen = []
    engine.whisper_words(tmp_path / "a.wav", tmp_path, "de", seen.append)
    assert seen == [0.5]
    assert module.tqdm is original


def test_vocabulary_prompt_is_cleaned():
    assert engine.vocabulary_prompt("  \n , ") is None
    assert engine.vocabulary_prompt("Müller, Kubernetes\nMüller\n Q3 ") == "Müller, Kubernetes, Q3."
    assert len(engine.vocabulary_prompt("x" * 2000)) <= engine.MAX_VOCABULARY_CHARS + 1


def test_vocabulary_is_carried_to_every_window(monkeypatch, tmp_path):
    created = []

    def options(**kwargs):
        created.append(kwargs.get("prompt"))
        return kwargs

    module = SimpleNamespace(DecodingOptions=options)

    def api(*_, **kwargs):
        assert kwargs["initial_prompt"] == "Yilmaz."
        # first window gets the initial prompt, later windows an empty one
        module.DecodingOptions(prompt=[1, 2], temperature=0)
        module.DecodingOptions(prompt=[], temperature=0)
        module.DecodingOptions(prompt=[7], temperature=0)
        return {"segments": []}

    monkeypatch.setitem(sys.modules, "mlx_whisper", SimpleNamespace(transcribe=api))
    monkeypatch.setitem(sys.modules, "mlx_whisper.transcribe", module)
    engine.whisper_words(tmp_path / "a.wav", tmp_path, "de", vocabulary="Yilmaz")
    assert created == [[1, 2], [1, 2], [7]]
    assert module.DecodingOptions is options


def test_without_vocabulary_no_prompt_is_used(monkeypatch, tmp_path):
    api = Mock(return_value={"segments": []})
    monkeypatch.setitem(sys.modules, "mlx_whisper", SimpleNamespace(transcribe=api))
    engine.whisper_words(tmp_path / "a.wav", tmp_path, "de")
    assert api.call_args.kwargs["initial_prompt"] is None


def test_falls_back_to_legacy_small_model(monkeypatch, tmp_path):
    monkeypatch.setattr(engine, "MODEL_ROOT", tmp_path)
    assert engine.whisper_model() is None
    (tmp_path / "whisper").mkdir()
    (tmp_path / "whisper" / ".ready").touch()
    assert engine.whisper_model() == (tmp_path / "whisper", engine.LEGACY_WHISPER[1])
    turbo = tmp_path / engine.MODEL_DIRS["whisper"]
    turbo.mkdir()
    (turbo / ".ready").touch()
    assert engine.model_path("whisper") == turbo
