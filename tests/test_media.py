import io
import shutil
import subprocess
import wave

import pytest

from transcriber.core import TranscriptionError
from transcriber.media import prepared_audio


@pytest.fixture
def wav_bytes():
    data = io.BytesIO()
    with wave.open(data, "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(b"\0" * 8000 * 4)
    return data.getvalue()


@pytest.fixture(autouse=True)
def require_ffmpeg():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg integration tests require ffmpeg and ffprobe")


def test_normalizes_audio_and_cleans_up(wav_bytes):
    phases = []
    with prepared_audio(io.BytesIO(wav_bytes), phases.append, 100_000) as path:
        parent = path.parent
        with wave.open(str(path)) as audio:
            assert audio.getnchannels() == 1
            assert audio.getframerate() == 16000
            assert audio.getnframes() == 16000
    assert not parent.exists()
    assert phases == ["Datei prüfen …", "Audio aufbereiten …"]


def test_cleanup_when_inference_fails(wav_bytes):
    with pytest.raises(RuntimeError, match="inference failed"):
        with prepared_audio(io.BytesIO(wav_bytes), lambda _: None, 100_000) as path:
            parent = path.parent
            raise RuntimeError("inference failed")
    assert not parent.exists()


@pytest.mark.parametrize(
    "data,limit,message",
    [(b"", 100, "leer"), (b"broken", 100, "beschädigt"), (b"12345", 4, "Upload-Limit")],
)
def test_invalid_media_and_upload_size(data, limit, message):
    with pytest.raises(TranscriptionError, match=message):
        with prepared_audio(io.BytesIO(data), lambda _: None, limit):
            pytest.fail("invalid media accepted")


def test_video_with_and_without_audio(tmp_path, wav_bytes):
    wav = tmp_path / "voice.wav"
    wav.write_bytes(wav_bytes)
    for with_audio in (False, True):
        video = tmp_path / f"video-{with_audio}.mp4"
        command = [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=32x32:d=1",
        ]
        if with_audio:
            command += ["-i", str(wav), "-shortest"]
        subprocess.run([*command, "-c:v", "libx264", str(video)], check=True)
        with video.open("rb") as source:
            if with_audio:
                with prepared_audio(source, lambda _: None, 100_000) as path:
                    assert path.is_file()
            else:
                with pytest.raises(TranscriptionError, match="keine lesbare Tonspur"):
                    with prepared_audio(source, lambda _: None, 100_000):
                        pytest.fail("video without audio accepted")


def test_missing_ffmpeg_has_install_hint(monkeypatch):
    monkeypatch.setattr("transcriber.media.shutil.which", lambda _: None)
    with pytest.raises(TranscriptionError, match="brew install ffmpeg"):
        with prepared_audio(io.BytesIO(b"data"), lambda _: None, 100):
            pytest.fail("missing ffmpeg accepted")
