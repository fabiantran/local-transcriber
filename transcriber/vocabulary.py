import os
import tempfile
from pathlib import Path

VOCABULARY_FILE = Path(__file__).resolve().parent.parent / "fachbegriffe.txt"


def load() -> str:
    try:
        return VOCABULARY_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


def save(text: str) -> None:
    text = text.strip()
    if not text:
        VOCABULARY_FILE.unlink(missing_ok=True)
        return
    # Write to a temp file and rename so a crash never leaves a half-written list.
    fd, tmp = tempfile.mkstemp(dir=VOCABULARY_FILE.parent, prefix=".fachbegriffe-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        os.replace(tmp, VOCABULARY_FILE)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
