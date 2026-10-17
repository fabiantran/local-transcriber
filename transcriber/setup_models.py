"""Explicit, download-only setup; inference never downloads models."""

from __future__ import annotations

import argparse
import os
import sys

from transcriber.engine import MODEL_DIRS, MODEL_ROOT, SPEAKER_REPO, WHISPER_REPO


def main() -> int:
    parser = argparse.ArgumentParser(description="Lokale Transkriptionsmodelle herunterladen.")
    parser.add_argument(
        "--speakers", action="store_true", help="Auch Sprechererkennung einrichten."
    )
    parser.add_argument("--force", action="store_true", help="Modelldateien erneut herunterladen.")
    args = parser.parse_args()

    # Only this explicit setup command may access the model hub.
    os.environ["HF_HUB_OFFLINE"] = "0"
    from httpx import HTTPError
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

    try:
        models = [("whisper", WHISPER_REPO)]
        if args.speakers:
            models.append(("speakers", SPEAKER_REPO))
        for name, repository in models:
            target = MODEL_ROOT / MODEL_DIRS[name]
            if (target / ".ready").is_file() and not args.force:
                print(f"{name}: bereits lokal eingerichtet.")
                continue
            (target / ".ready").unlink(missing_ok=True)
            print(f"{name}: lade {repository} herunter …")
            snapshot_download(
                repository,
                local_dir=target,
                token=os.environ.get("HF_TOKEN") if name == "speakers" else False,
                ignore_patterns=["*.md", ".gitattributes"],
                force_download=args.force,
            )
            if name == "whisper":
                from mlx_whisper.load_models import load_model

                load_model(str(target))
            else:
                from transcriber.engine import speaker_pipeline

                speaker_pipeline(target)
            (target / ".ready").touch()
            print(f"{name}: erfolgreich eingerichtet.")
    except (HfHubHTTPError, LocalEntryNotFoundError, HTTPError):
        print(
            "Modell-Download nicht möglich. Internetverbindung prüfen. Für Sprechererkennung "
            "zuerst die Bedingungen auf https://huggingface.co/"
            "pyannote/speaker-diarization-community-1 akzeptieren und einen "
            "berechtigten Read-Token über HF_TOKEN oder `hf auth login` bereitstellen.",
            file=sys.stderr,
        )
        return 1
    except (OSError, RuntimeError, ValueError, ImportError) as error:
        print(
            f"Modelleinrichtung fehlgeschlagen ({type(error).__name__}). "
            "Bitte freien Speicher, FFmpeg und die Paketinstallation prüfen.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
