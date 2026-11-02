import time
from pathlib import Path

import streamlit as st

from transcriber import jobs, vocabulary
from transcriber.core import TranscriptionError, timestamp
from transcriber.engine import MAX_VOCABULARY_CHARS, clock, whisper_model
from transcriber.logs import LOG_FILE, configure_logging
from transcriber.media import FORMATS, audio_clip

configure_logging()
st.set_page_config(page_title="Lokaler Transkribierer", page_icon="🎙", layout="centered")


def current_job() -> jobs.Job | None:
    try:
        return jobs.load_saved(st.query_params.get("job"))
    except TranscriptionError:
        return None


def invalidate() -> None:
    job = jobs.get(st.query_params.get("job"))
    if job is None or job.state != "running":
        jobs.discard(st.query_params.get("job"))
        st.query_params.pop("job", None)


def open_saved(job_id: str) -> None:
    try:
        loaded = jobs.load_saved(job_id)
    except TranscriptionError:
        st.session_state["library_error"] = True
        return
    if loaded is None:
        st.session_state["library_error"] = True
        return
    st.query_params["job"] = job_id


def store_vocabulary() -> None:
    invalidate()
    try:
        vocabulary.save(st.session_state["vocabulary"])
    except OSError:
        st.session_state["vocabulary_error"] = True


def reset() -> None:
    invalidate()
    st.session_state["upload_version"] = st.session_state.get("upload_version", 0) + 1


def review_words(job: jobs.Job) -> None:
    with st.expander("Unsichere Wörter prüfen & korrigieren", expanded=True):
        st.caption(
            "Niedrige Konfidenz ist ein Prüfhinweis, keine Fehlergarantie. Auch Wörter mit "
            "hohem Wert können falsch sein. Fehlende Werte werden als unbekannt aufgeführt."
        )
        threshold = st.slider(
            "Prüfschwelle (Modell-Konfidenz)", 0.0, 1.0, 0.6, 0.05, key=f"threshold_{job.id}"
        )
        show_all = st.checkbox("Alle Wörter anzeigen", key=f"all_words_{job.id}")
        candidates = [
            (si, wi)
            for si, segment in enumerate(job.segments)
            for wi, word in enumerate(segment.words)
            if show_all or word.probability is None or word.probability < threshold
        ]
        st.caption(
            f"{len(candidates)} Wörter zur Auswahl · "
            f"{sum(key in job.reviewed for key in candidates)} davon bereits geprüft"
        )
        if not candidates:
            st.info("Keine Wörter unter der Prüfschwelle. Bei Bedarf alle Wörter anzeigen.")
            return

        def label(key: tuple[int, int]) -> str:
            si, wi = key
            word = job.segments[si].words[wi]
            confidence = (
                "Konfidenz unbekannt"
                if word.probability is None
                else f"Konfidenz {word.probability:.2f}"
            )
            checked = " · geprüft" if key in job.reviewed else ""
            return f"{timestamp(word.start)} · {word.text.strip()} · {confidence}{checked}"

        selection_key = f"review_word_{job.id}"
        error_key = f"review_error_{job.id}"
        selected = st.selectbox(
            "Wortstelle auswählen",
            candidates,
            format_func=label,
            key=selection_key,
        )
        si, wi = selected
        segment = job.segments[si]
        word = segment.words[wi]
        st.caption(
            f"Wortstelle: {timestamp(word.start)} – {timestamp(word.end)}"
            + (f" · {segment.speaker}" if segment.speaker else "")
        )
        context = [
            f"⟦{item.text.strip()}⟧" if index == wi else item.text.strip()
            for index, item in enumerate(segment.words)
            if max(0, wi - 6) <= index <= wi + 6
        ]
        st.text(" ".join(context))
        if job.audio is not None:
            try:
                clip = audio_clip(job.audio, word.start, word.end)
            except TranscriptionError as error:
                st.error(str(error))
            else:
                st.audio(clip, format="audio/wav")
                st.caption(
                    "Audioausschnitt mit bis zu zwei Sekunden Kontext vor und nach dem Wort."
                )
        else:
            st.warning("Für diesen Auftrag ist keine Tonspur zum Nachhören verfügbar.")
        correction_key = f"correction_{job.id}_{si}_{wi}"

        def apply_correction() -> None:
            try:
                job.correct_word(si, wi, st.session_state[correction_key])
            except TranscriptionError as error:
                st.session_state[error_key] = str(error)
            else:
                st.session_state.pop(error_key, None)
                index = candidates.index(selected)
                following = candidates[index + 1 :] + candidates[:index]
                next_word = next((key for key in following if key not in job.reviewed), None)
                if next_word is not None:
                    st.session_state[selection_key] = next_word

        with st.form(f"review_form_{job.id}", enter_to_submit=True):
            st.text_input("Korrektur", value=word.text.strip(), key=correction_key)
            st.form_submit_button(
                "Korrektur übernehmen / als geprüft markieren",
                key=f"apply_{job.id}",
                on_click=apply_correction,
            )
        if error_key in st.session_state:
            st.error(st.session_state[error_key])
        if all(key in job.reviewed for key in candidates):
            st.success("Alle Wörter der aktuellen Auswahl sind geprüft.")
        st.caption(
            "Klick oder Enter übernimmt die Korrektur und öffnet die nächste ungeprüfte "
            "Wortstelle der aktuellen Auswahl. "
            "Korrekturen werden sofort in transkripte/ gespeichert (TXT und JSON) und "
            "überstehen ein Neuladen sowie einen App-Neustart. Das Modell wird dadurch nicht "
            "trainiert."
        )


def render_library(saved_jobs: list[jobs.Job]) -> None:
    st.caption(
        "Fertige Transkripte bleiben im Ordner transkripte/ erhalten, auch nach dem Schließen "
        "der App. Öffne ein Transkript, um die Wortprüfung dort fortzusetzen."
    )
    if not saved_jobs:
        st.info("Noch keine gespeicherten Transkripte. Fertige Aufträge erscheinen hier.")
        return
    for saved in saved_jobs:
        with st.container(border=True):
            st.markdown(f"**{saved.download_name}**")
            language = "Deutsch" if saved.language == "de" else "Englisch"
            speakers = (
                "mit Sprechererkennung" if saved.detect_speakers else "ohne Sprechererkennung"
            )
            started = time.strftime("%d.%m.%Y %H:%M", time.localtime(saved.started))
            st.caption(
                f"{started} · {language} · {speakers} · {len(saved.reviewed)} Wortstellen geprüft"
            )
            st.button(
                "Öffnen",
                key=f"open_saved_{saved.id}",
                on_click=open_saved,
                args=(saved.id,),
            )


@st.fragment(run_every=2)
def show_progress(job_id: str) -> None:
    job = jobs.get(job_id)
    if job is None or job.state != "running":
        st.rerun()
        return
    with st.container(border=True):
        st.markdown(f"**⏳ {job.phase}**")
        if job.fraction is not None:
            st.progress(job.fraction, text=f"{int(job.fraction * 100)} %")
        details = [
            f"Gestartet um {time.strftime('%H:%M', time.localtime(job.started))}",
            f"läuft seit {clock(job.elapsed)}",
        ]
        if job.remaining is not None:
            details.append(f"dieser Schritt noch ca. {clock(job.remaining)}")
        st.caption(" · ".join(details))
    st.caption(
        "Du kannst diese Seite neu laden oder später wieder öffnen, der Auftrag läuft weiter. "
        f"Protokoll: Terminal „Transkribierer starten“ oder `{LOG_FILE}`."
    )


st.title("Audio & Video transkribieren")
st.write("Deine Aufnahme wird auf diesem Mac verarbeitet. Kein Cloud-Upload.")
st.caption("Deutsch oder Englisch · optionale Sprechererkennung · TXT-Download")

with st.expander("Einmalige Einrichtung & Datenschutz"):
    st.write(
        "Die Modelle müssen einmalig heruntergeladen werden. Danach arbeitet die App offline. "
        "Für Sprechererkennung ist zusätzlich eine kostenlose Hugging-Face-Freischaltung nötig."
    )
    st.code(
        "uv run python -m transcriber.setup_models\n"
        "uv run python -m transcriber.setup_models --speakers",
        language="bash",
    )
    st.write(
        "Dateien werden temporär aufbereitet und anschließend gelöscht. Fertige Aufträge werden "
        "automatisch im Git-ignorierten Ordner `transkripte/` gespeichert: als TXT, als "
        "Tonspur (WAV) zum Nachhören und als JSON mit Wortprüfung. Diese Dateien sind nicht "
        "verschlüsselt und bleiben bis zum manuellen Löschen erhalten, auch nach einem "
        "App-Neustart. Im Arbeitsspeicher bleiben Aufträge höchstens 12 Stunden. Das Protokoll "
        "enthält keine Dateinamen und keinen Text. Die Fachbegriff-Liste wird lokal in "
        "`fachbegriffe.txt` gespeichert. Details und Modellfreischaltung stehen in der README."
    )

if st.session_state.pop("library_error", False):
    st.error(
        "Das Transkript konnte nicht geöffnet werden. Die gespeicherten Dateien sind "
        "unvollständig oder beschädigt."
    )
with st.sidebar:
    saved_jobs = jobs.list_saved()
    st.subheader(f"Gespeicherte Transkripte ({len(saved_jobs)})")
    render_library(saved_jobs)

limit_mb = st.get_option("server.maxUploadSize")
upload = st.file_uploader(
    "Audio- oder Videodatei",
    type=list(FORMATS),
    key=f"upload_{st.session_state.get('upload_version', 0)}",
    on_change=invalidate,
    help=f"Bis {limit_mb} MB. Bei großen Dateien wird viel Arbeitsspeicher benötigt.",
)
language_label = st.selectbox("Gesprochene Sprache", ["Deutsch", "Englisch"], on_change=invalidate)
detect_speakers = st.toggle("Sprecher erkennen", value=False, on_change=invalidate)
if detect_speakers:
    st.caption(
        "Erkennt Sprecher 1, Sprecher 2 usw., keine Namen. Überlappende Stimmen oder kurze "
        "Beiträge können falsch zugeordnet werden. Die zusätzliche Analyse läuft auf der CPU."
    )

job = current_job()
if job is None and "job" in st.query_params:
    st.query_params.pop("job")
    st.warning(
        "Der vorherige Auftrag ist nicht mehr verfügbar und konnte nicht aus transkripte/ "
        "geladen werden. Fertige Transkripte findest du unter „Gespeicherte Transkripte“. "
        "Aufträge aus älteren Versionen haben keine Wortprüfungsdaten und lassen sich nicht "
        "öffnen. Bitte dann die Datei erneut transkribieren."
    )
running = job is not None and job.state == "running"

if "vocabulary" not in st.session_state:
    st.session_state["vocabulary"] = vocabulary.load()[:MAX_VOCABULARY_CHARS]
terms = st.text_area(
    "Fachbegriffe & Namen (optional)",
    key="vocabulary",
    max_chars=MAX_VOCABULARY_CHARS,
    placeholder="z. B. Müller, Kubernetes, Q3-Roadmap, Frau Dr. Yilmaz",
    help=(
        "Wörter, die in der Aufnahme vorkommen und oft falsch erkannt werden, getrennt durch "
        "Komma oder Zeilenumbruch. Das Modell bevorzugt dann diese Schreibweisen. Nur echte "
        "Begriffe eintragen: Bei Stille kann das Modell sonst einzelne Begriffe erfinden. "
        f"Die Liste wird automatisch lokal in `{vocabulary.VOCABULARY_FILE.name}` gespeichert."
    ),
    on_change=store_vocabulary,
)
if st.session_state.pop("vocabulary_error", False):
    st.warning(f"Die Liste konnte nicht in `{vocabulary.VOCABULARY_FILE}` gespeichert werden.")

model = whisper_model()
if model is None:
    st.caption("Sprachmodell: nicht eingerichtet")
elif model[1].startswith("Whisper small"):
    st.warning(
        f"Es wird noch das kleine Modell verwendet ({model[1]}). Für bessere Genauigkeit "
        "einmalig `uv run python -m transcriber.setup_models` ausführen und die App neu starten."
    )
else:
    st.caption(f"Sprachmodell: {model[1]}")

if st.button("Transkribieren", type="primary", disabled=upload is None or running):
    invalidate()
    if upload is not None:
        job = jobs.start(
            upload,
            "de" if language_label == "Deutsch" else "en",
            detect_speakers,
            f"{Path(upload.name).stem}-transkript.txt",
            max_bytes=limit_mb * 1024 * 1024,
            vocabulary=terms,
        )
        st.query_params["job"] = job.id
        st.rerun()

if job is not None and job.state == "running":
    show_progress(job.id)
elif job is not None and job.state == "error":
    st.error(job.error)
    st.button("Fehlermeldung schließen", on_click=reset)
elif job is not None:
    st.subheader("Dein Transkript")
    settings = "Deutsch" if job.language == "de" else "Englisch"
    settings += " · mit Sprechererkennung" if job.detect_speakers else " · ohne Sprechererkennung"
    st.caption(f"Fertig nach {clock(job.elapsed)} · {settings}")
    if job.save_error:
        st.warning(job.save_error)
    elif job.saved_path is not None:
        st.caption(f"Automatisch gespeichert: {job.saved_path}")
    if job.result:
        st.text(job.result)
        if any(segment.words for segment in job.segments):
            review_words(job)
        st.download_button(
            "Als TXT herunterladen",
            data=job.result.encode("utf-8"),
            file_name=job.download_name,
            mime="text/plain; charset=utf-8",
            on_click="ignore",
        )
    else:
        st.info("Keine Sprache erkannt. Bitte Aufnahme und ausgewählte Sprache prüfen.")
    st.button("Datei & Ergebnis verwerfen", on_click=reset)
