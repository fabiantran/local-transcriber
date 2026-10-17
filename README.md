# Lokaler Transkribierer

Audio und Video auf einem Mac mit Apple Silicon transkribieren, ohne die Aufnahme
an einen Cloud-Dienst zu senden. Die deutsche Browser-Oberfläche bietet:

- eine Datei pro Auftrag: WAV, MP3, M4A, FLAC, OGG, MP4, MOV, MKV oder WebM;
- Deutsch oder Englisch als gesprochene Sprache (keine Übersetzung);
- optionale automatische Sprechertrennung mit `Sprecher 1`, `Sprecher 2` usw.;
- ein Transkript mit Zeitstempeln und UTF-8-TXT-Download.

Sprechererkennung identifiziert **keine Namen**. Bei Dialekt, Geräuschen,
überlappenden Stimmen oder kurzen Beiträgen können Text und Sprecherzuordnung
ungenau sein. Unzuordenbare Wörter erhalten `Sprecher unbekannt`.

## Voraussetzungen

- Mac mit Apple Silicon, macOS mit Unterstützung für die installierten MLX-Wheels;
- Python 3.11–3.13; die lokale Einrichtung verwendet Python 3.12;
- [uv](https://docs.astral.sh/uv/getting-started/installation/);
- FFmpeg mit geteilten Bibliotheken, zum Beispiel über Homebrew;
- mehrere GB freier Speicher für Pakete und Modelle sowie ausreichend RAM.

```bash
brew install ffmpeg
uv sync --python 3.12
uv run python -m transcriber.setup_models
uv run streamlit run app.py
```

Alle Befehle im Projektordner ausführen. Dann
**http://127.0.0.1:8501** öffnen. Der Server ist nur lokal erreichbar.
Beenden mit `Ctrl+C`. `uv.lock` hält die überprüften Paketversionen fest.
Installation und der explizite Modell-Download benötigen Internet.
In VS Code ist außerdem die Aufgabe **Transkribierer starten** verfügbar.

Die Transkription verwendet das mehrsprachige
[MLX-Whisper-large-v3-turbo-Modell](https://huggingface.co/mlx-community/whisper-large-v3-turbo)
(ca. 1,6 GB, in `.models/whisper-large-v3-turbo/`). Ist es noch nicht
eingerichtet, nutzt die App ein früher installiertes, deutlich ungenaueres
Whisper-small-Modell aus `.models/whisper/` und zeigt einen Hinweis an. Dieses
kann gelöscht werden, sobald das neue Modell läuft. MLX nutzt Apple Silicon. Die zusätzliche Sprecheranalyse läuft bewusst auf der
CPU und kann länger dauern. Der erste Lauf lädt die Modelle in den Arbeitsspeicher.

## Sprechererkennung einmalig freischalten

1. Ein kostenloses [Hugging-Face-Konto](https://huggingface.co/join) erstellen.
2. Die Bedingungen von
   [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
   lesen und akzeptieren.
3. Einen [Read-Token](https://huggingface.co/settings/tokens) mit Zugriff auf das
   freigeschaltete Modell erstellen.
4. Im Terminal anmelden und das Modell herunterladen:

   ```bash
   uv run hf auth login
   uv run python -m transcriber.setup_models --speakers
   ```

Alternativ unterstützt die Einrichtung einen bereits gesetzten `HF_TOKEN`.
**Tokens nicht in Code, Chat, Screenshots oder Versionsverwaltung einfügen.**
`hf auth login` kann den Token lokal im Hugging-Face-Profil speichern; für die
spätere Transkription wird er nicht benötigt.

Die App verwendet ausschließlich das lokale Community-1-Modell, keinen
gehosteten pyannote-Dienst. Ohne Freischaltung ist Transkription ohne
Sprechererkennung weiterhin möglich. Ein fehlendes Sprechermodell führt bei
aktivierter Option zu einem sichtbaren Fehler, nicht zu einem scheinbar
vollständigen Ergebnis ohne Sprecherlabels.

## Bedienung

1. Audio oder Video hochladen.
2. Die tatsächlich gesprochene Sprache auswählen.
3. Falls gewünscht `Sprecher erkennen` aktivieren.
   Optional unter `Fachbegriffe & Namen` Eigennamen, Firmen- oder Fachbegriffe
   eintragen (durch Komma oder Zeilenumbruch getrennt). Das Modell bevorzugt
   dann diese Schreibweisen in der ganzen Aufnahme. Nur Begriffe eintragen, die
   wirklich vorkommen; bei Stille kann das Modell sonst Begriffe erfinden.
4. `Transkribieren` drücken. Die Seite zeigt den aktuellen Schritt, Prozent,
   Startzeit, Laufzeit und eine grobe Restzeit des aktuellen Schritts.
5. Transkript lesen und `Als TXT herunterladen` wählen.

Die Verarbeitung läuft im Hintergrund der App. Die Adresse enthält danach
`?job=…`: Du kannst die Seite neu laden, den Tab schließen und dieselbe Adresse
später wieder öffnen, solange die App selbst weiterläuft. Ein Neustart der App
bricht laufende Aufträge ab und verwirft Ergebnisse.

### Fortschritt und Logs

Jeder Auftrag protokolliert Schritte, Dauer, Audiolänge, Fortschritt in
10-%-Schritten und Fehlertypen, aber keine Dateinamen und keinen
Transkripttext. Das Protokoll erscheint im Terminal der App (VS-Code-Task
„Transkribierer starten“) und in `logs/transcriber.log`:

```bash
tail -f logs/transcriber.log
```

Die Sprechererkennung läuft auf der CPU und kann bei langen Aufnahmen deutlich
länger dauern als die Transkription selbst.

Es gibt keinen Editor, Medienplayer, Stapelbetrieb oder Untertitel-Export.
Ein Download startet die Analyse nicht erneut. Änderungen an Datei, Sprache
oder Sprecheroption verwerfen das vorherige Ergebnis. Mit
`Datei & Ergebnis verwerfen` werden Upload und Ergebnis zurückgesetzt.

## Datenschutz und Speicher

- Die Verarbeitung erfolgt lokal. Es gibt keinen Cloud-Upload und keine
  Ergebnisbibliothek.
- Nur der explizite Einrichtungsbefehl lädt Modelle herunter. Während der
  App-Verarbeitung sind Hugging-Face-Offline-Modus und die Abschaltung von
  Streamlit-, Hugging-Face- und pyannote-Nutzungsmetriken aktiv.
- Modelle bleiben in `.models/` im Projektordner. Dieser Ordner wird nicht
  versioniert und kann mehrere GB umfassen.
- Upload und Ergebnis liegen nur im Arbeitsspeicher der laufenden App, damit ein
  Neuladen der Seite sie nicht verliert. Ergebnisse verschwinden mit
  `Datei & Ergebnis verwerfen`, beim App-Neustart oder spätestens nach 12 Stunden
  (beim Start eines neuen Auftrags). Wer die `?job=…`-Adresse kennt, sieht das
  Ergebnis; die App ist nur über localhost erreichbar.
- Zur Medienaufbereitung werden die Originaldatei und eine Mono-WAV-Datei im
  temporären Systemordner abgelegt und nach Verarbeitung auch bei üblichen
  Fehlern gelöscht. Prozessabsturz oder hartes Beenden können Reste in
  `local-transcriber-*`-Unterordnern hinterlassen. Nur solche konkret geprüften,
  nicht mehr verwendeten Ordner bei Bedarf manuell entfernen.
- Downloads bleiben als bewusst gespeicherte TXT-Dateien auf deinem Rechner.
- `logs/transcriber.log` (max. 4 × 1 MB, nicht versioniert) enthält nur
  Verarbeitungsmetadaten. Es werden keine Dateinamen, Transkripttexte oder
  Tokens absichtlich protokolliert.

Die lokale App ist kein Mehrbenutzer- oder öffentlich erreichbarer Dienst.
Die Deaktivierung von Bibliotheks-Telemetrie ist keine Betriebssystem-Firewall.

## Große Dateien

In [`.streamlit/config.toml`](.streamlit/config.toml) ist `server.maxUploadSize`
auf **2048 MB** gesetzt. Dort lässt sich das Limit ändern; anschließend die App
neu starten. Auch das Backend prüft das konfigurierte Limit.

Streamlit hält den Upload im RAM. Zusätzlich brauchen die extrahierte Tonspur,
das Sprachmodell und gegebenenfalls die Sprecheranalyse Speicher. Das
Upload-Limit garantiert daher nicht, dass jede so große Datei verarbeitet
werden kann. Bei langen Aufnahmen oder wenig RAM kürzere Dateien verwenden.
Die App arbeitet jeweils einen Inferenzauftrag ab; weitere warten. Prozent und
Restzeit sind Schätzungen je Schritt.

## Fehlerbehebung

- **FFmpeg fehlt:** `brew install ffmpeg`; anschließend Terminal/App neu starten.
- **Lokales Modell fehlt:** den in der Fehlermeldung angezeigten
  `transcriber.setup_models`-Befehl ausführen.
- **Modell-Download verweigert:** Konto, Modellbedingungen und Token-Berechtigung
  prüfen; die App lädt Modelle nicht automatisch während der Verarbeitung.
- **Keine Tonspur:** Video enthält möglicherweise nur Bild; andere Datei wählen.
- **Beschädigte Datei:** Lesbarkeit in einem lokalen Medienprogramm prüfen.
- **Keine Sprache erkannt:** Sprache und Aufnahme prüfen; Stille ergibt kein
  herunterladbares Transkript.
- **Viele falsch erkannte Wörter:** prüfen, dass die App „Whisper
  large-v3-turbo“ als Sprachmodell anzeigt; richtige Sprache wählen;
  Fachbegriffe & Namen eintragen. Hall, Störgeräusche, Dialekt und
  überlappende Stimmen verschlechtern das Ergebnis weiterhin.
- **Arbeitsspeicher/Modellfehler:** andere speicherintensive Programme schließen
  und kleinere Dateien verwenden. Bei beschädigten Modelldateien die
  Einrichtung mit `--force` wiederholen, bei Bedarf zusätzlich mit `--speakers`.

## Entwicklung und Tests

```bash
uv sync --python 3.12
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Die Tests benötigen für die Medien-Integration FFmpeg, aber weder Modelldownloads
noch einen Hugging-Face-Token. Sie prüfen unter anderem Sprecherwechsel,
Zeitstempel, Fehler, Bereinigung, Video mit/ohne Ton, Spracheinstellungen,
Ergebnisinvalidierung und Download ohne erneute Inferenz.
Diese deterministischen Tests ersetzen keine Qualitätsprüfung mit echten
Aufnahmen und freigeschaltetem Sprechermodell.

Lokal geprüft wurden zusätzlich echte Modellläufe mit synthetischer deutscher
und englischer Sprache, Video mit Ton, Stille und einem Dialog mit zwei
abwechselnden Stimmen. Dabei waren Python-Netzwerkverbindungen blockiert.
Die Zwei-Sprecher-Aufnahme wurde auch über die Browser-Oberfläche verarbeitet.
Das bestätigt die Integration, garantiert aber keine fehlerfreien Transkripte
bei beliebigen natürlichen Aufnahmen.

## Lizenz

Der Code steht unter der [MIT-Lizenz](LICENSE). Die Modelle (Whisper,
pyannote) haben eigene Lizenzen und Nutzungsbedingungen und sind nicht Teil
dieses Repositorys.
