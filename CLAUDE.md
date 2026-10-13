# Projekt: Neuronales Netz from Scratch (4AHITS, INSY)

Aufgabe: siehe `4AHITS_NN_from_scratch.pdf`. Umgesetzt werden Part 1–5, **Part 6 (ESP32/GPIO) entfällt**, wir arbeiten rein digital.
Danach folgen Extras: 7-Segment-Anzeige, Visualisierung der Gewichte und TicTacToe – **alle auf EINER Webseite im Browser** (nicht Tkinter: Fenster erscheint bei VS Code im Vollbild nicht vorne). Python trainiert, setzt die Gewichte in eine HTML-Vorlage (`web/vorlage_*.html`) ein und öffnet sie; im Browser läuft nur der Forward-Pass in JavaScript.
Der Plan mit allen Phasen und dem aktuellen Status steht in `PLAN.md`.

## Regeln für die Zusammenarbeit
- **Erst planen, dann programmieren.** Wir arbeiten Phase für Phase, und nach jeder Phase wird abgestimmt, bevor es weitergeht.
- **Bei Unklarheiten nachfragen** und gemeinsam eine Lösung finden, statt einfach etwas anzunehmen.
- **Nichts installieren** (z.B. TensorFlow oder Python 3.12), ohne vorher zu fragen.
- Nach jedem Prompt einen Eintrag in `PROMPTS.md` ergänzen.
- **Alles von vorne bis hinten testen** (auch Fehlerfälle, z.B. fehlende Dateien) und gefundene Probleme gleich beheben, bevor etwas als fertig gemeldet wird. Programme sollen ohne manuelle Vorbereitung laufen.

## Code-Stil
- Sprache: Python 3 + numpy; C für Part 4/5 (Compiler: clang); HTML/CSS/JavaScript für die Extras
- TensorFlow nur in `.venv-tf/` (Python 3.14, TF 2.22.0-rc0): `.venv-tf/bin/python tf_vergleich.py`
- Webseiten testen: headless Chrome (`--dump-dom` mit `?test`, `--screenshot`); Chrome beendet sich danach nicht selbst → mit Zeitlimit starten und den Prozess beenden
- **Kommentare auf Deutsch, in gesunder Mitte:** jede Klasse und Funktion bekommt einen kurzen Docstring, und bei nicht offensichtlichen Stellen wird das *Warum* erklärt (gerne mit Bezug zur Formel im PDF). Keine Kommentare, die nur den Code wiederholen.
- Der Code soll für einen Schüler nachvollziehbar sein: lieber klar und einfach als clever.

## Dateien
- `neural_network.py`: Hauptdatei mit Part 1 + 2 (Klassen, Export/Import, XOR, 7-Segment)
- `export/`: erzeugte `.json`-, `.txt`- und `.h`-Dateien
- `c/`: Part 4 + 5 (Forward-Pass in C)
- `tf_vergleich.py`: Part 3
- `gui_7seg.py`: erzeugt `web/7seg.html` aus `web/vorlage_7seg.html` und öffnet sie. Ansicht 1: 7-Segment-Anzeige + Netzdiagramm (Signal w·a); Ansicht 2 (`#tictactoe`, per Button): TicTacToe gegen das Netz
- `tictactoe.py`: Minimax + Training des TicTacToe-Netzes (dauert ca. 3 min) → `export/tictactoe.json`
- `diagramme/`: Diagramme und Screenshots für die Doku
- `DOKUMENTATION.md`, `PROMPTS.md`: Dokumentation
- `dokumentation.qmd` → `dokumentation.html`: Abgabe-Doku in Quarto (Rückblick + Technik kompakt). Erzeugen: `~/Applications/quarto/bin/quarto render dokumentation.qmd` (Quarto ohne Admin-Rechte in `~/Applications/quarto`)
