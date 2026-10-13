# Neuronales Netz from Scratch

**4AHITS · INSY** – Aldin Mukic & Lisa Hintermaier

Ein neuronales Netz, **komplett selbst programmiert** mit Python und numpy (ohne KI-Bibliothek). Es lernt per Backpropagation, speichert sich als JSON, TXT und C-Header und läuft danach auch **in C** und **im Browser**. Grundlage ist die Angabe [`4AHITS_NN_from_scratch.pdf`](4AHITS_NN_from_scratch.pdf), Part 1–5. Part 6 (ESP32) entfällt, wir arbeiten rein digital.

## Dokumentation

| Datei | Inhalt |
|---|---|
| **[`dokumentation.html`](dokumentation.html)** | **Abgabe-Doku (Quarto):** Vorgehen, was gut lief und wo wir nachbessern mussten, Ergebnisse, Technik kompakt. Herunterladen und im Browser öffnen. |
| [`DOKUMENTATION.md`](DOKUMENTATION.md) | Ausführliche technische Doku: Theorie mit Formeln, Code, Ergebnisse, Entscheidungen |
| [`PROMPTS.md`](PROMPTS.md) | Alle Prompts an die KI (Claude Code) mit Ergebnis, Zeit und Tokens |
| [`PLAN.md`](PLAN.md) | Der Phasenplan mit Status |
| [`CLAUDE.md`](CLAUDE.md) | Die Regeln, nach denen die KI gearbeitet hat |

## Ergebnisse

| Aufgabe | Ergebnis |
|---|---|
| XOR (2 → 2 → 1) | 4/4 richtig |
| 7-Segment-Decoder (4 → 16 → 7) | 10/10 Ziffern, 70/70 Segmente |
| Speichern und Laden (JSON) | geladenes Netz = Original (Abweichung 0) |
| C (Part 4 + 5) | 10/10, Werte gleich wie in Python |
| Vergleich mit TensorFlow (Part 3) | gleiche Ergebnisse (Unterschied ~10⁻⁹), unser Netz 14–180× schneller |
| TicTacToe (Extra) | 98,7 % optimale Züge |

![7-Segment-Seite mit Netzdiagramm](diagramme/web_7seg_ziffer2.png)

## Starten

Voraussetzung: Python 3 mit numpy, für Part 4/5 ein C-Compiler (z. B. clang).

```bash
python3 neural_network.py      # Part 1 + 2: XOR und 7-Segment trainieren, speichern, laden, testen
cd c && make run               # Part 4 + 5: Forward-Pass in C
python3 gui_7seg.py            # Webseite: 7-Segment mit Netzdiagramm + TicTacToe gegen das Netz
```

Die Webseite gibt es auch fertig erzeugt: [`web/7seg.html`](web/7seg.html) herunterladen und im Browser öffnen.

**Part 3 (TensorFlow)** läuft in einer eigenen Umgebung:
```bash
python3 -m venv .venv-tf
.venv-tf/bin/pip install tensorflow matplotlib
.venv-tf/bin/python tf_vergleich.py
```

## Aufbau

| Datei / Ordner | Inhalt |
|---|---|
| `neural_network.py` | Klassen `Neuron`, `Layer`, `NeuralNetwork`, Training, Export/Import, XOR + 7-Segment (Part 1 + 2) |
| `c/` | Forward-Pass in C: aus der `.txt` (Part 4) bzw. dem `.h`-Header (Part 5) |
| `tf_vergleich.py` | Part 3: dieselben Netze in TensorFlow |
| `tictactoe.py` | Minimax als „Lehrer“ und Training des TicTacToe-Netzes |
| `gui_7seg.py`, `web/` | erzeugt die Webseite und öffnet sie im Browser |
| `export/` | die trainierten Netze (JSON, TXT, C-Header) |
| `diagramme/` | Diagramme, Screenshots und Logs für die Doku |
| `dokumentation.qmd` | Quelltext der Quarto-Doku |
