# Plan: Neuronales Netz from Scratch (4AHITS) – ausbauen

## Status
- [x] Phase 0 – Setup
- [x] Phase 1 – Part 1: Export/Import (JSON, TXT, .h)
- [x] Phase 2 – Part 2: XOR + 7-Segment trainieren, laden, testen
- [x] Phase 3 – Part 4 + 5: Forward-Pass in C
- [x] Phase 4 – Part 3: Vergleich mit TensorFlow
- [x] Phase 5 – Extras (7-Seg-Seite mit Netzdiagramm, TicTacToe)
- [x] Phase 6 – Dokumentation (DOKUMENTATION.md + Quarto-Version `dokumentation.qmd` → `dokumentation.html`)

Bonusaufgaben aus dem PDF (ReLU/tanh, Mini-Batch, Momentum, Testsuite, verrauschte Daten): vorerst weggelassen, Entscheidung folgt.

## Kontext
Aufgabe laut `4AHITS_NN_from_scratch.pdf`: ein eigenes neuronales Netz in Python bauen, trainieren, speichern und laden (Part 1+2), mit TensorFlow vergleichen (Part 3) und den Forward-Pass in C nachbauen (Part 4+5). **Part 6 (ESP32/GPIO) fällt weg**, weil wir rein digital arbeiten.
Es gibt schon ein Grundgerüst (`neural_network 1.py`): `Neuron`, `Layer` und `NeuralNetwork` mit Forward-Pass, Backprop (SGD) und Training sowie das XOR-Training (`train_xor()`). Damit sind Part 1 Steps 1–3 erledigt, und die Backprop stimmt mit den PDF-Formeln überein (`delta = (A−Y)·σ'(Z)`, Fehler werden mit den alten Gewichten zurückgegeben).
Danach kommen Extras: eine Tkinter-GUI für den 7-Segment-Decoder, eine Visualisierung der Gewichte und TicTacToe gegen den Menschen.

## Zusammenarbeit (kommt in die CLAUDE.md)
- Wir arbeiten Phase für Phase. **Nach jeder Phase stimmen wir uns ab**, erst dann geht es weiter.
- Wenn etwas unklar ist, frage ich nach, statt einfach etwas anzunehmen.
- Code-Kommentare sind auf Deutsch, in gesunder Menge: jede Klasse und Funktion bekommt einen kurzen Docstring, und bei nicht offensichtlichen Stellen steht das „Warum“ (z.B. der Bezug zur Formel im PDF).
- Ich schreibe jeden Prompt mit einer kurzen Zusammenfassung in `PROMPTS.md`.
- Zeit und Tokens: Genaue Werte zeigt dir der Befehl `/cost` bzw. `/usage` in Claude Code. Ich kann meinen eigenen Verbrauch nicht exakt messen, also trägst du die Werte pro Phase in `PROMPTS.md` ein (ich lege die Spalten dafür an).

## Dateistruktur (Ziel)
```
Neuronales_Netzwerk/
├── CLAUDE.md               # Regeln/Kontext für jede Sitzung
├── neural_network.py       # (umbenannt aus "neural_network 1.py") – Part 1 + 2 komplett
├── tf_vergleich.py         # Part 3 (TensorFlow)
├── c/
│   ├── nn_forward.c        # Part 4: Forward-Pass, lädt Gewichte aus .txt
│   ├── nn_header_main.c    # Part 5: Forward-Pass, initialisiert über .h
│   └── Makefile            # optional, sonst reicht ein clang-Befehl
├── export/                 # erzeugte Dateien: xor.json/.txt/.h, 7seg.json/.txt/.h
├── gui_7seg.py             # Extra: erzeugt web/7seg.html (Anzeige + Netzdiagramm) und öffnet sie
├── web/                    # HTML-Vorlagen + erzeugte Seiten der Extras
├── tictactoe.py            # Extra: Minimax + Training des TicTacToe-Netzes → export/tictactoe.json
├── DOKUMENTATION.md
└── PROMPTS.md
```

## Phase 0 – Setup (klein)
- `neural_network 1.py` in `neural_network.py` umbenennen
- `CLAUDE.md` (Regeln von oben), `PROMPTS.md` (Tabelle: Nr. | Prompt | Ergebnis | Zeit | Tokens) und das Grundgerüst von `DOKUMENTATION.md` anlegen
- **→ Abstimmung**

## Phase 1 – Part 1 fertigstellen (in `neural_network.py`)
Die bestehenden Klassen bleiben, sie bekommen nur neue Methoden in `NeuralNetwork`:
- `dump_json(path)` / `load(path)` (als `@classmethod`): speichert bzw. lädt Layer-Größen, Aktivierungsfunktion (als Name, z.B. `"sigmoid"`), Lernrate sowie Gewichte und Bias pro Neuron. Dafür wird eine kleine Zuordnung `ACTIVATIONS = {"sigmoid": (sigmoid, sigmoid_derivative), ...}` gebraucht, weil sich Funktionen selbst nicht als JSON speichern lassen.
- `dump_txt(path)`: ein lesbares Format, das sich auch mit C einfach über `fscanf` einlesen lässt. Vorschlag: Kopfzeilen mit Anzahl der Layer und Neuronen, danach pro Neuron eine Zeile `bias w1 w2 …`. Kommentarzeilen mit `#` machen es lesbar, der C-Parser überspringt sie.
- `dump_initfile(path, name)`: erzeugt eine `.h` mit `#define`s (Layer-Anzahl, Größen), `static const float`-Arrays für Gewichte und Biases und einer `init_network(Network*)`-Funktion.
- **→ Abstimmung** (gemeinsam die exportierten Dateien anschauen)

## Phase 2 – Part 2 (in `neural_network.py`)
- `train_xor()` erweitern: danach `export/xor.json` (+ `.txt` und `.h`) exportieren
- `test_geladenes_netz(path, X, Y)`: lädt die JSON, vergleicht die Ausgabe mit dem Original-Netz (Toleranz z.B. 1e-9) und mit den Sollwerten, und gibt eine Tabelle aus
- 7-Segment-Datensatz (Tabelle aus dem PDF) als Konstanten `X_7SEG`, `Y_7SEG`, dazu `train_7seg()` mit [4, 16, 7], 20000 Epochen, α = 0.1, und dem Export nach `export/7seg.*`
- Test der geladenen 7-Segment-Datei, inklusive Accuracy (Anteil korrekter Segmente, Schwelle 0.5)
- `if __name__ == "__main__":` führt alles nacheinander aus: XOR, laden und testen, dann 7-Segment, laden und testen
- **→ Abstimmung**

## Phase 3 – Part 4 + 5 (C)
Diese Phase kommt vor TensorFlow, weil sie direkt auf den Exporten aus Phase 1 aufbaut und nichts installiert werden muss.
- Structs `Neuron {float *weights; float bias; float output;}`, `Layer` und `Network`, dazu `forward()` und `predict()`, nur mit Sigmoid
- Part 4: `load_from_txt()` liest `export/7seg.txt` und testet die Ziffern 0–9 gegen die Sollwerte
- Part 5: dieselben Structs, initialisiert über `#include "../export/7seg.h"`
- Kompilieren mit `clang` (ist auf deinem Mac vorhanden)
- **→ Abstimmung**

## Phase 4 – Part 3 (TensorFlow)
- **Hinweis:** Du hast Python 3.14, und TensorFlow unterstützt das wahrscheinlich noch nicht. Die Lösung wäre `brew install python@3.12` plus eine eigene virtuelle Umgebung nur für diese Datei. **Das besprechen wir vor dem Installieren.**
- `tf_vergleich.py`: XOR [2,2,1] und 7-Segment [4,10,7] (wie im PDF, dazu zum Vergleich auch 16) mit Sigmoid, MSE, SGD, lr 0.1, 20000 Epochen und batch_size 1
- Vergleich mit unserem Netz: Ausgaben, Accuracy, Trainingszeit (`time.perf_counter`) und Kostenverlauf
- **→ Abstimmung**

## Phase 5 – Extras (Reihenfolge und Details stimmen wir dann ab)
**Entscheidung:** alle Extras laufen im **Browser**, und zwar auf **einer gemeinsamen Seite** (Python trainiert → Gewichte in HTML → JS-Forward-Pass). Grund: Tkinter-Fenster erscheint bei VS Code im Vollbild nicht im Vordergrund (Details: DOKUMENTATION.md 5.1).
1. ✅ **`gui_7seg.py` (Webseite):** **Beide Eingabearten:** Buttons 0–9 (bevorzugt) UND 4 anklickbare Bit-Buttons (0↔1), die beim Ziffer-Klick automatisch mitgesetzt werden – so sind auch ungelernte Eingaben 10–15 testbar. Dazu auf einem Canvas gezeichnete Segmente, deren Helligkeit dem Ausgabewert (0–1) entspricht, dazu die Rohwerte als Zahlen. Lädt `export/7seg.json` über `NeuralNetwork.load`.
2. ✅ **Gewichte visualisieren:** jetzt **in derselben Seite** wie die 7-Segment-Anzeige: Netzdiagramm 4→16→7, Linien = Signal w·a (ändert sich bei jedem Klick), kein Play/Kostenkurve/XOR. Die frühere eigene Seite `gui_gewichte.py` wurde verworfen (siehe DOKUMENTATION.md 5.1, Entscheidung 2).
3. ✅ **TicTacToe (`tictactoe.py` + zweite Ansicht in `web/7seg.html`):** Minimax (bevorzugt schnelle Siege/späte Niederlagen) erzeugt für alle 4520 Stellungen die besten Züge, unser Netz 9→64→9 lernt sie; Button „TicTacToe spielen →“ / „← Zurück“ (Adresse `#tictactoe`), wählbarer Beginner, Punktestand, abschaltbare Netz-Gedanken (rückblickend).
4. Optional aus dem PDF: ReLU/tanh, Mini-Batch, Momentum, eine Testsuite (`unittest`) und Training mit verrauschten Daten

## Phase 6 – Dokumentation
- `DOKUMENTATION.md`: Theorie kurz erklärt (Forward, Backprop, Kettenregel), Aufbau des Codes, Anleitung zum Ausführen, Ergebnisse (Tabellen, Screenshots der GUI) und der TF-Vergleich
- `PROMPTS.md` vervollständigen, dazu Zeit und Tokens pro Phase

## Verifikation
- `python3 neural_network.py`: XOR 4/4 korrekt, 7-Segment 10/10 Ziffern korrekt, geladenes Netz == Original (Toleranz)
- Die exportierten `.json`, `.txt` und `.h` anschauen und prüfen, ob sie plausibel sind
- `clang c/nn_forward.c -o nn_forward -lm && ./nn_forward` → 10/10 Ziffern, Werte gleich wie in Python; dasselbe für die Header-Variante
- `tf_vergleich.py` gibt die Vergleichstabelle aus
- GUIs von Hand starten und ausprobieren
