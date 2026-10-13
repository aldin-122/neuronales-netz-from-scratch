"""
Extras im Browser: 7-Segment-Anzeige mit Netzdiagramm + TicTacToe gegen das Netz.

Ablauf:
  1. Die trainierten Netze holen: export/7seg.json und export/tictactoe.json
     (fehlt eine Datei, wird das Netz automatisch trainiert - TicTacToe dauert ca. 3 Minuten)
  2. Netze und Sollwerte in die Vorlage web/vorlage_7seg.html einsetzen -> web/7seg.html
  3. web/7seg.html im Standard-Browser öffnen

Gerechnet wird dann im Browser mit JavaScript (nur Forward-Pass, wie in C bei Part 4/5).
Warum Browser statt Tkinter: siehe DOKUMENTATION.md, Abschnitt 5.1.

Starten:  python3 gui_7seg.py
"""
import json
import os
import webbrowser
from pathlib import Path

import tictactoe
from neural_network import EXPORT_DIR, Y_7SEG, train_7seg

WEB_DIR = Path(__file__).resolve().parent / "web"
VORLAGE = WEB_DIR / "vorlage_7seg.html"
ZIEL = WEB_DIR / "7seg.html"


def netz_datei():
    """Pfad zu export/7seg.json. Fehlt die Datei, wird das Netz einmal trainiert und exportiert."""
    pfad = Path(EXPORT_DIR) / "7seg.json"
    if not pfad.exists():
        print("export/7seg.json fehlt - das Netz wird jetzt trainiert (ca. 8 Sekunden) ...")
        train_7seg()   # speichert das Netz auch gleich nach export/
    return pfad


def ttt_datei():
    """Pfad zu export/tictactoe.json. Fehlt die Datei, wird das TicTacToe-Netz einmal trainiert."""
    pfad = Path(tictactoe.EXPORT_PFAD)
    if not pfad.exists():
        print("export/tictactoe.json fehlt - das TicTacToe-Netz wird jetzt trainiert (ca. 3 Minuten) ...")
        tictactoe.trainieren()
    return pfad


def html_erzeugen():
    """Setzt beide Netze und die Sollwerte in die Vorlage ein und speichert das Ergebnis als web/7seg.html."""
    html = VORLAGE.read_text(encoding="utf-8")
    # Die JSON-Datei ist schon gültiges JavaScript, sie kann direkt eingesetzt werden
    html = html.replace("/*NETZ_JSON*/null", netz_datei().read_text(encoding="utf-8"))
    html = html.replace("/*SOLL_JSON*/null", json.dumps(Y_7SEG))
    html = html.replace("/*TTT_JSON*/null", ttt_datei().read_text(encoding="utf-8"))
    ZIEL.write_text(html, encoding="utf-8")
    return ZIEL


def main():
    pfad = html_erzeugen()
    print(f"Webseite erzeugt: {os.path.relpath(pfad)}")
    # as_uri() macht daraus file:///..., das versteht jeder Browser
    webbrowser.open(pfad.as_uri())
    print("Im Browser geöffnet.")


if __name__ == "__main__":
    main()
