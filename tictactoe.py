"""
Extra: TicTacToe - unser neuronales Netz lernt die besten Züge von einem Minimax-Algorithmus.

Idee:
  1. Minimax spielt TicTacToe perfekt. Damit berechnen wir für JEDE mögliche Stellung die besten Züge.
  2. Unser Netz (aus neural_network.py) lernt: Brett -> welche Felder sind gute Züge.
  3. Im Browser (web/7seg.html, Ansicht "TicTacToe") spielt das Netz dann gegen dich.

Brett-Darstellung (immer aus Sicht des Spielers, der gerade am Zug ist):
  +1 = eigener Stein, -1 = Stein des Gegners, 0 = leer
  Felder 0-8 zeilenweise:  0 1 2 / 3 4 5 / 6 7 8
So reicht EIN Netz für beide Seiten - egal ob es mit X oder O spielt.

Starten:  python3 tictactoe.py      (trainiert, misst die Qualität, speichert export/tictactoe.json)
"""
import os
import random
import time
from functools import lru_cache

import numpy as np

from neural_network import EXPORT_DIR, NeuralNetwork

# Alle 8 Gewinnlinien: 3 Zeilen, 3 Spalten, 2 Diagonalen
LINIEN = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]

ARCHITEKTUR = [9, 64, 9]
EPOCHEN = 300
LERNRATE = 0.1
EXPORT_PFAD = os.path.join(EXPORT_DIR, "tictactoe.json")


# ----------------------------------------------------------------------
# Spielregeln
# ----------------------------------------------------------------------
def gewinner(brett):
    """+1 / -1 wenn eine Seite drei in einer Reihe hat, sonst 0."""
    for a, b, c in LINIEN:
        if brett[a] != 0 and brett[a] == brett[b] == brett[c]:
            return brett[a]
    return 0


def freie_felder(brett):
    return [i for i, f in enumerate(brett) if f == 0]


def zug(brett, feld):
    """Setzt einen eigenen Stein (+1) und dreht das Brett um: danach ist der Gegner "am Zug" (+1)."""
    neu = list(brett)
    neu[feld] = 1
    return tuple(-f for f in neu)


# ----------------------------------------------------------------------
# Minimax: perfekter Spieler
# ----------------------------------------------------------------------
@lru_cache(maxsize=None)
def minimax(brett):
    """
    Bewertung für den Spieler am Zug bei perfektem Spiel: > 0 gewinnt, 0 unentschieden, < 0 verliert.
    Der Betrag ist umso größer, je FRÜHER das Spiel endet (1 + Anzahl freier Felder). Dadurch bevorzugt
    Minimax schnelle Siege und späte Niederlagen - es blockt also auch in verlorenen Stellungen noch,
    statt "egal wohin" zu setzen (sonst hätte das Netz ein schlechtes Vorbild).
    lru_cache merkt sich jede Stellung - sonst würde dieselbe Stellung tausendfach neu berechnet.
    """
    felder = freie_felder(brett)
    if gewinner(brett) != 0:      # der Gegner hat mit seinem letzten Zug gewonnen
        return -(1 + len(felder))
    if not felder:
        return 0
    # Nach meinem Zug ist der Gegner am Zug: seine Bewertung ist aus meiner Sicht umgedreht
    return max(-minimax(zug(brett, f)) for f in felder)


def beste_zuege(brett):
    """Alle Felder, die bei perfektem Spiel das bestmögliche Ergebnis erreichen."""
    werte = {f: -minimax(zug(brett, f)) for f in freie_felder(brett)}
    bester = max(werte.values())
    return [f for f, w in werte.items() if w == bester]


def alle_stellungen():
    """Alle Stellungen, die in einem echten Spiel vorkommen können und in denen noch gezogen werden muss."""
    gefunden = set()

    def besuchen(brett):
        if brett in gefunden or gewinner(brett) != 0 or not freie_felder(brett):
            return
        gefunden.add(brett)
        for f in freie_felder(brett):
            besuchen(zug(brett, f))

    besuchen((0,) * 9)
    return sorted(gefunden)


def trainingsdaten():
    """X = Stellungen, Y = 1 für jedes optimale Feld, sonst 0 (mehrere gleich gute Züge sind erlaubt)."""
    X, Y = [], []
    for brett in alle_stellungen():
        ziel = [0] * 9
        for f in beste_zuege(brett):
            ziel[f] = 1
        X.append(list(brett))
        Y.append(ziel)
    return X, Y


# ----------------------------------------------------------------------
# Das Netz als Spieler
# ----------------------------------------------------------------------
def netz_zug(net, brett):
    """Das Netz wählt das freie Feld mit der höchsten Ausgabe (belegte Felder werden ignoriert)."""
    ausgabe = net.predict(list(brett))
    return max(freie_felder(brett), key=lambda f: ausgabe[f])


def spielen(spieler_a, spieler_b):
    """Spielt eine Partie, A beginnt. Rückgabe aus Sicht von A: +1 Sieg, 0 Remis, -1 Niederlage."""
    brett = (0,) * 9
    am_zug, anderer, vorzeichen = spieler_a, spieler_b, 1
    while True:
        brett = zug(brett, am_zug(brett))
        if gewinner(brett) != 0:      # wer gerade gezogen hat, hat gewonnen
            return vorzeichen
        if not freie_felder(brett):
            return 0
        am_zug, anderer, vorzeichen = anderer, am_zug, -vorzeichen


def qualitaet_messen(net, X, Y, spiele=1000):
    """Wie gut spielt das Netz? Optimale Züge in allen Stellungen + Partien gegen Zufall und Minimax."""
    optimal = sum(netz_zug(net, tuple(x)) in [f for f, v in enumerate(y) if v] for x, y in zip(X, Y))
    print(f"Optimaler Zug in {optimal}/{len(X)} Stellungen ({100 * optimal / len(X):.1f} %)")

    netz = lambda b: netz_zug(net, b)
    zufall = lambda b: random.choice(freie_felder(b))
    perfekt = lambda b: beste_zuege(b)[0]

    random.seed(1)
    ergebnisse = {}
    for name, gegner in [("Zufall", zufall), ("Minimax", perfekt)]:
        for netz_beginnt in (True, False):
            n = spiele if name == "Zufall" else 1   # Minimax ohne Zufall: jede Partie wäre gleich
            werte = [spielen(netz, gegner) if netz_beginnt else -spielen(gegner, netz) for _ in range(n)]
            s, u, v = werte.count(1), werte.count(0), werte.count(-1)
            wer = "Netz beginnt" if netz_beginnt else "Gegner beginnt"
            print(f"gegen {name:<8} ({wer:<14}): {s:4d} Siege, {u:4d} Remis, {v:4d} Niederlagen")
            ergebnisse[(name, netz_beginnt)] = (s, u, v)
    return optimal / len(X), ergebnisse


def trainieren():
    """Erzeugt die Trainingsdaten, trainiert das Netz, misst die Qualität und speichert es."""
    X, Y = trainingsdaten()
    print(f"{len(X)} Stellungen als Trainingsdaten, Netz {ARCHITEKTUR}, {EPOCHEN} Epochen, Lernrate {LERNRATE}")
    np.random.seed(1)
    random.seed(1)
    net = NeuralNetwork(ARCHITEKTUR, learning_rate=LERNRATE)

    # Reihenfolge der Stellungen pro Epoche mischen, sonst lernt das Netz immer in derselben Abfolge
    t = time.perf_counter()
    reihenfolge = list(range(len(X)))
    for epoche in range(EPOCHEN):
        random.shuffle(reihenfolge)
        kosten = net.train([X[i] for i in reihenfolge], [Y[i] for i in reihenfolge], epochs=1)[0]
        if epoche % 25 == 0 or epoche == EPOCHEN - 1:
            print(f"Epoche {epoche:4d}  Kosten {kosten:.5f}  ({time.perf_counter() - t:.0f} s)")

    qualitaet_messen(net, X, Y)
    net.dump_json(EXPORT_PFAD)
    print(f"Gespeichert: {os.path.relpath(EXPORT_PFAD)}")
    return net


if __name__ == "__main__":
    trainieren()
