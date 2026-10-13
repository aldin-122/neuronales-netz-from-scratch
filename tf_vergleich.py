"""
Part 3: Vergleich unseres Netzes mit TensorFlow/Keras.

Starten (in der TensorFlow-Umgebung, dauert einige Minuten):
    .venv-tf/bin/python tf_vergleich.py
    .venv-tf/bin/python tf_vergleich.py 2000     # schneller Test mit nur 2000 Epochen

Für einen fairen Vergleich bekommen beide Implementierungen:
  - dieselbe Architektur, Sigmoid überall, SGD mit Lernrate 0.1, batch_size 1
  - dieselbe Kostenfunktion J = 1/2 * Summe (A - Y)^2 (wie im PDF)
  - dieselben Startgewichte (Keras übernimmt die Zufallsgewichte unseres Netzes)
Wenn unsere Backpropagation stimmt, müssen dann fast identische Ergebnisse herauskommen.
"""
import os
import sys
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # TensorFlow-Infomeldungen ausblenden

import matplotlib
matplotlib.use("Agg")                       # nur Bilder speichern, kein Fenster öffnen
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import tensorflow as tf

from neural_network import (EPOCHS, LEARNING_RATE, NeuralNetwork,
                            X_7SEG, X_XOR, Y_7SEG, Y_XOR)

DIAGRAMM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagramme")


def kosten_wie_pdf(y_true, y_pred):
    """
    J = 1/2 * Summe (A - Y)^2 über alle Ausgänge - genau wie in unserem Netz.
    Keras' eingebautes "mse" wäre der Mittelwert OHNE 1/2, also anders skaliert.
    Dann wären die Gradienten (und damit die effektive Lernrate) nicht vergleichbar.
    """
    return 0.5 * tf.reduce_sum(tf.square(y_pred - y_true), axis=-1)


def keras_modell_wie(net, anzahl_samples):
    """Baut ein Keras-Modell mit derselben Architektur und denselben Startgewichten wie net."""
    # float64 wie numpy in unserem Netz, sonst gäbe es Rundungsunterschiede
    schichten = [tf.keras.Input((net.layer_sizes[0],), dtype="float64")]
    for groesse in net.layer_sizes[1:]:
        schichten.append(tf.keras.layers.Dense(groesse, activation="sigmoid", dtype="float64"))
    model = tf.keras.Sequential(schichten)

    # Gewichte kopieren. Keras speichert sie als Matrix [Eingang, Neuron],
    # wir pro Neuron als Liste -> darum transponieren (.T).
    for keras_layer, unser_layer in zip(model.layers, net.layers):
        kernel = np.array([n.weights for n in unser_layer.neurons]).T
        bias = np.array([n.bias for n in unser_layer.neurons])
        keras_layer.set_weights([kernel, bias])

    # steps_per_execution: Keras rechnet eine ganze Epoche am Stück, statt für jedes
    # Sample einzeln aus Python aufgerufen zu werden. Die Mathematik bleibt gleich
    # (weiterhin ein Update pro Sample), es spart nur Verwaltungsaufwand.
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE),
                  loss=kosten_wie_pdf,
                  steps_per_execution=anzahl_samples)
    return model


def korrekt(ausgaben, Y):
    """Zählt die Samples, bei denen alle Ausgänge (Schwelle 0.5) stimmen."""
    return int(np.sum(np.all((ausgaben >= 0.5) == np.array(Y), axis=1)))


def vergleiche(name, layer_sizes, X, Y, epochen):
    """Trainiert unser Netz und Keras mit gleichen Startbedingungen und gibt die Ergebnisse zurück."""
    print(f"\n{'=' * 70}\n{name} {layer_sizes}, {epochen} Epochen\n{'=' * 70}")

    np.random.seed(1)
    unser_netz = NeuralNetwork(layer_sizes, learning_rate=LEARNING_RATE)
    tf_modell = keras_modell_wie(unser_netz, len(X))   # VOR dem Training kopieren!

    t = time.perf_counter()
    unser_verlauf = unser_netz.train(X, Y, epochs=epochen)
    unsere_zeit = time.perf_counter() - t
    print(f"Unser Netz fertig: {unsere_zeit:.1f} s")

    t = time.perf_counter()
    # shuffle=False: gleiche Sample-Reihenfolge wie in unserer train()-Schleife
    tf_verlauf = tf_modell.fit(np.array(X, float), np.array(Y, float), epochs=epochen,
                               batch_size=1, shuffle=False, verbose=0).history["loss"]
    tf_zeit = time.perf_counter() - t
    print(f"TensorFlow fertig: {tf_zeit:.1f} s")

    unsere_ausgaben = np.array([unser_netz.predict(x) for x in X])
    tf_ausgaben = tf_modell.predict(np.array(X, float), verbose=0)

    # Rohwerte nebeneinander (bei 7 Ausgängen gekürzt auf 2 Nachkommastellen)
    print(f"\n{'Input':<14}{'Soll':<9}{'unser Netz':<37}TensorFlow")
    for x, y, u, t_ in zip(X, Y, unsere_ausgaben, tf_ausgaben):
        soll = "".join(map(str, y))
        print(f"{str(x):<14}{soll:<9}{' '.join(f'{v:.2f}' for v in u):<37}"
              f"{' '.join(f'{v:.2f}' for v in t_)}")
    print(f"Max. Unterschied der Ausgaben: {np.max(np.abs(unsere_ausgaben - tf_ausgaben)):.2e}")

    return {
        "name": f"{name} {layer_sizes}",
        "verlauf": (unser_verlauf, tf_verlauf),
        "zeit": (unsere_zeit, tf_zeit),
        "kosten": (unser_verlauf[-1], tf_verlauf[-1]),
        "korrekt": (korrekt(unsere_ausgaben, Y), korrekt(tf_ausgaben, Y)),
        "anzahl": len(X),
    }


def tabelle_ausgeben(ergebnisse):
    """Druckt die Zusammenfassung aller Vergleiche."""
    print(f"\n{'=' * 70}\nZUSAMMENFASSUNG\n{'=' * 70}")
    print(f"{'Netz':<20}{'Implementierung':<17}{'Zeit':>9}{'End-Kosten':>13}{'Korrekt':>9}")
    for e in ergebnisse:
        for i, impl in enumerate(["unser Netz", "TensorFlow"]):
            print(f"{e['name'] if i == 0 else '':<20}{impl:<17}{e['zeit'][i]:>7.1f} s"
                  f"{e['kosten'][i]:>13.6f}{e['korrekt'][i]:>6}/{e['anzahl']}")


def lernkurven_zeichnen(ergebnisse, pfad):
    """Speichert die Kostenverläufe beider Implementierungen als Bild."""
    fig, achsen = plt.subplots(1, len(ergebnisse), figsize=(5 * len(ergebnisse), 4))
    for ax, e in zip(achsen, ergebnisse):
        unser, tf_ = e["verlauf"]
        ax.plot(unser, label="unser Netz", linewidth=2)
        # gestrichelt, damit man sieht, dass die Kurven übereinander liegen
        ax.plot(tf_, label="TensorFlow", linestyle="--", linewidth=2)
        ax.set_yscale("log")   # log-Skala: sonst sieht man nach dem ersten Abfall nichts mehr
        ax.set_title(e["name"])
        ax.set_xlabel("Epoche")
        ax.xaxis.set_major_locator(MaxNLocator(5))   # max. 5 Zahlen, sonst überlappen sie
        ax.set_ylabel("Kosten J (log)")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(pfad), exist_ok=True)
    fig.savefig(pfad, dpi=120)
    print(f"\nDiagramm gespeichert: {os.path.relpath(pfad)}")


if __name__ == "__main__":
    epochen = int(sys.argv[1]) if len(sys.argv) > 1 else EPOCHS
    print(f"TensorFlow {tf.__version__}")

    ergebnisse = [
        vergleiche("XOR", [2, 2, 1], X_XOR, Y_XOR, epochen),
        vergleiche("7-Seg", [4, 10, 7], X_7SEG, Y_7SEG, epochen),   # wie im PDF (Part 3)
        vergleiche("7-Seg", [4, 16, 7], X_7SEG, Y_7SEG, epochen),   # wie unser Netz (Part 2)
    ]
    tabelle_ausgeben(ergebnisse)
    lernkurven_zeichnen(ergebnisse, os.path.join(DIAGRAMM_DIR, "tf_vergleich_lernkurven.png"))
