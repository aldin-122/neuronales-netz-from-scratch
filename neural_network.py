import json
import os

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1.0 - s)


# Zuordnung Name -> (Funktion, Ableitung).
# Wird für den JSON-Export gebraucht: eine Python-Funktion kann man nicht in eine
# Datei schreiben, nur ihren Namen. Beim Laden wird der Name hier nachgeschlagen.
ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_derivative),
}


def _ordner_anlegen(path):
    """Legt den Ordner für eine Ausgabedatei an, falls er noch nicht existiert."""
    ordner = os.path.dirname(path)
    if ordner:
        os.makedirs(ordner, exist_ok=True)


class Neuron:
    def __init__(self, num_inputs, activation=sigmoid, activation_derivative=sigmoid_derivative):
        self.num_inputs = num_inputs
        self.weights = np.random.randn(num_inputs) * 0.1
        self.bias = 0.0
        self.activation = activation
        self.activation_derivative = activation_derivative
        # Werte aus dem Forward-Pass cachen, brauchen wir für backward
        self._last_inputs = None
        self._last_raw = None
        self._last_output = None

    def activate(self, inputs):
        inputs = np.array(inputs)
        self._last_inputs = inputs
        self._last_raw = np.dot(inputs, self.weights) + self.bias
        self._last_output = self.activation(self._last_raw)
        return self._last_output

    def update(self, delta, learning_rate):
        self.weights -= learning_rate * delta * self._last_inputs
        self.bias -= learning_rate * delta


class Layer:
    def __init__(self, num_neurons, num_inputs_per_neuron,
                 activation=sigmoid, activation_derivative=sigmoid_derivative):
        self.neurons = []
        for _ in range(num_neurons):
            neuron = Neuron(num_inputs_per_neuron, activation, activation_derivative)
            self.neurons.append(neuron)

    def forward(self, inputs):
        return np.array([neuron.activate(inputs) for neuron in self.neurons])

    def backward(self, errors, learning_rate):
        # 1) Deltas berechnen: delta = dJ/dA * sigma'(z)
        deltas = []
        for neuron, error in zip(self.neurons, errors):
            delta = error * neuron.activation_derivative(neuron._last_raw)
            deltas.append(delta)

        # 2) Fehler für die vorherige Schicht: gewichtete Summe der Deltas.
        #    WICHTIG: das muss mit den ALTEN Gewichten passieren, also vor dem Update,
        #    sonst stimmt die Kettenregel nicht mehr.
        num_inputs = len(self.neurons[0].weights)
        prev_errors = np.zeros(num_inputs)
        for neuron, delta in zip(self.neurons, deltas):
            prev_errors += delta * neuron.weights

        # 3) Erst jetzt die Gewichte und Biases aktualisieren
        for neuron, delta in zip(self.neurons, deltas):
            neuron.update(delta, learning_rate)

        return prev_errors


class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.1, activation="sigmoid"):
        """
        layer_sizes: z.B. [2, 4, 1] -> 2 Inputs, ein Hidden-Layer mit 4 Neuronen, 1 Output
        activation:  Name der Aktivierungsfunktion (Schlüssel in ACTIVATIONS)
        """
        # Architektur merken, damit wir sie später exportieren können
        self.layer_sizes = list(layer_sizes)
        self.learning_rate = learning_rate
        self.activation_name = activation
        activation_fn, activation_derivative = ACTIVATIONS[activation]

        self.layers = []
        for i in range(1, len(layer_sizes)):
            self.layers.append(
                Layer(layer_sizes[i], layer_sizes[i - 1], activation_fn, activation_derivative)
            )

    def forward(self, inputs):
        output = np.array(inputs, dtype=float)
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def predict(self, inputs):
        return self.forward(inputs)

    def backward(self, output, target, learning_rate):
        """
        Ein Backward-Pass für ein einzelnes Sample.
        Startfehler ist die Ableitung der Kostenfunktion J = 1/2 * (A - Y)^2,
        also dJ/dA = A - Y.
        """
        errors = output - np.array(target, dtype=float)
        for layer in reversed(self.layers):
            errors = layer.backward(errors, learning_rate)

    def train(self, inputs, targets, epochs=20000, learning_rate=None, verbose_every=None):
        """
        Trainiert das Netz mit Stochastic Gradient Descent:
        pro Epoche wird jedes Sample einzeln vorwärts und rückwärts propagiert.

        inputs:  Liste von Eingabevektoren, z.B. [[0,0],[0,1],[1,0],[1,1]]
        targets: Liste von Zielvektoren,   z.B. [[0],[1],[1],[0]]
        Rückgabe: Liste der mittleren Kosten pro Epoche
        """
        if learning_rate is None:
            learning_rate = self.learning_rate

        history = []
        for epoch in range(epochs):
            total_cost = 0.0
            for x, y in zip(inputs, targets):
                output = self.forward(x)
                target = np.array(y, dtype=float)
                # Kosten J = 1/2 * (A - Y)^2, summiert über die Output-Neuronen
                total_cost += 0.5 * np.sum((output - target) ** 2)
                self.backward(output, target, learning_rate)

            mean_cost = total_cost / len(inputs)
            history.append(mean_cost)

            if verbose_every and (epoch % verbose_every == 0 or epoch == epochs - 1):
                print(f"Epoche {epoch:6d}  Kosten: {mean_cost:.6f}")

        return history

    # ------------------------------------------------------------------
    # Part 1 / Step 4: Export und Import als JSON
    # ------------------------------------------------------------------
    def dump_json(self, path):
        """Speichert Architektur, Aktivierung, Lernrate und alle Gewichte/Biases als JSON."""
        data = {
            "layer_sizes": self.layer_sizes,
            "activation": self.activation_name,
            "learning_rate": self.learning_rate,
            # Pro Layer eine Liste von Neuronen, jedes mit seinen Gewichten und seinem Bias.
            # tolist()/float() nötig, weil json keine numpy-Typen kennt.
            "layers": [
                [{"weights": n.weights.tolist(), "bias": float(n.bias)} for n in layer.neurons]
                for layer in self.layers
            ],
        }
        _ordner_anlegen(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path):
        """Baut ein Netz aus einer mit dump_json() gespeicherten Datei wieder auf."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # Zuerst ein Netz mit derselben Architektur erzeugen (hat noch Zufallsgewichte) ...
        net = cls(data["layer_sizes"], data["learning_rate"], data["activation"])
        # ... und dann die gespeicherten Werte darüberschreiben.
        for layer, layer_data in zip(net.layers, data["layers"]):
            for neuron, neuron_data in zip(layer.neurons, layer_data):
                neuron.weights = np.array(neuron_data["weights"], dtype=float)
                neuron.bias = float(neuron_data["bias"])
        return net

    # ------------------------------------------------------------------
    # Part 1 / Step 5: Export als lesbare .txt
    # ------------------------------------------------------------------
    def dump_txt(self, path):
        """
        Exportiert das Netz als Textdatei, die man mit freiem Auge prüfen kann.
        Zeilen mit '#' sind nur Erklärungen für Menschen. Das C-Programm (Part 4)
        überspringt sie und liest die Zahlen einfach der Reihe nach ein:
          Anzahl Layer -> Layer-Größen -> pro Neuron: bias w0 w1 ...
        """
        zeilen = [
            "# Neuronales Netz - Export aus neural_network.py",
            f"# Aktivierung: {self.activation_name}",
            "# Anzahl Layer (ohne Input)",
            str(len(self.layers)),
            "# Layer-Groessen inkl. Input",
            " ".join(str(s) for s in self.layer_sizes),
        ]
        for l, layer in enumerate(self.layers, start=1):
            anz_inputs = layer.neurons[0].num_inputs
            gewicht_namen = " ".join(f"w{j}" for j in range(anz_inputs))
            zeilen.append("")
            zeilen.append(f"# ===== Layer {l}: {len(layer.neurons)} Neuronen, je {anz_inputs} Eingaenge =====")
            for i, neuron in enumerate(layer.neurons):
                zeilen.append(f"# Layer {l}, Neuron {i}: bias {gewicht_namen}")
                werte = [neuron.bias, *neuron.weights]
                zeilen.append(" ".join(f"{v:.10f}" for v in werte))

        _ordner_anlegen(path)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(zeilen) + "\n")

    # ------------------------------------------------------------------
    # Part 1 / Step 6: C-Header (.h) zum Initialisieren des Netzes
    # ------------------------------------------------------------------
    def dump_initfile(self, path, name="nn"):
        """
        Erzeugt einen C-Header, mit dem ein C-Programm das Netz ohne Datei-Einlesen
        aufbauen kann (Part 5). Inhalt: Größen als #define, Gewichte und Biases als
        Arrays und die Funktion <name>_init(Network *net).

        Das C-Programm muss VOR dem #include diese Structs definieren:
          Neuron  { int num_inputs; float *weights; float bias; float output; }
          Layer   { int num_neurons; Neuron *neurons; }
          Network { int num_layers; Layer *layers; }
        name wird Präfix aller C-Namen (z.B. "xor", "seg7") und muss ein gültiger
        C-Bezeichner sein, darum "seg7" statt "7seg".
        """
        if not name.isidentifier():
            raise ValueError(f"'{name}' ist kein gültiger C-Name")
        N = name.upper()
        guard = f"{N}_NET_H"

        h = [
            "/* Automatisch erzeugt von neural_network.py - nicht von Hand bearbeiten! */",
            f"#ifndef {guard}",
            f"#define {guard}",
            "",
            f"#define {N}_NUM_LAYERS  {len(self.layers)}   /* ohne Input-Layer */",
            f"#define {N}_NUM_INPUTS  {self.layer_sizes[0]}",
            f"#define {N}_NUM_OUTPUTS {self.layer_sizes[-1]}",
            f"static const int {name}_layer_sizes[] = {{{', '.join(map(str, self.layer_sizes))}}};",
        ]

        # Pro Layer: Gewichte als 2D-Array [Neuron][Eingang], Biases als 1D-Array
        # und ein (noch leeres) Neuron-Array, das die init-Funktion befüllt.
        for l, layer in enumerate(self.layers, start=1):
            anz_neuronen = len(layer.neurons)
            anz_inputs = layer.neurons[0].num_inputs
            h.append("")
            h.append(f"/* Layer {l}: {anz_neuronen} Neuronen mit je {anz_inputs} Eingaengen */")
            h.append(f"static float {name}_w{l}[{anz_neuronen}][{anz_inputs}] = {{")
            for neuron in layer.neurons:
                h.append("    {" + ", ".join(f"{w:.10f}f" for w in neuron.weights) + "},")
            h.append("};")
            biases = ", ".join(f"{n.bias:.10f}f" for n in layer.neurons)
            h.append(f"static float {name}_b{l}[{anz_neuronen}] = {{{biases}}};")
            h.append(f"static Neuron {name}_neurons{l}[{anz_neuronen}];")

        h.append(f"static Layer {name}_layers[{N}_NUM_LAYERS];")

        # Die init-Funktion hängt die Arrays von oben in die Structs ein (nur Zeiger,
        # es wird nichts kopiert und kein malloc gebraucht).
        h.append("")
        h.append("/* Verbindet die Arrays oben mit den Structs - danach ist das Netz einsatzbereit */")
        h.append(f"static void {name}_init(Network *net) {{")
        for l, layer in enumerate(self.layers, start=1):
            h += [
                f"    for (int i = 0; i < {len(layer.neurons)}; i++) {{",
                f"        {name}_neurons{l}[i].num_inputs = {layer.neurons[0].num_inputs};",
                f"        {name}_neurons{l}[i].weights    = {name}_w{l}[i];",
                f"        {name}_neurons{l}[i].bias       = {name}_b{l}[i];",
                f"        {name}_neurons{l}[i].output     = 0.0f;",
                "    }",
                f"    {name}_layers[{l - 1}].num_neurons = {len(layer.neurons)};",
                f"    {name}_layers[{l - 1}].neurons     = {name}_neurons{l};",
            ]
        h += [
            f"    net->num_layers = {N}_NUM_LAYERS;",
            f"    net->layers     = {name}_layers;",
            "}",
            "",
            f"#endif /* {guard} */",
        ]

        _ordner_anlegen(path)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(h) + "\n")


# ======================================================================
# Part 2: Training und Persistenz
# ======================================================================

# Alle Exporte landen im Ordner "export" neben dieser Datei (egal von wo man startet)
EXPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "export")

# Trainingsparameter laut Angabe (gelten für XOR und 7-Segment)
EPOCHS = 20000
LEARNING_RATE = 0.1

# Trainingsdaten XOR
X_XOR = [[0, 0],
         [0, 1],
         [1, 0],
         [1, 1]]

Y_XOR = [[0],
         [1],
         [1],
         [0]]

# Trainingsdaten 7-Segment: Ziffer 0-9 als 4 Bit -> Segmente a b c d e f g
#
#      aaa
#     f   b
#      ggg
#     e   c
#      ddd
X_7SEG = [[0, 0, 0, 0],   # 0
          [0, 0, 0, 1],   # 1
          [0, 0, 1, 0],   # 2
          [0, 0, 1, 1],   # 3
          [0, 1, 0, 0],   # 4
          [0, 1, 0, 1],   # 5
          [0, 1, 1, 0],   # 6
          [0, 1, 1, 1],   # 7
          [1, 0, 0, 0],   # 8
          [1, 0, 0, 1]]   # 9

Y_7SEG = [[1, 1, 1, 1, 1, 1, 0],   # 0
          [0, 1, 1, 0, 0, 0, 0],   # 1
          [1, 1, 0, 1, 1, 0, 1],   # 2
          [1, 1, 1, 1, 0, 0, 1],   # 3
          [0, 1, 1, 0, 0, 1, 1],   # 4
          [1, 0, 1, 1, 0, 1, 1],   # 5
          [1, 0, 1, 1, 1, 1, 1],   # 6
          [1, 1, 1, 0, 0, 0, 0],   # 7
          [1, 1, 1, 1, 1, 1, 1],   # 8
          [1, 1, 1, 1, 0, 1, 1]]   # 9


def exportieren(net, basisname, c_name):
    """Speichert das Netz als export/<basisname>.json, .txt und .h."""
    pfad = os.path.join(EXPORT_DIR, basisname)
    net.dump_json(pfad + ".json")
    net.dump_txt(pfad + ".txt")
    net.dump_initfile(pfad + ".h", c_name)
    print(f"Exportiert: export/{basisname}.json, .txt, .h")


def ergebnisse_ausgeben(net, X, Y, original=None, toleranz=1e-9):
    """
    Gibt pro Sample Eingabe, Soll, gerundete Ausgabe und Rohwerte als Tabelle aus.
    Gerundet wird mit Schwelle 0.5 (ab 0.5 -> 1, sonst 0).
    Wird 'original' übergeben, prüft die Funktion zusätzlich, ob 'net' exakt
    dieselben Werte liefert wie 'original' (= Test, ob Export/Import funktioniert).
    Rückgabe: Anzahl der Samples, bei denen ALLE Ausgaben stimmen.
    """
    print(f"{'Input':<14}{'Soll':<10}{'Ist':<10}{'OK':<6}Rohwerte")
    samples_ok = 0
    bits_ok = 0
    max_abweichung = 0.0

    for x, y in zip(X, Y):
        out = net.predict(x)
        gerundet = (out >= 0.5).astype(int)
        richtig = np.array_equal(gerundet, y)
        samples_ok += richtig
        bits_ok += int(np.sum(gerundet == y))

        if original is not None:
            abweichung = np.max(np.abs(out - original.predict(x)))
            max_abweichung = max(max_abweichung, abweichung)

        soll = "".join(map(str, y))
        ist = "".join(map(str, gerundet))
        roh = " ".join(f"{v:.2f}" for v in out)
        print(f"{str(x):<14}{soll:<10}{ist:<10}{'ja' if richtig else 'NEIN':<6}{roh}")

    # Accuracy: einmal pro Sample (alles richtig?) und einmal pro einzelnem Ausgang
    anz_bits = len(Y) * len(Y[0])
    print(f"\nSamples korrekt: {samples_ok}/{len(X)}")
    print(f"Ausgaenge korrekt: {bits_ok}/{anz_bits} ({100 * bits_ok / anz_bits:.1f} %)")

    if original is not None:
        status = "OK" if max_abweichung <= toleranz else "FEHLER"
        print(f"Max. Abweichung zum Original-Netz: {max_abweichung:.2e} -> Export/Import {status}")

    return samples_ok


def test_geladenes_netz(original, basisname, X, Y):
    """Part 2 / Step 2+4: lädt export/<basisname>.json und testet es gegen Soll und Original."""
    print(f"\n--- Geladenes Netz aus export/{basisname}.json ---")
    geladen = NeuralNetwork.load(os.path.join(EXPORT_DIR, basisname + ".json"))
    ergebnisse_ausgeben(geladen, X, Y, original=original)
    return geladen


def train_xor():
    """Part 2 / Step 1: XOR trainieren (2 -> 2 -> 1) und exportieren."""
    print("=" * 60)
    print("XOR-Gatter (2 -> 2 -> 1)")
    print("=" * 60)

    # Seed setzen, damit der Lauf reproduzierbar ist.
    # Ohne Seed landet das Netz je nach Startgewichten manchmal in einem
    # lokalen Minimum und lernt XOR mit nur 2 Hidden-Neuronen nicht sauber.
    np.random.seed(1)

    net = NeuralNetwork([2, 2, 1], learning_rate=LEARNING_RATE)

    print(f"Training: {EPOCHS} Epochen, Lernrate {LEARNING_RATE}\n")
    net.train(X_XOR, Y_XOR, epochs=EPOCHS, verbose_every=5000)

    print("\n--- Trainiertes Netz ---")
    ergebnisse_ausgeben(net, X_XOR, Y_XOR)
    exportieren(net, "xor", "xor")
    return net


def train_7seg():
    """Part 2 / Step 3: 7-Segment-Decoder trainieren (4 -> 16 -> 7) und exportieren."""
    print("\n" + "=" * 60)
    print("7-Segment-Decoder (4 -> 16 -> 7)")
    print("=" * 60)

    np.random.seed(1)
    net = NeuralNetwork([4, 16, 7], learning_rate=LEARNING_RATE)

    print(f"Training: {EPOCHS} Epochen, Lernrate {LEARNING_RATE}\n")
    net.train(X_7SEG, Y_7SEG, epochs=EPOCHS, verbose_every=5000)

    print("\n--- Trainiertes Netz ---")
    ergebnisse_ausgeben(net, X_7SEG, Y_7SEG)
    # "seg7" statt "7seg", weil C-Namen nicht mit einer Ziffer beginnen dürfen
    exportieren(net, "7seg", "seg7")
    return net


if __name__ == "__main__":
    # Step 1 + 2: XOR trainieren, exportieren, wieder laden und vergleichen
    xor_net = train_xor()
    test_geladenes_netz(xor_net, "xor", X_XOR, Y_XOR)

    # Step 3 + 4: dasselbe für den 7-Segment-Decoder
    seg_net = train_7seg()
    test_geladenes_netz(seg_net, "7seg", X_7SEG, Y_7SEG)
