/*
 * Part 4: Forward-Inference in C, Gewichte werden aus der .txt geladen.
 *
 * Kompilieren:  clang -Wall -o nn_forward nn_forward.c -lm   (oder: make)
 * Starten:      ./nn_forward [pfad/zur/datei.txt]    Standard: ../export/7seg.txt
 */
#include <ctype.h>
#include <stdlib.h>

#include "nn.h"

/*
 * Liest die nächste Zahl aus der Datei und überspringt dabei Kommentarzeilen (#...).
 * Rückgabe: 1 = Zahl gelesen, 0 = Dateiende/Fehler.
 */
static int lese_zahl(FILE *f, float *wert) {
    int c;
    while ((c = fgetc(f)) != EOF) {
        if (c == '#') {
            /* Kommentar: alles bis zum Zeilenende ignorieren */
            while ((c = fgetc(f)) != EOF && c != '\n') {}
        } else if (!isspace(c)) {
            /* Anfang einer Zahl gefunden -> Zeichen zurücklegen und mit fscanf lesen */
            ungetc(c, f);
            return fscanf(f, "%f", wert) == 1;
        }
    }
    return 0;
}

/*
 * Baut das Netz aus der .txt von dump_txt() auf. Reihenfolge in der Datei:
 *   Anzahl Layer -> Layer-Größen (inkl. Input) -> pro Neuron: bias w0 w1 ...
 * Rückgabe: 1 = ok, 0 = Fehler.
 */
static int load_from_txt(Network *net, const char *pfad) {
    FILE *f = fopen(pfad, "r");
    if (!f) {
        perror(pfad);
        return 0;
    }

    float wert;
    if (!lese_zahl(f, &wert)) goto fehler;
    net->num_layers = (int)wert;

    /* Layer-Größen einlesen: num_layers + 1 Werte, weil der Input-Layer mitzählt */
    int groessen[16];
    if (net->num_layers < 1 || net->num_layers + 1 > 16) goto fehler;
    for (int i = 0; i <= net->num_layers; i++) {
        if (!lese_zahl(f, &wert)) goto fehler;
        groessen[i] = (int)wert;
        if (groessen[i] < 1 || groessen[i] > MAX_NEURONS) goto fehler;
    }

    net->layers = calloc(net->num_layers, sizeof(Layer));
    for (int l = 0; l < net->num_layers; l++) {
        Layer *layer = &net->layers[l];
        layer->num_neurons = groessen[l + 1];
        layer->neurons = calloc(layer->num_neurons, sizeof(Neuron));

        for (int i = 0; i < layer->num_neurons; i++) {
            Neuron *n = &layer->neurons[i];
            n->num_inputs = groessen[l];   /* Eingänge = Größe des vorherigen Layers */
            n->weights = malloc(n->num_inputs * sizeof(float));

            if (!lese_zahl(f, &n->bias)) goto fehler;
            for (int j = 0; j < n->num_inputs; j++) {
                if (!lese_zahl(f, &n->weights[j])) goto fehler;
            }
        }
    }
    fclose(f);
    return 1;

fehler:
    fprintf(stderr, "Fehler: %s hat nicht das erwartete Format\n", pfad);
    fclose(f);
    return 0;
}

/* Gibt den mit malloc/calloc reservierten Speicher wieder frei */
static void free_network(Network *net) {
    for (int l = 0; l < net->num_layers; l++) {
        for (int i = 0; i < net->layers[l].num_neurons; i++) {
            free(net->layers[l].neurons[i].weights);
        }
        free(net->layers[l].neurons);
    }
    free(net->layers);
}

int main(int argc, char **argv) {
    const char *pfad = argc > 1 ? argv[1] : "../export/7seg.txt";
    Network net = {0};

    printf("Part 4: 7-Segment-Netz aus %s\n\n", pfad);
    if (!load_from_txt(&net, pfad)) {
        return 1;
    }

    /* Der Test braucht ein Netz mit 4 Eingängen und 7 Ausgängen (z.B. nicht xor.txt) */
    int eingaenge = net.layers[0].neurons[0].num_inputs;
    int ausgaenge = net.layers[net.num_layers - 1].num_neurons;
    if (eingaenge != 4 || ausgaenge != 7) {
        fprintf(stderr, "Fehler: Netz hat %d Eingaenge / %d Ausgaenge, erwartet 4 / 7\n",
                eingaenge, ausgaenge);
        free_network(&net);
        return 1;
    }

    int richtig = test_7seg(&net);
    free_network(&net);
    return richtig == 10 ? 0 : 1;
}
