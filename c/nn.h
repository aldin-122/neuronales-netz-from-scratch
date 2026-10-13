/*
 * nn.h - Gemeinsamer Teil für Part 4 und Part 5:
 * Structs für Neuron/Layer/Network, forward(), predict() und ein Test für den 7-Segment-Decoder.
 * Nur Forward-Pass (kein Training) - die Gewichte kommen fertig aus Python.
 */
#ifndef NN_H
#define NN_H

#include <math.h>
#include <stdio.h>

/* Größter erlaubter Layer, reicht für unsere Netze (max. 16 Neuronen) */
#define MAX_NEURONS 64

typedef struct {
    int num_inputs;   /* Anzahl Eingänge = Anzahl Gewichte */
    float *weights;
    float bias;
    float output;     /* letzte berechnete Ausgabe */
} Neuron;

typedef struct {
    int num_neurons;
    Neuron *neurons;
} Layer;

typedef struct {
    int num_layers;   /* ohne Input-Layer, wie in Python */
    Layer *layers;
} Network;

static float sigmoid(float z) {
    return 1.0f / (1.0f + expf(-z));
}

/*
 * Forward-Pass: pro Layer für jedes Neuron  output = sigmoid(bias + Summe(w_j * x_j)).
 * Die Ausgaben eines Layers sind die Eingaben des nächsten Layers.
 */
static void forward(Network *net, const float *input) {
    float buf[MAX_NEURONS];
    const float *x = input;

    for (int l = 0; l < net->num_layers; l++) {
        Layer *layer = &net->layers[l];

        for (int i = 0; i < layer->num_neurons; i++) {
            Neuron *n = &layer->neurons[i];
            float z = n->bias;
            for (int j = 0; j < n->num_inputs; j++) {
                z += n->weights[j] * x[j];
            }
            n->output = sigmoid(z);
        }

        /* Erst NACH dem ganzen Layer in den Puffer kopieren, weil x beim nächsten
           Layer auf buf zeigt und wir die Eingaben sonst mitten drin überschreiben würden. */
        for (int i = 0; i < layer->num_neurons; i++) {
            buf[i] = layer->neurons[i].output;
        }
        x = buf;
    }
}

/* Führt forward() aus und kopiert die Ausgaben des letzten Layers nach output */
static void predict(Network *net, const float *input, float *output) {
    forward(net, input);
    Layer *last = &net->layers[net->num_layers - 1];
    for (int i = 0; i < last->num_neurons; i++) {
        output[i] = last->neurons[i].output;
    }
}

/*
 * Testet ein 7-Segment-Netz mit den Ziffern 0-9 (Sollwerte wie im PDF).
 * Gibt eine Tabelle aus und liefert die Anzahl komplett richtiger Ziffern zurück.
 */
static int test_7seg(Network *net) {
    static const int soll[10][7] = {
        {1, 1, 1, 1, 1, 1, 0}, {0, 1, 1, 0, 0, 0, 0}, {1, 1, 0, 1, 1, 0, 1},
        {1, 1, 1, 1, 0, 0, 1}, {0, 1, 1, 0, 0, 1, 1}, {1, 0, 1, 1, 0, 1, 1},
        {1, 0, 1, 1, 1, 1, 1}, {1, 1, 1, 0, 0, 0, 0}, {1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 0, 1, 1},
    };
    int richtig = 0;

    printf("Ziffer  Input     Soll     Ist      OK    Rohwerte\n");
    for (int ziffer = 0; ziffer < 10; ziffer++) {
        /* Ziffer -> 4 Bit, höchstes Bit zuerst (z.B. 5 -> 0 1 0 1) */
        float input[4];
        for (int b = 0; b < 4; b++) {
            input[b] = (float)((ziffer >> (3 - b)) & 1);
        }

        float out[7];
        predict(net, input, out);

        int ok = 1;
        char soll_str[8], ist_str[8];
        for (int s = 0; s < 7; s++) {
            int bit = out[s] >= 0.5f;
            if (bit != soll[ziffer][s]) ok = 0;
            soll_str[s] = '0' + soll[ziffer][s];
            ist_str[s] = '0' + bit;
        }
        soll_str[7] = ist_str[7] = '\0';
        richtig += ok;

        printf("%d       %.0f%.0f%.0f%.0f      %s  %s  %-4s ", ziffer,
               input[0], input[1], input[2], input[3], soll_str, ist_str, ok ? "ja" : "NEIN");
        for (int s = 0; s < 7; s++) printf(" %.4f", out[s]);
        printf("\n");
    }
    printf("\n%d/10 Ziffern korrekt\n", richtig);
    return richtig;
}

#endif /* NN_H */
