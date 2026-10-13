/* Automatisch erzeugt von neural_network.py - nicht von Hand bearbeiten! */
#ifndef XOR_NET_H
#define XOR_NET_H

#define XOR_NUM_LAYERS  2   /* ohne Input-Layer */
#define XOR_NUM_INPUTS  2
#define XOR_NUM_OUTPUTS 1
static const int xor_layer_sizes[] = {2, 2, 1};

/* Layer 1: 2 Neuronen mit je 2 Eingaengen */
static float xor_w1[2][2] = {
    {-4.0997692910f, -4.1094350341f},
    {-5.8608813903f, -5.9183356281f},
};
static float xor_b1[2] = {6.0658643292f, 2.2666433275f};
static Neuron xor_neurons1[2];

/* Layer 2: 1 Neuronen mit je 2 Eingaengen */
static float xor_w2[1][2] = {
    {8.0835745337f, -8.3532008407f},
};
static float xor_b2[1] = {-3.7342474438f};
static Neuron xor_neurons2[1];
static Layer xor_layers[XOR_NUM_LAYERS];

/* Verbindet die Arrays oben mit den Structs - danach ist das Netz einsatzbereit */
static void xor_init(Network *net) {
    for (int i = 0; i < 2; i++) {
        xor_neurons1[i].num_inputs = 2;
        xor_neurons1[i].weights    = xor_w1[i];
        xor_neurons1[i].bias       = xor_b1[i];
        xor_neurons1[i].output     = 0.0f;
    }
    xor_layers[0].num_neurons = 2;
    xor_layers[0].neurons     = xor_neurons1;
    for (int i = 0; i < 1; i++) {
        xor_neurons2[i].num_inputs = 2;
        xor_neurons2[i].weights    = xor_w2[i];
        xor_neurons2[i].bias       = xor_b2[i];
        xor_neurons2[i].output     = 0.0f;
    }
    xor_layers[1].num_neurons = 1;
    xor_layers[1].neurons     = xor_neurons2;
    net->num_layers = XOR_NUM_LAYERS;
    net->layers     = xor_layers;
}

#endif /* XOR_NET_H */
