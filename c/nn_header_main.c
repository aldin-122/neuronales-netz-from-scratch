/*
 * Part 5: Forward-Inference in C, das Netz wird über den generierten Header initialisiert.
 * Es wird keine Datei gelesen - die Gewichte sind beim Kompilieren schon im Programm.
 *
 * Kompilieren:  clang -Wall -o nn_header nn_header_main.c -lm   (oder: make)
 * Starten:      ./nn_header
 */
#include "nn.h"               /* zuerst: Structs, forward(), predict() */
#include "../export/7seg.h"   /* danach: Gewichte + seg7_init(), braucht die Structs */

int main(void) {
    Network net;
    seg7_init(&net);

    printf("Part 5: 7-Segment-Netz aus export/7seg.h (%d Layer, %d Inputs, %d Outputs)\n\n",
           SEG7_NUM_LAYERS, SEG7_NUM_INPUTS, SEG7_NUM_OUTPUTS);

    int richtig = test_7seg(&net);
    return richtig == 10 ? 0 : 1;
}
