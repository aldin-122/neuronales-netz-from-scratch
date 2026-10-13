# "make" baut beide Programme, "make run" baut und startet sie, "make clean" räumt auf.
CC = clang
CFLAGS = -Wall -Wextra -O2

all: nn_forward nn_header

nn_forward: nn_forward.c nn.h
	$(CC) $(CFLAGS) -o $@ nn_forward.c -lm

nn_header: nn_header_main.c nn.h ../export/7seg.h
	$(CC) $(CFLAGS) -o $@ nn_header_main.c -lm

run: all
	./nn_forward
	@echo
	./nn_header

clean:
	rm -f nn_forward nn_header
