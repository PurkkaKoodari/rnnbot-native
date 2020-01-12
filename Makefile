rnn: *.c *.h
	gcc -g -Wall -O3 -o rnn rnn.c unicode.c map.c randdouble.c -lm

debug:
	gcc -g -Wall -O0 -fsanitize=address -o rnn rnn.c unicode.c map.c randdouble.c -lm

clean:
	rm rnn
