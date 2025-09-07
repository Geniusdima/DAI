CC = gcc
CFLAGS = -g
LDFLAGS = -lm

TARGET = a.out
SOURCES = main.c ./dai/aimath.c ./dai/aiperceptron.c ./dai/aibase.c

all:
	$(CC) $(CFLAGS) $(SOURCES) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)
