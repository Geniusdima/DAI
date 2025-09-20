CC = gcc
CFLAGS = -g
LDFLAGS = -lm -march=native

TARGET = a.out
SOURCES = main.c ./dai/aimath.c ./dai/aiperceptron.c ./dai/aibase.c

ifdef AVX_SUPPORT
    SOURCES += ./dai/fastComputing/aiavx.c
else
    SOURCES += ./dai/fastComputing/aiscalar.c
endif


all:
	$(CC) $(CFLAGS) $(SOURCES) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)
