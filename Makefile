CC      := gcc
CFLAGS  := -std=c11 -O2 -Wall -Wextra -Wpedantic \
            -D_POSIX_C_SOURCE=200809L \
            -I include \
            $(shell pkg-config --cflags libavcodec libavformat libavutil libswscale) \
            $(shell pkg-config --cflags libpng)

LDFLAGS := $(shell pkg-config --libs libavcodec libavformat libavutil libswscale libavdevice) \
            $(shell pkg-config --libs libpng) \
            -lpthread -lm

SRCS := src/main.c src/frame.c src/capture.c src/output.c \
        src/overlay.c src/registry.c src/ipc.c src/filter.c
OBJS := $(SRCS:.c=.o)
BIN  := vcam
TEST := test_screen

.PHONY: all clean install

all: $(BIN) $(TEST)

$(BIN): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(TEST): src/test_screen.c
	$(CC) -std=c11 -O2 -Wall -D_POSIX_C_SOURCE=200809L -D_DEFAULT_SOURCE \
	    -o $@ src/test_screen.c

src/%.o: src/%.c include/vcam.h
	$(CC) $(CFLAGS) -c -o $@ $<

install: $(BIN)
	install -m 755 $(BIN) /usr/local/bin/
	install -m 755 scripts/vcam-ctl /usr/local/bin/
	install -m 755 scripts/vcam-setup /usr/local/bin/
	install -m 755 scripts/vcam-tune /usr/local/bin/

clean:
	rm -f $(OBJS) $(BIN) $(TEST)
