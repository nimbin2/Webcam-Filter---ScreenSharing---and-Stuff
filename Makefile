# SPDX-License-Identifier: GPL-2.0-only
# vcam — Copyright (C) 2026 the vcam authors. GPL-2.0, see LICENSE.
CC      := gcc
CFLAGS  := -std=c11 -O2 -Wall -Wextra -Wpedantic \
            -D_POSIX_C_SOURCE=200809L \
            -I include \
            $(shell pkg-config --cflags libavcodec libavformat libavutil libswscale) \
            $(shell pkg-config --cflags libpng)

LDFLAGS := $(shell pkg-config --libs libavcodec libavformat libavutil libswscale libavdevice) \
            $(shell pkg-config --libs libpng) \
            -lpthread -lm

# Installation paths — overridable on the command line:
#   make install                       (default: /usr/local)
#   make install PREFIX=$HOME/.local   (user-local)
#   make install PREFIX=/usr           (system-wide)
#   make install DESTDIR=/tmp/stage    (staged install for packaging)
PREFIX  ?= /usr/local
BINDIR  ?= $(PREFIX)/bin
DOCDIR  ?= $(PREFIX)/share/doc/vcam

SRCS := src/main.c src/frame.c src/capture.c src/output.c \
        src/overlay.c src/registry.c src/ipc.c src/filter.c \
        src/record.c
OBJS := $(SRCS:.c=.o)
BIN  := vcam
TEST := test_screen

.PHONY: all clean install uninstall

all: $(BIN) $(TEST)

$(BIN): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(TEST): src/test_screen.c
	$(CC) -std=c11 -O2 -Wall -D_POSIX_C_SOURCE=200809L -D_DEFAULT_SOURCE \
	    -o $@ src/test_screen.c

src/%.o: src/%.c include/vcam.h
	$(CC) $(CFLAGS) -c -o $@ $<

install: $(BIN)
	install -d $(DESTDIR)$(BINDIR)
	install -m 755 $(BIN)               $(DESTDIR)$(BINDIR)/
	install -m 755 scripts/vcam-ctl     $(DESTDIR)$(BINDIR)/
	install -m 755 scripts/vcam-setup   $(DESTDIR)$(BINDIR)/
	install -m 755 scripts/vcam-tune    $(DESTDIR)$(BINDIR)/
	install -d $(DESTDIR)$(DOCDIR)
	install -m 644 LICENSE              $(DESTDIR)$(DOCDIR)/
	install -m 644 NOTICE               $(DESTDIR)$(DOCDIR)/
	install -m 644 README.md            $(DESTDIR)$(DOCDIR)/
	install -m 644 INSTALL.md           $(DESTDIR)$(DOCDIR)/
	install -m 644 EXPRESSIONS.md       $(DESTDIR)$(DOCDIR)/
	@echo
	@echo "Installed to $(DESTDIR)$(BINDIR)/  (docs in $(DESTDIR)$(DOCDIR)/)"
	@echo "Make sure that's on your PATH."

uninstall:
	rm -f $(DESTDIR)$(BINDIR)/$(BIN)
	rm -f $(DESTDIR)$(BINDIR)/vcam-ctl
	rm -f $(DESTDIR)$(BINDIR)/vcam-setup
	rm -f $(DESTDIR)$(BINDIR)/vcam-tune
	rm -rf $(DESTDIR)$(DOCDIR)

clean:
	rm -f $(OBJS) $(BIN) $(TEST)
