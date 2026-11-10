# SPDX-License-Identifier: GPL-2.0-only
# vcam — Copyright (C) 2026 the vcam authors. GPL-2.0, see LICENSE.

CC      := gcc
CXX     := g++

CFLAGS  := -std=c23 -O2 -Wall -Wextra -Wpedantic \
            -D_POSIX_C_SOURCE=200809L \
            -I include \
            $(shell pkg-config --cflags libavcodec libavformat libavutil libswscale) \
            $(shell pkg-config --cflags libpng)

LDFLAGS := $(shell pkg-config --libs libavcodec libavformat libavutil libswscale libavdevice) \
            $(shell pkg-config --libs libpng) \
            -lpthread -lm

# ── Optional OpenCV support ─────────────────────────────────────────
# Off by default: vcam builds as a pure-C program with no OpenCV
# dependency, and the cv_* effects resolve to no-op stubs.
#
#   make               # no OpenCV (default)
#   make OPENCV=1      # enable the OpenCV-backed filters
#
# When enabled, the OpenCV effects live in a small C++ shim
# (src/opencv_fx.cpp) that exposes a flat C ABI. The final binary is
# then linked with the C++ driver so the C++ runtime is pulled in.
#
# The filters are written OpenCV-5-first: built against 5.x they use
# the new graph-based DNN engine (ENGINE_AUTO — 80%+ ONNX operator
# coverage, shape inference, operator fusion) and first-class FP16
# (cv::hfloat) arithmetic for the segmentation blend. Built against
# 4.5–4.x they compile the classic-engine/FP32 fallback path, chosen
# at compile time from CV_VERSION_MAJOR. Run 'vcam-ctl opencv' on a
# live vcam to see which path your build uses.
#
# The pkg-config MODULE NAME is not tied to the major version and
# varies by packaging: it may be 'opencv5', 'opencv4' (the name used
# for "4 and later" — an OpenCV 5 install often still ships
# opencv4.pc), or plain 'opencv'. We probe all three and pick the
# first that exists, then require version >= 4.5. Override with
#   make OPENCV=1 OPENCV_PC=<name>
# Sticky default: if a previous build left a config stamp
# (.built-opencv-0/1), default OPENCV to that value, so
#   make OPENCV=1
#   sudo make install        <- installs the OpenCV flavor, as intended
# just works. An explicit OPENCV=0/1 on the command line always wins,
# and 'make clean' resets the memory.
LAST_STAMP  := $(firstword $(wildcard .built-opencv-*))
# stamp form: .built-opencv-<0|1>-ort-<0|1> ; split on '-' ->
#   word1=".built" word2="opencv" word3=<0|1> word4="ort" word5=<0|1>
LAST_OPENCV := $(word 3,$(subst -, ,$(LAST_STAMP)))
LAST_ORT    := $(word 5,$(subst -, ,$(LAST_STAMP)))
ifeq ($(strip $(LAST_OPENCV)),)
    OPENCV ?= 0
else
    OPENCV ?= $(LAST_OPENCV)
endif
ifeq ($(strip $(LAST_ORT)),)
    ORT ?= 0
else
    ORT ?= $(LAST_ORT)
endif
# the stamp file's CONTENT stores the ORT_HOME used, so plain
# 'sudo make install' after 'make OPENCV=1 ORT=1 ORT_HOME=…' just works
ifneq ($(strip $(LAST_STAMP)),)
    LAST_ORT_HOME := $(strip $(shell cat $(LAST_STAMP) 2>/dev/null))
    ifneq ($(strip $(LAST_ORT_HOME)),)
        ORT_HOME ?= $(LAST_ORT_HOME)
    endif
endif

ifeq ($(OPENCV),1)
    # Let the user force a specific pkg-config module name if they want.
    ifndef OPENCV_PC
        # First module in the list that pkg-config knows about wins.
        OPENCV_PC := $(firstword $(foreach m,opencv5 opencv4 opencv,\
                        $(shell pkg-config --exists $(m) 2>/dev/null && echo $(m))))
    endif
    ifeq ($(strip $(OPENCV_PC)),)
        $(error OPENCV=1 but no OpenCV pkg-config module (tried opencv5, opencv4, opencv). \
                Install OpenCV 4.5+ dev files, set OPENCV_PC=<name>, or build without OPENCV=1)
    endif
    # The module must actually exist (covers a hand-set OPENCV_PC typo).
    ifneq ($(shell pkg-config --exists $(OPENCV_PC) && echo ok),ok)
        $(error OpenCV pkg-config module '$(OPENCV_PC)' not found. \
                Check the name (try 'pkg-config --list-all | grep -i opencv'), or unset OPENCV=1)
    endif
    # Require a version new enough to have the APIs the filters use.
    ifneq ($(shell pkg-config --atleast-version=4.5 $(OPENCV_PC) && echo ok),ok)
        $(error OpenCV '$(OPENCV_PC)' is version $(shell pkg-config --modversion $(OPENCV_PC)); \
                vcam needs >= 4.5 (FaceDetectorYN + dnn). Upgrade OpenCV or unset OPENCV=1)
    endif
    CV_VER    := $(shell pkg-config --modversion $(OPENCV_PC))
    # -isystem instead of -I: OpenCV's own headers shouldn't trip our -Wall
    CV_CFLAGS := $(patsubst -I%,-isystem %,$(shell pkg-config --cflags $(OPENCV_PC)))
    CV_LIBS   := $(shell pkg-config --libs   $(OPENCV_PC))
    CXXFLAGS  := -std=c++26 -O2 -Wall -Wextra -I include $(CV_CFLAGS)
    CV_OBJ    := src/opencv_fx.o
    LDFLAGS   += $(CV_LIBS)
    # Optional X11: powers the 'virtual-room' screen-region grabber.
    # Auto-detected; without libx11-dev the mode explains what to install.
    ifeq ($(shell pkg-config --exists x11 && echo ok),ok)
        CXXFLAGS += -DVCAM_HAVE_X11 $(shell pkg-config --cflags x11)
        LDFLAGS  += $(shell pkg-config --libs x11)
    endif
    LINK      := $(CXX)
    CFLAGS    += -DVCAM_HAVE_OPENCV=1
    $(info vcam: OpenCV enabled — module '$(OPENCV_PC)' version $(CV_VER))
else
    CV_OBJ   := src/opencv_stub.o
    LINK     := $(CC)
    CFLAGS   += -DVCAM_HAVE_OPENCV=0
endif

# ── Optional ONNX Runtime backend (ORT=1) ──────────────────────────
# Runs the RVM video-matting model CORRECTLY (OpenCV 5's young graph
# engine loads it but miscomputes it — verified). Opt-in:
#
#   vcam-setup --ort            # one-time: fetch onnxruntime to
#                               #   ~/.local/share/vcam/onnxruntime
#   make OPENCV=1 ORT=1         # build with the backend
#
# ORT_HOME points at a directory containing include/onnxruntime_c_api.h
# and lib/libonnxruntime.so; override it for a custom install:
#   make OPENCV=1 ORT=1 ORT_HOME=/opt/onnxruntime
ifeq ($(ORT),1)
    ifneq ($(OPENCV),1)
        $(error ORT=1 requires OPENCV=1 (the matting pipeline lives in the OpenCV shim))
    endif
    ORT_HOME ?= $(HOME)/.local/share/vcam/onnxruntime
    ifeq ($(wildcard $(ORT_HOME)/include/onnxruntime_c_api.h),)
        ifeq ($(filter clean uninstall,$(MAKECMDGOALS)),)
            $(error ORT=1 but $(ORT_HOME)/include/onnxruntime_c_api.h not found. Run 'vcam-setup --ort' first, or set ORT_HOME=<dir>)
        endif
    endif
    CFLAGS  += -DVCAM_HAVE_ORT=1 -isystem $(ORT_HOME)/include
    LDFLAGS += -L$(ORT_HOME)/lib -lonnxruntime -Wl,-rpath,$(ORT_HOME)/lib
    $(info vcam: ONNX Runtime backend enabled — $(ORT_HOME))
else
    CFLAGS  += -DVCAM_HAVE_ORT=0
endif

# ── Install paths (overridable) ─────────────────────────────────────
#   make install                       (default: /usr/local)
#   make install PREFIX=$HOME/.local   (user-local)
#   make install DESTDIR=/tmp/stage    (staged install for packaging)
PREFIX  ?= /usr/local
BINDIR  ?= $(PREFIX)/bin
DOCDIR  ?= $(PREFIX)/share/doc/vcam
DATADIR ?= $(PREFIX)/share/vcam

SRCS := src/main.c src/frame.c src/capture.c src/output.c \
        src/overlay.c src/registry.c src/ipc.c src/filter.c \
        src/record.c src/ort_rvm.c
OBJS := $(SRCS:.c=.o) $(CV_OBJ)
BIN  := vcam
TEST := test_screen

# Build-config stamp: encodes the OPENCV setting the binary was linked
# with. Toggling OPENCV=0/1 changes the stamp name, which forces a
# relink — otherwise a stale vcam of the previous flavor could survive
# (both opencv_fx.o and opencv_stub.o may exist from earlier builds,
# and the binary would look up-to-date). This also makes
# 'make install' after 'make OPENCV=1' fail-safe: without the stamp,
# plain 'make install' (OPENCV defaulting to 0) would silently relink
# and install the stub flavor over your OpenCV build.
CONFIG_STAMP := .built-opencv-$(OPENCV)-ort-$(ORT)

.PHONY: all clean install uninstall

all: $(BIN) $(TEST)

$(CONFIG_STAMP):
	@rm -f .built-opencv-*
ifeq ($(ORT),1)
	@printf '%s\n' '$(ORT_HOME)' > $@
else
	@touch $@
endif

$(BIN): $(OBJS) $(CONFIG_STAMP)
	$(LINK) -o $@ $(OBJS) $(LDFLAGS)
ifeq ($(OPENCV),1)
ifeq ($(ORT),1)
	@echo "vcam: linked WITH OpenCV filters ($(OPENCV_PC) $(CV_VER)) + ONNX Runtime backend"
else
	@echo "vcam: linked WITH OpenCV filters ($(OPENCV_PC) $(CV_VER))"
endif
else
	@echo "vcam: linked WITHOUT OpenCV filters (build with 'make OPENCV=1' to enable)"
endif

$(TEST): src/test_screen.c
	$(CC) -std=c23 -O2 -Wall -D_POSIX_C_SOURCE=200809L -D_DEFAULT_SOURCE \
	    -o $@ src/test_screen.c

# C objects depend on both headers so an ABI change triggers a rebuild.
src/%.o: src/%.c include/vcam.h include/vcam_cv.h
	$(CC) $(CFLAGS) -c -o $@ $<

# The OpenCV shim (only compiled when OPENCV=1).
src/opencv_fx.o: src/opencv_fx.cpp include/vcam_cv.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

install: $(BIN)
ifeq ($(OPENCV),1)
ifeq ($(ORT),1)
	@echo "Installing vcam WITH OpenCV filters ($(OPENCV_PC) $(CV_VER)) + ONNX Runtime backend."
else
	@echo "Installing vcam WITH OpenCV filters ($(OPENCV_PC) $(CV_VER))."
endif
else
	@echo "Installing vcam WITHOUT OpenCV filters."
	@echo "  (If you built with 'make OPENCV=1', install with 'sudo make install OPENCV=1'"
	@echo "   or this step would relink and install the non-OpenCV flavor.)"
endif
	install -d $(DESTDIR)$(BINDIR)
	install -m 755 $(BIN)               $(DESTDIR)$(BINDIR)/
	install -m 755 scripts/vcam-ctl     $(DESTDIR)$(BINDIR)/
	install -m 755 scripts/vcam-setup   $(DESTDIR)$(BINDIR)/
	install -m 755 scripts/vcam-tune    $(DESTDIR)$(BINDIR)/
	install -d $(DESTDIR)$(DOCDIR)
	install -m 644 NOTICE               $(DESTDIR)$(DOCDIR)/
	install -m 644 README.md            $(DESTDIR)$(DOCDIR)/
	install -m 644 INSTALL.md           $(DESTDIR)$(DOCDIR)/
	install -m 644 EXPRESSIONS.md       $(DESTDIR)$(DOCDIR)/
	@echo
	@echo "Installed to $(DESTDIR)$(BINDIR)/  (docs in $(DESTDIR)$(DOCDIR)/)"
	@echo "Make sure that's on your PATH."
ifeq ($(OPENCV),1)
	@echo "OpenCV filters are built in. Fetch the models with:  vcam-setup --models"
endif

uninstall:
	rm -f $(DESTDIR)$(BINDIR)/$(BIN)
	rm -f $(DESTDIR)$(BINDIR)/vcam-ctl
	rm -f $(DESTDIR)$(BINDIR)/vcam-setup
	rm -f $(DESTDIR)$(BINDIR)/vcam-tune
	rm -rf $(DESTDIR)$(DOCDIR)

clean:
	rm -f $(SRCS:.c=.o) src/opencv_fx.o src/opencv_stub.o $(BIN) $(TEST) .built-opencv-*
