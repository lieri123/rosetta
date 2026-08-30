# rosetta-preprocess
#
# No dependencies beyond libc and libm: stb_image is vendored, and everything
# else is written against raw pixel buffers.

CC      ?= cc
CFLAGS  ?= -std=c11 -O2 -Wall -Wextra
CFLAGS  += -Isrc
LDLIBS   = -lm

BUILD   := build
LIB_SRC := src/img.c src/geom.c src/deskew.c src/threshold.c src/lines.c src/synth.c
LIB_OBJ := $(LIB_SRC:%.c=$(BUILD)/%.o)

BIN       := $(BUILD)/rosetta-preprocess
TEST_BIN  := $(BUILD)/test_preprocess
MKPAGE    := $(BUILD)/mkpage

.PHONY: all test demo clean

all: $(BIN) $(MKPAGE)

$(BIN): $(BUILD)/src/main.o $(LIB_OBJ)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(TEST_BIN): $(BUILD)/test/test_preprocess.o $(LIB_OBJ)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(MKPAGE): $(BUILD)/tools/mkpage.o $(LIB_OBJ)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c -o $@ $<

test: $(TEST_BIN)
	$(TEST_BIN)

# Synthesise a skewed, perspective-distorted, unevenly lit page and run the
# whole pipeline over it, leaving the intermediates in build/demo.
demo: all
	@mkdir -p $(BUILD)/demo
	$(MKPAGE) -o $(BUILD)/demo/input.png --skew 3.5 --perspective 0.10 \
		--shading 0.6 --strike 4 --noise 0.02
	$(BIN) $(BUILD)/demo/input.png -o $(BUILD)/demo/clean.png \
		--strikethrough --debug-dir $(BUILD)/demo --json $(BUILD)/demo/meta.json
	@echo "--- meta.json"
	@cat $(BUILD)/demo/meta.json

clean:
	rm -rf $(BUILD)
