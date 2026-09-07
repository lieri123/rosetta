# Rosetta -- handwritten pages into text you can post.
#
# Five languages and three processes is more infrastructure than most projects
# need, so the build tries to make the whole thing one command. `make` builds
# everything, `make test` runs every suite, `make demo` proves the pipeline
# works end to end without an API key.

.DEFAULT_GOAL := all

PYTHON  ?= python3
DB      ?= data/rosetta.db
DATA    ?= data
ADDR    ?= 127.0.0.1:8800
RESCORER_PORT ?= 8801

PREPROCESS_BIN := preprocess/build/rosetta-preprocess
SERVICE_BIN    := service/bin/rosetta

.PHONY: all preprocess service web deps test test-preprocess test-service test-rescorer \
        run run-service run-rescorer demo eval seed clean distclean fmt

all: preprocess service web

# --- build ------------------------------------------------------------------

preprocess:
	$(MAKE) -C preprocess all

service:
	cd service && go build -o bin/rosetta ./cmd/rosetta

web: web/node_modules
	cd web && npm run build

web/node_modules: web/package.json
	cd web && npm install --no-audit --no-fund
	@touch web/node_modules

deps: web/node_modules
	cd service && go mod download

# --- test -------------------------------------------------------------------

test: test-preprocess test-rescorer test-service

test-preprocess:
	@echo "== C preprocessing =="
	$(MAKE) -C preprocess test

test-rescorer:
	@echo "== Python rescorer =="
	cd rescorer && $(PYTHON) -m unittest discover -s tests -q

test-service:
	@echo "== Go service =="
	cd service && go test ./...

fmt:
	cd service && gofmt -w .
	cd web && npx tsc --noEmit

# --- run --------------------------------------------------------------------

# Two processes. Run them in two terminals, or use deploy/docker-compose.yml.
run:
	@echo "start these in separate terminals:"
	@echo "  make run-rescorer"
	@echo "  make run-service"

run-rescorer:
	cd rescorer && $(PYTHON) -m rosetta_rescorer --db ../$(DB) serve --port $(RESCORER_PORT)

run-service: $(SERVICE_BIN)
	$(SERVICE_BIN) -db $(DB) -data $(DATA) -static web/dist \
		-preprocess $(PREPROCESS_BIN) -addr $(ADDR)

# Seed the language prior from writing you already have, so a fresh install has
# some idea of your vocabulary before it has seen a single correction.
seed:
	cd rescorer && $(PYTHON) -m rosetta_rescorer --db ../$(DB) ingest data/sample-notes.txt

# --- demonstrate ------------------------------------------------------------

# Synthesise a page, run it through the whole pipeline, correct it, and run the
# same page again to show the error rate fall. No API key, no photograph.
demo: all
	./scripts/demo.sh

# Measure the rescorer against the raw recogniser on held-out lines.
eval:
	cd rescorer && $(PYTHON) -m rosetta_rescorer --db /tmp/rosetta-eval.db \
		eval --synthesize data/sample-notes.txt --prime-prior --train-fraction 0.7 --sweep

# --- clean ------------------------------------------------------------------

clean:
	$(MAKE) -C preprocess clean
	rm -rf service/bin web/dist
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

distclean: clean
	rm -rf web/node_modules $(DATA)
