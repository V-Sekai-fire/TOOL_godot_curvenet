REPO := $(abspath $(CURDIR)/../..)

.PHONY: all gen test clean

all: test

# Generate the spec-coverage test vectors (Python safetensors lib → bytes).
gen:
	cd $(REPO) && python3 tests/safetensors/gen_vectors.py

# Build the Lean exe + run the Python ↔ Lean round-trip verifier.
test: gen
	cd $(REPO)/lean && lake build safetensors_roundtrip
	cd $(REPO) && python3 tests/safetensors/run_tests.py

clean:
	rm -rf $(CURDIR)/rt
