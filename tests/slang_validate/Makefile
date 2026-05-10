# Real-input validators for the Curvenet.SlangCodegen.* kernels.
#
# Each kernel is converted by `slangc -target cpp` to a CPU function
# `main_0(varyingInput, params, globals)`, then compiled together
# with a hand-written test harness that:
#   1. populates the kernel's input buffers with known values,
#   2. invokes the kernel as a single dispatch,
#   3. asserts the output matches an independent reference.
#
# `make test` runs every harness; non-zero exit on any failure.

REPO_ROOT  := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
SLANGC     := $(REPO_ROOT)/bin/slangc
SLANG_DIR  := $(REPO_ROOT)/bin/.slang
EMITTED    := $(CURDIR)/build
LEAN_DIR   := $(REPO_ROOT)/lean

CXX        ?= clang++
CXXFLAGS   ?= -std=c++17 -O2 -Wall -Wextra
INCLUDES   := -I$(SLANG_DIR)/include -I$(EMITTED)

KERNELS    := axpy saxpby jacobi
SHADERS    := $(addprefix $(EMITTED)/,$(addsuffix _emit.cpp,$(KERNELS)))
BINS       := $(addprefix $(EMITTED)/,$(addsuffix _test,$(KERNELS)))

.PHONY: test clean emit-shaders

test: $(BINS)
	@fails=0; \
	for b in $(BINS); do \
	  echo "=== $$b"; $$b || fails=$$(( fails + 1 )); \
	done; \
	if [ $$fails -ne 0 ]; then echo "FAILED: $$fails"; exit 1; fi; \
	echo "all kernels OK"

# Make sure the .slang files exist before we ask slangc for cpp output.
emit-shaders:
	@cd $(LEAN_DIR) && lake exe emit_shaders /tmp/emitted_shaders >/dev/null

# slang → cpp source.
$(EMITTED)/%_emit.cpp: emit-shaders
	@mkdir -p $(EMITTED)
	$(SLANGC) -target cpp -stage compute -entry main \
	  -o $@ /tmp/emitted_shaders/$*.slang

# test harness + kernel cpp → executable.
$(EMITTED)/%_test: %_test.cpp $(EMITTED)/%_emit.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $<

clean:
	rm -rf $(EMITTED)
