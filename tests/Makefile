# Copyright (c) 2026 K. S. Ernest (iFire) Lee.
# SPDX-License-Identifier: MIT
#
# Pure C++ test runner for the curvenet math library — does NOT depend on godot-cpp.
# Build a static librapidcheck.a once, then iterate quickly on test_*.cpp files.

PROJECT_ROOT := $(abspath ..)
RC := $(PROJECT_ROOT)/thirdparty/rapidcheck
SRC := $(PROJECT_ROOT)/src

CXX ?= c++
CXXFLAGS := -std=c++14 -O0 -g -Wall -Wno-missing-braces -Wno-unused-command-line-argument \
            -I$(RC)/include -I$(SRC) -MMD -MP
LDLIBS := -lpthread

# All curvenet runtime is header-only — no compiled .cpp objects to link.
CURVENET_OBJ :=

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  CXXFLAGS += -stdlib=libc++
endif

RC_SRC := $(shell find $(RC)/src -name "*.cpp")
RC_OBJ := $(RC_SRC:.cpp=.o)
LIBRC := librapidcheck.a

TESTS := test_halfedge test_polygon_laplacian test_dense_linalg test_scaled_frames test_cut_algorithm test_deform_solve test_halfedge_builder test_curvenet_builder test_surface_projection
TEST_BINS := $(addprefix bin/,$(TESTS))

BENCH_CXXFLAGS := -std=c++14 -O3 -DNDEBUG -Wall -Wno-missing-braces \
                  -I$(SRC) -MMD -MP
ifeq ($(UNAME_S),Darwin)
  BENCH_CXXFLAGS += -stdlib=libc++
endif

.PHONY: all test bench clean
all: $(TEST_BINS)

$(LIBRC): $(RC_OBJ)
	ar rcs $@ $(RC_OBJ)

# Pattern rule for rapidcheck objects (in-place next to source).
$(RC)/src/%.o: $(RC)/src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(SRC)/curvenet/%.o: $(SRC)/curvenet/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

bin/%: %.cpp $(LIBRC) $(CURVENET_OBJ) | bin
	$(CXX) $(CXXFLAGS) -o $@ $< $(CURVENET_OBJ) $(LIBRC) $(LDLIBS)

-include $(addprefix bin/,$(addsuffix .d,$(TESTS)))

bin:
	mkdir -p bin

test: $(TEST_BINS)
	@for t in $(TEST_BINS); do \
		echo ">> $$t"; \
		$$t || exit 1; \
	done

bin/bench_deform: bench_deform.cpp | bin
	$(CXX) $(BENCH_CXXFLAGS) -o $@ $<

bench: bin/bench_deform
	@bin/bench_deform

clean:
	rm -rf bin $(LIBRC)
	find $(RC)/src -name "*.o" -delete
	find $(SRC)/curvenet -name "*.o" -delete 2>/dev/null || true
