# Copyright (c) 2026 K. S. Ernest (iFire) Lee.
# SPDX-License-Identifier: MIT
#
# Pure C++ test runner for the curvenet math library — does NOT depend on godot-cpp.
# Build a static librapidcheck.a once, then iterate quickly on test_*.cpp files.

PROJECT_ROOT := $(abspath ..)
RC := $(PROJECT_ROOT)/thirdparty/rapidcheck
LEMON := $(PROJECT_ROOT)/thirdparty/lemon
SRC := $(PROJECT_ROOT)/src

CXX ?= c++
CXXFLAGS := -std=c++14 -O0 -g -Wall -Wno-missing-braces -Wno-unused-command-line-argument \
            -I$(RC)/include -I$(LEMON) -I$(SRC) -MMD -MP
LDLIBS := -lpthread

CURVENET_OBJ := $(SRC)/curvenet/tris_to_quads.o $(SRC)/curvenet/curvenet.o

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  CXXFLAGS += -stdlib=libc++
endif

RC_SRC := $(shell find $(RC)/src -name "*.cpp")
RC_OBJ := $(RC_SRC:.cpp=.o)
LIBRC := librapidcheck.a

TESTS := test_bezier test_profile_curve test_coons_patch test_ngon_patch test_tris_to_quads test_curvenet test_bilinear_inverse test_binding test_pipeline test_mean_value test_halfedge test_polygon_laplacian test_dense_linalg test_scaled_frames
TEST_BINS := $(addprefix bin/,$(TESTS))

.PHONY: all test clean
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

clean:
	rm -rf bin $(LIBRC)
	find $(RC)/src -name "*.o" -delete
	find $(SRC)/curvenet -name "*.o" -delete 2>/dev/null || true
