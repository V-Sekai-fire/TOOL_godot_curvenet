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

.PHONY: all test bench gpu clean
all: $(TEST_BINS)

# ----------------------------------------------------------------
# Standalone GPU compute test — Vulkan + MoltenVK on macOS, no Godot.
# Compiles src/curvenet/shaders/spmv.comp to SPIR-V via glslangValidator
# and dispatches it through a thin Vulkan wrapper, comparing the result
# to curvenet::sparse::spmv on a few SPD systems.
# ----------------------------------------------------------------

GLSLANG ?= $(shell command -v glslangValidator 2>/dev/null)

ifeq ($(UNAME_S),Darwin)
  VULKAN_PREFIX  ?= $(shell brew --prefix vulkan-loader 2>/dev/null)
  VULKAN_HEADERS ?= $(shell brew --prefix vulkan-headers 2>/dev/null)/include
  MOLTENVK_PREFIX ?= $(shell brew --prefix molten-vk 2>/dev/null)
  GPU_CXXFLAGS    := -std=c++14 -O2 -g -Wall -Wno-missing-braces \
                     -I$(SRC) -I$(VULKAN_HEADERS) -stdlib=libc++ -MMD -MP
  GPU_LDFLAGS     := -L$(VULKAN_PREFIX)/lib -lvulkan \
                     -Wl,-rpath,$(VULKAN_PREFIX)/lib
  GPU_RUN_ENV     := DYLD_FALLBACK_LIBRARY_PATH=$(MOLTENVK_PREFIX)/lib \
                     VK_ICD_FILENAMES=$(MOLTENVK_PREFIX)/etc/vulkan/icd.d/MoltenVK_icd.json
else
  GPU_CXXFLAGS    := -std=c++14 -O2 -g -Wall -Wno-missing-braces -I$(SRC) -MMD -MP
  GPU_LDFLAGS     := -lvulkan
  GPU_RUN_ENV     :=
endif

bin/%.spv: $(SRC)/curvenet/shaders/%.comp | bin
ifeq ($(GLSLANG),)
	$(error glslangValidator not found. Install with: brew install glslang)
endif
	$(GLSLANG) -V $< -o $@

bin/gpu_spmv_test: gpu_spmv_test.cpp gpu_compute_helpers.h | bin
	$(CXX) $(GPU_CXXFLAGS) -o $@ $< $(GPU_LDFLAGS)

bin/gpu_dot_test: gpu_dot_test.cpp gpu_compute_helpers.h | bin
	$(CXX) $(GPU_CXXFLAGS) -o $@ $< $(GPU_LDFLAGS)

bin/gpu_vec_ops_test: gpu_vec_ops_test.cpp gpu_compute_helpers.h | bin
	$(CXX) $(GPU_CXXFLAGS) -o $@ $< $(GPU_LDFLAGS)

bin/gpu_cg_test: gpu_cg_test.cpp gpu_compute_helpers.h gpu_cg_solver.h | bin
	$(CXX) $(GPU_CXXFLAGS) -o $@ $< $(GPU_LDFLAGS)

bin/gpu_cg_bench: gpu_cg_bench.cpp gpu_compute_helpers.h gpu_cg_solver.h | bin
	$(CXX) $(GPU_CXXFLAGS) -O3 -DNDEBUG -o $@ $< $(GPU_LDFLAGS)

bin/gpu_multi_rhs_test: gpu_multi_rhs_test.cpp gpu_compute_helpers.h | bin
	$(CXX) $(GPU_CXXFLAGS) -o $@ $< $(GPU_LDFLAGS)

gpu_bench: bin/gpu_cg_bench bin/spmv.spv bin/dot_reduce.spv bin/axpy.spv bin/jacobi.spv bin/saxpby.spv
	@$(GPU_RUN_ENV) bin/gpu_cg_bench

gpu: bin/gpu_spmv_test bin/spmv.spv \
     bin/gpu_dot_test bin/dot_reduce.spv \
     bin/gpu_vec_ops_test bin/axpy.spv bin/jacobi.spv bin/saxpby.spv \
     bin/gpu_cg_test \
     bin/gpu_multi_rhs_test bin/spmv_multi.spv
	@echo ">> spmv"
	@$(GPU_RUN_ENV) bin/gpu_spmv_test bin/spmv.spv
	@echo ""
	@echo ">> dot_reduce (df32)"
	@$(GPU_RUN_ENV) bin/gpu_dot_test bin/dot_reduce.spv
	@echo ""
	@echo ">> axpy / jacobi / saxpby"
	@$(GPU_RUN_ENV) bin/gpu_vec_ops_test
	@echo ""
	@echo ">> end-to-end GPU CG"
	@$(GPU_RUN_ENV) bin/gpu_cg_test
	@echo ""
	@echo ">> multi-RHS kernels"
	@$(GPU_RUN_ENV) bin/gpu_multi_rhs_test

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

-include bin/gpu_spmv_test.d
