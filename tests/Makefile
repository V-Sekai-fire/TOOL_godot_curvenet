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

TESTS := test_halfedge test_polygon_laplacian test_dense_linalg test_scaled_frames test_cut_algorithm test_deform_solve test_halfedge_builder test_curvenet_builder test_surface_projection test_chebyshev_accel test_two_level_schwarz test_multi_level_schwarz test_heavy_edge_matching test_kernel_projection test_incomplete_cholesky test_deformer_pipeline_identity
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

bin/gpu_cg_multi_test: gpu_cg_multi_test.cpp gpu_compute_helpers.h gpu_cg_solver.h gpu_cg_multi_solver.h | bin
	$(CXX) $(GPU_CXXFLAGS) -o $@ $< $(GPU_LDFLAGS)

gpu_bench: bin/gpu_cg_bench bin/spmv.spv bin/dot_reduce.spv bin/axpy.spv bin/jacobi.spv bin/saxpby.spv
	@$(GPU_RUN_ENV) bin/gpu_cg_bench

bin/gpu_cg_multi_bench: gpu_cg_multi_bench.cpp gpu_compute_helpers.h gpu_cg_solver.h gpu_cg_multi_solver.h | bin
	$(CXX) $(GPU_CXXFLAGS) -O3 -DNDEBUG -o $@ $< $(GPU_LDFLAGS)

gpu_multi_bench: bin/gpu_cg_multi_bench \
                  bin/spmv.spv bin/dot_reduce.spv bin/axpy.spv bin/jacobi.spv bin/saxpby.spv \
                  bin/spmv_multi.spv bin/dot_reduce_multi.spv \
                  bin/axpy_multi.spv bin/jacobi_multi.spv bin/saxpby_multi.spv
	@$(GPU_RUN_ENV) bin/gpu_cg_multi_bench

gpu: bin/gpu_spmv_test bin/spmv.spv \
     bin/gpu_dot_test bin/dot_reduce.spv \
     bin/gpu_vec_ops_test bin/axpy.spv bin/jacobi.spv bin/saxpby.spv \
     bin/gpu_cg_test \
     bin/gpu_multi_rhs_test bin/spmv_multi.spv \
     bin/axpy_multi.spv bin/jacobi_multi.spv bin/saxpby_multi.spv bin/dot_reduce_multi.spv \
     bin/gpu_cg_multi_test
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
	@echo ""
	@echo ">> multi-RHS GPU CG"
	@$(GPU_RUN_ENV) bin/gpu_cg_multi_test

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

bin/bench_deform: bench_deform.cpp mire_body_data.h | bin
	$(CXX) $(BENCH_CXXFLAGS) -o $@ $<

bench: bin/bench_deform
	@bin/bench_deform

bin/bench_deform_5k_icc: bench_deform_5k_icc.cpp mire_body_data.h | bin
	$(CXX) $(BENCH_CXXFLAGS) -o $@ $<

bench_5k_icc: bin/bench_deform_5k_icc
	@bin/bench_deform_5k_icc

bin/diag_warm_start_drift_5k: diag_warm_start_drift_5k.cpp mire_body_data.h | bin
	$(CXX) $(BENCH_CXXFLAGS) -o $@ $<

diag_warm_start_drift_5k: bin/diag_warm_start_drift_5k
	@bin/diag_warm_start_drift_5k

bin/bench_deform_70k_icc: bench_deform_70k_icc.cpp mire_body_70k_data.h | bin
	$(CXX) $(BENCH_CXXFLAGS) -o $@ $<

bench_70k_icc: bin/bench_deform_70k_icc
	@bin/bench_deform_70k_icc

bin/bench_deform_70k: bench_deform_70k.cpp mire_body_70k_data.h | bin
	$(CXX) $(BENCH_CXXFLAGS) -o $@ $<

bench_70k: bin/bench_deform_70k
	@bin/bench_deform_70k

bin/bench_deform_70k_gpu: bench_deform_70k_gpu.cpp mire_body_70k_data.h gpu_compute_helpers.h gpu_cg_solver.h | bin
	$(CXX) $(GPU_CXXFLAGS) -O3 -DNDEBUG -o $@ $< $(GPU_LDFLAGS)

bench_70k_gpu: bin/bench_deform_70k_gpu \
                bin/spmv.spv bin/dot_reduce.spv bin/axpy.spv bin/jacobi.spv bin/saxpby.spv
	@$(GPU_RUN_ENV) bin/bench_deform_70k_gpu

bin/bench_deform_5k_gpu: bench_deform_5k_gpu.cpp mire_body_data.h gpu_compute_helpers.h gpu_cg_solver.h | bin
	$(CXX) $(GPU_CXXFLAGS) -O3 -DNDEBUG -o $@ $< $(GPU_LDFLAGS)

bench_5k_gpu: bin/bench_deform_5k_gpu \
               bin/spmv.spv bin/dot_reduce.spv bin/axpy.spv bin/jacobi.spv bin/saxpby.spv
	@$(GPU_RUN_ENV) bin/bench_deform_5k_gpu

bin/diag_real_mesh: diag_real_mesh.cpp mire_body_data.h | bin
	$(CXX) $(BENCH_CXXFLAGS) -o $@ $<

diag: bin/diag_real_mesh
	@bin/diag_real_mesh

# meshoptimizer — vendored under thirdparty/meshoptimizer. Compile its
# .cpp sources directly into the meshlet diagnostic; no static lib step,
# the whole library is small.
MESHOPT_DIR := $(PROJECT_ROOT)/thirdparty/meshoptimizer
MESHOPT_SRC := $(wildcard $(MESHOPT_DIR)/src/*.cpp)
MESHOPT_OBJ := $(MESHOPT_SRC:.cpp=.o)
MESHOPT_CXXFLAGS := -std=c++14 -O3 -DNDEBUG -Wall -Wno-missing-braces

ifeq ($(UNAME_S),Darwin)
  MESHOPT_CXXFLAGS += -stdlib=libc++
endif

$(MESHOPT_DIR)/src/%.o: $(MESHOPT_DIR)/src/%.cpp
	$(CXX) $(MESHOPT_CXXFLAGS) -c $< -o $@

bin/diag_meshlet: diag_meshlet.cpp mire_body_data.h $(MESHOPT_OBJ) | bin
	$(CXX) $(BENCH_CXXFLAGS) -I$(MESHOPT_DIR)/src -o $@ $< $(MESHOPT_OBJ)

diag_meshlet: bin/diag_meshlet
	@bin/diag_meshlet

bin/diag_meshlet_lhsm: diag_meshlet_lhsm.cpp mire_body_data.h $(MESHOPT_OBJ) | bin
	$(CXX) $(BENCH_CXXFLAGS) -I$(MESHOPT_DIR)/src -o $@ $< $(MESHOPT_OBJ)

diag_meshlet_lhsm: bin/diag_meshlet_lhsm
	@bin/diag_meshlet_lhsm

bin/diag_meshlet_solve: diag_meshlet_solve.cpp mire_body_data.h $(MESHOPT_OBJ) | bin
	$(CXX) $(BENCH_CXXFLAGS) -I$(MESHOPT_DIR)/src -o $@ $< $(MESHOPT_OBJ)

diag_meshlet_solve: bin/diag_meshlet_solve
	@bin/diag_meshlet_solve

bin/diag_meshlet_schwarz: diag_meshlet_schwarz.cpp mire_body_data.h $(MESHOPT_OBJ) | bin
	$(CXX) $(BENCH_CXXFLAGS) -I$(MESHOPT_DIR)/src -o $@ $< $(MESHOPT_OBJ)

diag_meshlet_schwarz: bin/diag_meshlet_schwarz
	@bin/diag_meshlet_schwarz

bin/diag_meshlet_schwarz_overlap: diag_meshlet_schwarz_overlap.cpp mire_body_data.h $(MESHOPT_OBJ) | bin
	$(CXX) $(BENCH_CXXFLAGS) -I$(MESHOPT_DIR)/src -o $@ $< $(MESHOPT_OBJ)

diag_meshlet_schwarz_overlap: bin/diag_meshlet_schwarz_overlap
	@bin/diag_meshlet_schwarz_overlap

bin/diag_meshlet_pcg: diag_meshlet_pcg.cpp mire_body_data.h $(MESHOPT_OBJ) | bin
	$(CXX) $(BENCH_CXXFLAGS) -I$(MESHOPT_DIR)/src -o $@ $< $(MESHOPT_OBJ)

diag_meshlet_pcg: bin/diag_meshlet_pcg
	@bin/diag_meshlet_pcg

bin/diag_meshlet_pcg_chol: diag_meshlet_pcg_chol.cpp mire_body_data.h $(MESHOPT_OBJ) | bin
	$(CXX) $(BENCH_CXXFLAGS) -I$(MESHOPT_DIR)/src -o $@ $< $(MESHOPT_OBJ)

diag_meshlet_pcg_chol: bin/diag_meshlet_pcg_chol
	@bin/diag_meshlet_pcg_chol

bin/diag_meshlet_pcg_chol_70k: diag_meshlet_pcg_chol_70k.cpp mire_body_70k_data.h $(MESHOPT_OBJ) | bin
	$(CXX) $(BENCH_CXXFLAGS) -I$(MESHOPT_DIR)/src -o $@ $< $(MESHOPT_OBJ)

diag_meshlet_pcg_chol_70k: bin/diag_meshlet_pcg_chol_70k
	@bin/diag_meshlet_pcg_chol_70k

bin/diag_meshlet_local_cg: diag_meshlet_local_cg.cpp mire_body_70k_data.h $(MESHOPT_OBJ) | bin
	$(CXX) $(BENCH_CXXFLAGS) -I$(MESHOPT_DIR)/src -o $@ $< $(MESHOPT_OBJ)

diag_meshlet_local_cg: bin/diag_meshlet_local_cg
	@bin/diag_meshlet_local_cg

bin/diag_meshlet_chebyshev_5k: diag_meshlet_chebyshev_5k.cpp mire_body_data.h $(MESHOPT_OBJ) | bin
	$(CXX) $(BENCH_CXXFLAGS) -I$(MESHOPT_DIR)/src -o $@ $< $(MESHOPT_OBJ)

diag_meshlet_chebyshev_5k: bin/diag_meshlet_chebyshev_5k
	@bin/diag_meshlet_chebyshev_5k

bin/diag_two_level_schwarz_5k: diag_two_level_schwarz_5k.cpp mire_body_data.h $(MESHOPT_OBJ) | bin
	$(CXX) $(BENCH_CXXFLAGS) -I$(MESHOPT_DIR)/src -o $@ $< $(MESHOPT_OBJ)

diag_two_level_schwarz_5k: bin/diag_two_level_schwarz_5k
	@bin/diag_two_level_schwarz_5k

bin/diag_multi_level_schwarz_5k: diag_multi_level_schwarz_5k.cpp mire_body_data.h $(MESHOPT_OBJ) | bin
	$(CXX) $(BENCH_CXXFLAGS) -I$(MESHOPT_DIR)/src -o $@ $< $(MESHOPT_OBJ)

diag_multi_level_schwarz_5k: bin/diag_multi_level_schwarz_5k
	@bin/diag_multi_level_schwarz_5k

bin/diag_multi_level_schwarz_70k: diag_multi_level_schwarz_70k.cpp mire_body_70k_data.h $(MESHOPT_OBJ) | bin
	$(CXX) $(BENCH_CXXFLAGS) -I$(MESHOPT_DIR)/src -o $@ $< $(MESHOPT_OBJ)

diag_multi_level_schwarz_70k: bin/diag_multi_level_schwarz_70k
	@bin/diag_multi_level_schwarz_70k

bin/diag_70k_cg_baseline: diag_70k_cg_baseline.cpp mire_body_70k_data.h | bin
	$(CXX) $(BENCH_CXXFLAGS) -o $@ $<

diag_70k_cg_baseline: bin/diag_70k_cg_baseline
	@bin/diag_70k_cg_baseline

bin/diag_70k_icc_pcg: diag_70k_icc_pcg.cpp mire_body_70k_data.h | bin
	$(CXX) $(BENCH_CXXFLAGS) -o $@ $<

diag_70k_icc_pcg: bin/diag_70k_icc_pcg
	@bin/diag_70k_icc_pcg

bin/diag_5k_icc_pcg: diag_5k_icc_pcg.cpp mire_body_data.h | bin
	$(CXX) $(BENCH_CXXFLAGS) -o $@ $<

diag_5k_icc_pcg: bin/diag_5k_icc_pcg
	@bin/diag_5k_icc_pcg

clean:
	rm -rf bin $(LIBRC)
	find $(RC)/src -name "*.o" -delete
	find $(SRC)/curvenet -name "*.o" -delete 2>/dev/null || true

-include bin/gpu_spmv_test.d
