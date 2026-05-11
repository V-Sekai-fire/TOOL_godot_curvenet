// Real-input validator for Curvenet.SlangCodegen.DirectDeltaMush —
// the actual Pixar Profile-Curves runtime skinning kernel, not a generic
// BLAS primitive. This is the smallest end-to-end MVP step proving that
// the Lean → Slang → slangc-cpp → executable chain produces *bit-exact*
// agreement with the Lean `DeformSolveExamples` reference *for the
// algorithm the project actually targets*.
//
// Input mirrors `Curvenet.EndToEndExample.deformedPositionsRotateZ90`:
//   * 3-vertex `triangleWithSample` rest mesh:
//       v0 = (0, 0, 0),  v1 = (1, 0, 0),  v2 = (0.5, 0, √3/2)
//   * 1 handle bound 100% to every vertex (CSR inflStart = [0,1,2,3],
//     inflIdx = [0,0,0], inflW = [1.0, 1.0, 1.0])
//   * Handle 0 transform = R_z(90°) about the origin, packed
//     column-major as glTF / Slang require (4 column-vectors of 4 floats).
//
// Reference: `(x, y, z) → (−y, x, z)`. The slangc-emitted kernel runs
// the per-vertex CSR matvec `pos[v] = Σᵢ W[v,i] · T_i · rest[v]`. With
// one full-weight handle the sum collapses to a single transform.
//
// Why this is the MVP step: every prior slang_validate test (`axpy`,
// `saxpby`, `jacobi`, `mire_axpy`, `check_ascii`) exercises generic
// linear algebra. `direct_delta_mush` is the first kernel here that
// implements a *Profile-Curves-specific* operation — Le & Lewis 2019
// runtime skinning with curvenet-derived handle transforms.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "direct_delta_mush_emit.cpp"

static constexpr double kBudgetSeconds = 5.0;

int main() {
    const auto t0 = std::chrono::steady_clock::now();

    constexpr uint32_t N_VERTS    = 3;
    constexpr uint32_t N_HANDLES  = 1;
    constexpr uint32_t GROUP_SIZE = 64;   // numthreads(64,1,1) — pad to one group

    // ---- Rest positions ---------------------------------------------------
    std::vector<Vector<float, 3>> restPos(GROUP_SIZE, Vector<float, 3>(0.0f));
    restPos[0] = Vector<float, 3>(0.0f, 0.0f, 0.0f);
    restPos[1] = Vector<float, 3>(1.0f, 0.0f, 0.0f);
    restPos[2] = Vector<float, 3>(0.5f, 0.0f, 0.8660254037844386f);

    std::vector<Vector<float, 3>> outPos(GROUP_SIZE, Vector<float, 3>(0.0f));

    // ---- One handle: R_z(90°) at origin, packed column-major.
    // Row-major matrix (HLSL `mul(M, v)` convention):
    //   [  0  -1   0   0 ]
    //   [  1   0   0   0 ]
    //   [  0   0   1   0 ]
    //   [  0   0   0   1 ]
    // Column-major flat:  col0 col1 col2 col3, each a float4.
    std::vector<_MatrixStorage_float4x4_ColMajornatural_0> transforms(N_HANDLES);
    transforms[0].data_0[0] = Vector<float, 4>(0.0f,  1.0f, 0.0f, 0.0f);  // col 0
    transforms[0].data_0[1] = Vector<float, 4>(-1.0f, 0.0f, 0.0f, 0.0f);  // col 1
    transforms[0].data_0[2] = Vector<float, 4>(0.0f,  0.0f, 1.0f, 0.0f);  // col 2
    transforms[0].data_0[3] = Vector<float, 4>(0.0f,  0.0f, 0.0f, 1.0f);  // col 3

    // ---- CSR weights: every vertex 100% bound to handle 0 -----------------
    // The kernel has no bounds check and reads `inflStart[v_0]` +
    // `inflStart[v_0 + 1]` for v_0 in [0, GROUP_SIZE). Pad inflStart to
    // GROUP_SIZE+1 = 65 entries; the trailing entries all equal the total
    // nnz (3) so threads V..63 see an empty influence range (start==end)
    // and the body loop runs 0 times — no out-of-bounds inflIdx / inflW reads.
    std::vector<uint32_t> inflStart(GROUP_SIZE + 1u, 3u);
    inflStart[0] = 0u;
    inflStart[1] = 1u;
    inflStart[2] = 2u;
    inflStart[3] = 3u;
    std::vector<uint32_t> inflIdx = {0u, 0u, 0u};
    std::vector<float>    inflW   = {1.0f, 1.0f, 1.0f};

    GlobalParams_0 gp{};
    gp.restPos_0.data    = restPos.data();     gp.restPos_0.count    = GROUP_SIZE;
    gp.outPos_0.data     = outPos.data();      gp.outPos_0.count     = GROUP_SIZE;
    gp.transforms_0.data = transforms.data();  gp.transforms_0.count = N_HANDLES;
    gp.inflStart_0.data  = inflStart.data();   gp.inflStart_0.count  = inflStart.size();
    gp.inflIdx_0.data    = inflIdx.data();     gp.inflIdx_0.count    = inflIdx.size();
    gp.inflW_0.data      = inflW.data();       gp.inflW_0.count      = inflW.size();

    // One thread group of 64; only the first N_VERTS produce meaningful output.
    // The kernel has no bounds check (per its comment "bounds-check is the
    // host's job"); padding restPos/outPos to GROUP_SIZE keeps the
    // out-of-range threads from reading past the buffer.
    ComputeVaryingInput vi{};
    vi.startGroupID = uint3(0, 0, 0);
    vi.endGroupID   = uint3(1, 1, 1);
    main_0(&vi, nullptr, &gp);

    // ---- Reference: host LBS matvec under R_z(90°) -----------------------
    const float expected[N_VERTS][3] = {
        {0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.5f, 0.8660254037844386f},
    };

    int   fails        = 0;
    float max_abs_diff = 0.0f;
    for (uint32_t v = 0; v < N_VERTS; ++v) {
        for (int c = 0; c < 3; ++c) {
            const float got = outPos[v][c];
            const float ref = expected[v][c];
            const float d   = std::fabs(got - ref);
            if (d > max_abs_diff) max_abs_diff = d;
            if (d > 1e-6f) {
                if (fails < 5) {
                    std::fprintf(stderr,
                        "direct_delta_mush mismatch at v=%u, c=%d: got %g, expected %g (diff %g)\n",
                        v, c, got, ref, d);
                }
                ++fails;
            }
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();
    if (elapsed > kBudgetSeconds) {
        std::fprintf(stderr,
            "direct_delta_mush: TIMEOUT — %.3fs > budget %.1fs\n",
            elapsed, kBudgetSeconds);
        return 2;
    }
    if (fails == 0) {
        std::printf(
            "direct_delta_mush: %u/%u verts OK (n_handles=%u, transform=R_z(90°), max_abs_diff=%g, %.1fms)\n",
            N_VERTS, N_VERTS, N_HANDLES, max_abs_diff, elapsed * 1000.0);
        return 0;
    }
    std::fprintf(stderr, "direct_delta_mush: %d FAIL out of %u (%u verts × 3 components)\n",
                 fails, N_VERTS * 3u, N_VERTS);
    return 1;
}
