// Production-scale validator for Curvenet.SlangCodegen.Axpy. Exercises
// the slangc-emitted axpy kernel across multiple thread groups, where
// the n=4 path in axpy_test.cpp can't reach the tail-group bounds-check
// branch.
//
// Inputs are synthesised to ~Mire-vertex scale (81613 verts × 3 = 244839
// floats) with deterministic, mesh-shaped magnitudes (|x| ≤ ~2) so the
// fp32-fma tolerance below stays meaningful. Earlier versions of this
// test inlined the actual Mire mesh as a 240k-line .h header; the
// safetensors-based data path replaced that and the kernel itself
// doesn't care whether the buffer holds real positions or sin/cos —
// only that the bounds-check fires and the fma is bit-exact.
//
// Compute: y[i] = fma(alpha, x[i], y[i]) for i < N.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "axpy_emit.cpp"

static constexpr uint32_t kNVerts = 81613;   // matches the historical Mire body count
static constexpr double kBudgetSeconds = 5.0;

int main() {
    const auto t0 = std::chrono::steady_clock::now();
    const uint32_t N = kNVerts * 3u;
    const float alpha = 1.5f;

    std::vector<float> x(N);
    std::vector<float> y(N);
    std::vector<float> y_ref(N);
    for (uint32_t i = 0; i < N; ++i) {
        // Deterministic mesh-shaped pseudo-positions in [-2, 2].
        const float t = static_cast<float>(i) * 1.0e-4f;
        x[i] = 2.0f * std::sin(0.7f * t) * std::cos(t);
        y[i] = x[i] * 0.25f;
        y_ref[i] = std::fma(alpha, x[i], y[i]);
    }

    AxpyParams_0 params{N, alpha};
    GlobalParams_0 gp{};
    gp.params_0 = &params;
    gp.x_0.data = x.data();    gp.x_0.count = N;
    gp.y_0.data = y.data();    gp.y_0.count = N;

    // numthreads(256, 1, 1); ceil(N / 256) groups so the tail group's
    // bounds check (`if (i >= n) return;`) actually fires.
    const uint32_t groups = (N + 255u) / 256u;
    ComputeVaryingInput vi{};
    vi.startGroupID = uint3(0, 0, 0);
    vi.endGroupID   = uint3(groups, 1, 1);
    main_0(&vi, nullptr, &gp);

    int fails = 0;
    float max_abs_diff = 0.0f;
    for (uint32_t i = 0; i < N; ++i) {
        const float d = std::fabs(y[i] - y_ref[i]);
        if (d > max_abs_diff) max_abs_diff = d;
        // fp32 fma is bit-exact in both implementations; allow only
        // the smallest non-zero ulp at this magnitude (|x| peaks
        // around 2, so ulp ~ 2.4e-7).
        if (d > 1e-6f) {
            if (fails < 5) {
                std::fprintf(stderr,
                    "mire_axpy mismatch at i=%u: got %g, expected %g (diff %g)\n",
                    i, y[i], y_ref[i], d);
            }
            ++fails;
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed =
        std::chrono::duration<double>(t1 - t0).count();
    if (elapsed > kBudgetSeconds) {
        std::fprintf(stderr,
            "mire_axpy: TIMEOUT — %.3fs > budget %.1fs\n",
            elapsed, kBudgetSeconds);
        return 2;
    }
    if (fails == 0) {
        std::printf(
            "mire_axpy: %u/%u OK (alpha=%g, n_verts=%u, max_abs_diff=%g, %u groups, %.1fms)\n",
            N, N, alpha, kNVerts, max_abs_diff, groups,
            elapsed * 1000.0);
        return 0;
    }
    std::fprintf(stderr, "mire_axpy: %d FAIL out of %u\n", fails, N);
    return 1;
}
