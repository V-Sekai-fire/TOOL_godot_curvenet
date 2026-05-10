// Real-input validator for Curvenet.SlangCodegen.Axpy at production scale.
// Uses the Mire 70k-vertex mesh (auto-generated from MireQuest.blend; see
// tests/mire_body_70k_data.h) as input data, exercising the slangc-emitted
// axpy kernel across multiple thread groups instead of the single-group
// n=4 path that axpy_test.cpp covers.
//
// Compute: y[i] = fma(alpha, x[i], y[i]) for i < N where
//   N      = mire_body_70k::n_verts * 3 = 244839 floats
//   alpha  = 1.5
//   x[i]   = mire_body_70k::positions[i]            (the mesh positions)
//   y[i]   = mire_body_70k::positions[i] * 0.25     (a scaled copy)
// Reference: host computes the same fma on the same inputs and the test
// asserts pointwise equality within an fp32-fma tolerance.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "axpy_emit.cpp"
#include "../mire_body_70k_data.h"

// Hard wallclock cap. axpy on 245k floats is microseconds on any CPU
// from the last decade; anything above 5 s means the kernel hung or
// the C++ optimizer chose something pathological. Fail loud.
static constexpr double kBudgetSeconds = 5.0;

int main() {
    const auto t0 = std::chrono::steady_clock::now();
    const uint32_t N = static_cast<uint32_t>(mire_body_70k::n_verts) * 3u;
    const float alpha = 1.5f;

    std::vector<float> x(N);
    std::vector<float> y(N);
    std::vector<float> y_ref(N);
    for (uint32_t i = 0; i < N; ++i) {
        x[i] = mire_body_70k::positions[i];
        y[i] = mire_body_70k::positions[i] * 0.25f;
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
        // the smallest non-zero ulp at this magnitude (Mire positions
        // peak around |2|, so ulp ~ 2.4e-7).
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
            "mire_axpy: %u/%u OK (alpha=%g, n_verts=%d, max_abs_diff=%g, %u groups, %.1fms)\n",
            N, N, alpha, mire_body_70k::n_verts, max_abs_diff, groups,
            elapsed * 1000.0);
        return 0;
    }
    std::fprintf(stderr, "mire_axpy: %d FAIL out of %u\n", fails, N);
    return 1;
}
