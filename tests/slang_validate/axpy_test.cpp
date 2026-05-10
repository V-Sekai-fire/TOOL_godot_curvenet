// Real-input validator for Curvenet.SlangCodegen.Axpy.
// Compiled and linked with the slangc -target cpp output of axpy.slang.
// y[i] += alpha * x[i] for i in [0, n)
//
// Build via: make -C tests/slang_validate axpy_test && ./tests/slang_validate/axpy_test

#include <cmath>
#include <cstdint>
#include <cstdio>

// The emitted C++ kernel includes slang-cpp-prelude.h itself.
#include "axpy_emit.cpp"

int main() {
    constexpr uint32_t N = 4;
    AxpyParams_0 params{N, 2.0f};

    float x[N] = {1.0f, 2.0f, 3.0f, 4.0f};
    float y[N] = {10.0f, 20.0f, 30.0f, 40.0f};
    const float expected[N] = {12.0f, 24.0f, 36.0f, 48.0f};

    GlobalParams_0 gp{};
    gp.params_0 = &params;
    gp.x_0.data = x;
    gp.x_0.count = N;
    gp.y_0.data = y;
    gp.y_0.count = N;

    // numthreads(256, 1, 1) and N=4 → one group covers all four lanes.
    ComputeVaryingInput vi{};
    vi.startGroupID = uint3(0, 0, 0);
    vi.endGroupID = uint3(1, 1, 1);

    main_0(&vi, nullptr, &gp);

    int fails = 0;
    for (uint32_t i = 0; i < N; ++i) {
        if (std::fabs(y[i] - expected[i]) > 1e-6f) {
            std::fprintf(stderr,
                "axpy mismatch at i=%u: got %f, expected %f\n",
                i, y[i], expected[i]);
            ++fails;
        }
    }

    if (fails == 0) {
        std::printf("axpy: %u/%u OK (alpha=%g, n=%u)\n",
            N, N, params.alpha_0, params.n_0);
        return 0;
    }
    return 1;
}
