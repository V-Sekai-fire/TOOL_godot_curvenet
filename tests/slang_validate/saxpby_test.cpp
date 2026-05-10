// Real-input validator for Curvenet.SlangCodegen.Saxpby.
// dst[i] = alpha * x[i] + beta * y[i].

#include <cmath>
#include <cstdint>
#include <cstdio>

#include "saxpby_emit.cpp"

int main() {
    constexpr uint32_t N = 5;
    SaxpbyParams_0 params{N, 3.0f, -1.0f};

    float x[N]   = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float y[N]   = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
    float dst[N] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const float expected[N] = {
        3.0f * 1.0f - 1.0f * 10.0f,    // -7
        3.0f * 2.0f - 1.0f * 20.0f,    // -14
        3.0f * 3.0f - 1.0f * 30.0f,    // -21
        3.0f * 4.0f - 1.0f * 40.0f,    // -28
        3.0f * 5.0f - 1.0f * 50.0f,    // -35
    };

    GlobalParams_0 gp{};
    gp.params_0 = &params;
    gp.x_0.data = x;       gp.x_0.count = N;
    gp.y_0.data = y;       gp.y_0.count = N;
    gp.dst_0.data = dst;   gp.dst_0.count = N;

    ComputeVaryingInput vi{};
    vi.startGroupID = uint3(0, 0, 0);
    vi.endGroupID   = uint3(1, 1, 1);

    main_0(&vi, nullptr, &gp);

    int fails = 0;
    for (uint32_t i = 0; i < N; ++i) {
        if (std::fabs(dst[i] - expected[i]) > 1e-6f) {
            std::fprintf(stderr,
                "saxpby mismatch at i=%u: got %f, expected %f\n",
                i, dst[i], expected[i]);
            ++fails;
        }
    }

    if (fails == 0) {
        std::printf("saxpby: %u/%u OK (alpha=%g, beta=%g, n=%u)\n",
            N, N, params.alpha_0, params.beta_0, params.n_0);
        return 0;
    }
    return 1;
}
