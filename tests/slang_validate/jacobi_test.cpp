// Real-input validator for Curvenet.SlangCodegen.Jacobi.
// y[i] = (d[i] == 0) ? 0 : b[i] / d[i].
// Promoted-vertex slots have d=0 and stay zero (caller overlays).

#include <cmath>
#include <cstdint>
#include <cstdio>

#include "jacobi_emit.cpp"

int main() {
    constexpr uint32_t N = 5;
    JacobiParams_0 params{N};

    float d[N] = {2.0f, 5.0f, 0.0f, 8.0f, 1.0f};   // d[2] = 0 → y[2] = 0
    float b[N] = {6.0f, 25.0f, 99.0f, 24.0f, -7.0f};
    float y[N] = {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
    const float expected[N] = {3.0f, 5.0f, 0.0f, 3.0f, -7.0f};

    GlobalParams_0 gp{};
    gp.params_0 = &params;
    gp.d_0.data = d;  gp.d_0.count = N;
    gp.b_0.data = b;  gp.b_0.count = N;
    gp.y_0.data = y;  gp.y_0.count = N;

    ComputeVaryingInput vi{};
    vi.startGroupID = uint3(0, 0, 0);
    vi.endGroupID   = uint3(1, 1, 1);

    main_0(&vi, nullptr, &gp);

    int fails = 0;
    for (uint32_t i = 0; i < N; ++i) {
        if (std::fabs(y[i] - expected[i]) > 1e-6f) {
            std::fprintf(stderr,
                "jacobi mismatch at i=%u: got %f, expected %f\n",
                i, y[i], expected[i]);
            ++fails;
        }
    }

    if (fails == 0) {
        std::printf("jacobi: %u/%u OK (n=%u, zero-diag at i=2)\n",
            N, N, params.n_0);
        return 0;
    }
    return 1;
}
