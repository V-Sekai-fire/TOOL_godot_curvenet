// Lean FFI shim around the slangc-emitted axpy CPU kernel.
//
// Bridges Lean's FloatArray runtime representation to the
// `main_0(varyingInput, params, globals)` entry point that
// `slangc -target cpp` produces from `axpy.slang`.
//
// Pure (deterministic, no I/O); native_decide can call this and
// reduce.

#include <cstdint>
#include <cstring>

#include <lean/lean.h>

#include "axpy_emit.cpp"

extern "C" LEAN_EXPORT lean_obj_res
lean_c_axpy_kernel(uint32_t n, double alpha,
                    b_lean_obj_arg x_arr, b_lean_obj_arg y_arr) {
    // Lean FloatArrays use the size_t-prefixed contiguous double[] layout
    // exposed by lean_float_array_*.
    const double* xd = lean_float_array_cptr(x_arr);
    const double* yd = lean_float_array_cptr(y_arr);

    // Slang kernel works in fp32 (matches the Slang AST's float type);
    // marshal the input doubles down + the output floats back up.
    float xf[1024];
    float yf[1024];
    if (n > 1024) { return lean_alloc_array(0, 0); }
    for (uint32_t i = 0; i < n; ++i) { xf[i] = (float)xd[i]; }
    for (uint32_t i = 0; i < n; ++i) { yf[i] = (float)yd[i]; }

    AxpyParams_0 params{n, (float)alpha};
    GlobalParams_0 gp{};
    gp.params_0 = &params;
    gp.x_0.data = xf; gp.x_0.count = n;
    gp.y_0.data = yf; gp.y_0.count = n;

    // numthreads(256, 1, 1); ceilDiv(n, 256) groups.
    uint32_t groups = (n + 255u) / 256u;
    ComputeVaryingInput vi{};
    vi.startGroupID = uint3(0, 0, 0);
    vi.endGroupID   = uint3(groups, 1, 1);
    main_0(&vi, nullptr, &gp);

    // Build a fresh Lean FloatArray (size_t length + raw double[] data).
    lean_object* out = lean_alloc_sarray(sizeof(double), n, n);
    double* od = (double*)lean_sarray_cptr(out);
    for (uint32_t i = 0; i < n; ++i) { od[i] = (double)yf[i]; }
    return out;
}
