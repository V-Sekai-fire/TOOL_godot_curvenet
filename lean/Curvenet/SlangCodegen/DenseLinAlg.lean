import LeanSlang
import Curvenet.SlangCodegen.Common

/-!
# `Curvenet.SlangCodegen.DenseLinAlg` — small-system dense solve

Slang re-port of the dense LU path in `src/curvenet/dense_linalg.h`.
Solves an N×N dense system `A x = b` with `N ≤ 16` (curvenet sample
matrices are tiny). One thread per system; the host packs many
independent systems for the same dispatch (e.g. per-handle bind-time
unit-perturbation solves).

LU factorization without pivoting; OK for the curvenet bind-time
SPD systems but not safe for general dense — production callers
verify `det != 0` before dispatch.

Bindings (set 0):
  0 — `ConstantBuffer<DenseLinAlgParams> { uint num_systems; uint n; }`
  1 — `StructuredBuffer<float> A` (n*n floats per system)
  2 — `StructuredBuffer<float> b` (n floats per system)
  3 — `RWStructuredBuffer<float> x` (n floats per system, output)

A `N_MAX = 16` stack array is used for the working LU. Set the
shader's `N_MAX` literal to match the host's max dimension.
-/

namespace Curvenet.SlangCodegen.DenseLinAlg

open LeanSlang
open Curvenet.SlangCodegen.Common


def shader : SlangShaderModule :=
  { structs :=
      [ { name := "DenseLinAlgParams"
        , fields :=
            [ ⟨"num_systems", .scalar .uint, Semantic.none, none, none, .qIn⟩
            , ⟨"n",           .scalar .uint, Semantic.none, none, none, .qIn⟩ ] } ]
  , globals :=
      [ ⟨"params", .const "DenseLinAlgParams", Semantic.none, some 0, some 0, .qIn⟩
      , ⟨"A",      .roBuf floatTy,             Semantic.none, some 1, some 0, .qIn⟩
      , ⟨"b",      .roBuf floatTy,             Semantic.none, some 2, some 0, .qIn⟩
      , ⟨"x",      .rwBuf floatTy,             Semantic.none, some 3, some 0, .qIn⟩ ]
  , functions := [{
      attrs  := [.shaderCompute, .numthreads 64 1 1]
      name   := "main"
      params := [⟨"tid", .vec .uint 3, .svDispatchThreadId, none, none, .qIn⟩]
      body   :=
        [ .declInit uintTy "s" (.member (.var "tid") "x")
        , .ifNoElse (.bin ">=" (.var "s") (.member (.var "params") "num_systems"))
            [ .ret none ]
        , .declInit uintTy "n" (.member (.var "params") "n")
        , .declInit uintTy "abase" (.bin "*" (.var "s") (.bin "*" (.var "n") (.var "n")))
        , .declInit uintTy "vbase" (.bin "*" (.var "s") (.var "n"))
        , .declareArray floatTy "M" 256  -- N_MAX² stack scratch
        , .declareArray floatTy "y" 16
        , -- Load A into M
          .forCount "i" (.litUint 0) (.var "n")
            [ .forCount "j" (.litUint 0) (.var "n")
                [ .assign
                    (.index (.var "M")
                      (.bin "+"
                        (.bin "*" (.var "i") (.var "n"))
                        (.var "j")))
                    (.index (.var "A")
                      (.bin "+" (.var "abase")
                        (.bin "+"
                          (.bin "*" (.var "i") (.var "n"))
                          (.var "j")))) ] ]
        , -- LU without pivoting: Doolittle in-place.
          .forCount "k" (.litUint 0) (.var "n")
            [ .declInit floatTy "akk"
                (.index (.var "M")
                  (.bin "+"
                    (.bin "*" (.var "k") (.var "n"))
                    (.var "k")))
            , .ifNoElse (.bin ">" (.call "abs" [.var "akk"]) (.litFloat 1e-30))
                [ .forCount "i" (.bin "+" (.var "k") (.litUint 1)) (.var "n")
                    [ .declInit floatTy "factor"
                        (.bin "/"
                          (.index (.var "M")
                            (.bin "+"
                              (.bin "*" (.var "i") (.var "n"))
                              (.var "k")))
                          (.var "akk"))
                    , .assign
                        (.index (.var "M")
                          (.bin "+"
                            (.bin "*" (.var "i") (.var "n"))
                            (.var "k")))
                        (.var "factor")
                    , .forCount "j" (.bin "+" (.var "k") (.litUint 1)) (.var "n")
                        [ .assign
                            (.index (.var "M")
                              (.bin "+"
                                (.bin "*" (.var "i") (.var "n"))
                                (.var "j")))
                            (.bin "-"
                              (.index (.var "M")
                                (.bin "+"
                                  (.bin "*" (.var "i") (.var "n"))
                                  (.var "j")))
                              (.bin "*" (.var "factor")
                                (.index (.var "M")
                                  (.bin "+"
                                    (.bin "*" (.var "k") (.var "n"))
                                    (.var "j")))))
                        ] ] ] ]
        , -- Forward solve  L y = b   (L is unit-lower in M)
          .forCount "i" (.litUint 0) (.var "n")
            [ .declInit floatTy "yi"
                (.index (.var "b") (.bin "+" (.var "vbase") (.var "i")))
            , .forCount "j" (.litUint 0) (.var "i")
                [ .assign (.var "yi")
                    (.bin "-" (.var "yi")
                      (.bin "*"
                        (.index (.var "M")
                          (.bin "+"
                            (.bin "*" (.var "i") (.var "n"))
                            (.var "j")))
                        (.index (.var "y") (.var "j")))) ]
            , .assign (.index (.var "y") (.var "i")) (.var "yi") ]
        , -- Back solve  U x = y
          .forCount "ii" (.litUint 0) (.var "n")
            [ .declInit uintTy "i"
                (.bin "-" (.bin "-" (.var "n") (.litUint 1)) (.var "ii"))
            , .declInit floatTy "xi" (.index (.var "y") (.var "i"))
            , .forCount "j" (.bin "+" (.var "i") (.litUint 1)) (.var "n")
                [ .assign (.var "xi")
                    (.bin "-" (.var "xi")
                      (.bin "*"
                        (.index (.var "M")
                          (.bin "+"
                            (.bin "*" (.var "i") (.var "n"))
                            (.var "j")))
                        (.index (.var "x")
                          (.bin "+" (.var "vbase") (.var "j"))))) ]
            , .declInit floatTy "uii"
                (.index (.var "M")
                  (.bin "+"
                    (.bin "*" (.var "i") (.var "n"))
                    (.var "i")))
            , .assign (.index (.var "x") (.bin "+" (.var "vbase") (.var "i")))
                (.ternary (.bin "<" (.call "abs" [.var "uii"]) (.litFloat 1e-30))
                  (.litFloat 0.0)
                  (.bin "/" (.var "xi") (.var "uii"))) ]
        , .ret none ] }] }

def expected : String :=
"struct DenseLinAlgParams {
  uint num_systems;
  uint n;
};

[[vk::binding(0, 0)]]
ConstantBuffer<DenseLinAlgParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> A;
[[vk::binding(2, 0)]]
StructuredBuffer<float> b;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float> x;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint s = tid.x;
  if ((s >= params.num_systems)) {
    return;
  }
  uint n = params.n;
  uint abase = (s * (n * n));
  uint vbase = (s * n);
  float M[256];
  float y[16];
  for (uint i = 0u; i < n; ++i) {
    for (uint j = 0u; j < n; ++j) {
      M[((i * n) + j)] = A[(abase + ((i * n) + j))];
    }
  }
  for (uint k = 0u; k < n; ++k) {
    float akk = M[((k * n) + k)];
    if ((abs(akk) > 0.000000)) {
      for (uint i = (k + 1u); i < n; ++i) {
        float factor = (M[((i * n) + k)] / akk);
        M[((i * n) + k)] = factor;
        for (uint j = (k + 1u); j < n; ++j) {
          M[((i * n) + j)] = (M[((i * n) + j)] - (factor * M[((k * n) + j)]));
        }
      }
    }
  }
  for (uint i = 0u; i < n; ++i) {
    float yi = b[(vbase + i)];
    for (uint j = 0u; j < i; ++j) {
      yi = (yi - (M[((i * n) + j)] * y[j]));
    }
    y[i] = yi;
  }
  for (uint ii = 0u; ii < n; ++ii) {
    uint i = ((n - 1u) - ii);
    float xi = y[i];
    for (uint j = (i + 1u); j < n; ++j) {
      xi = (xi - (M[((i * n) + j)] * x[(vbase + j)]));
    }
    float uii = M[((i * n) + i)];
    x[(vbase + i)] = ((abs(uii) < 0.000000) ? 0.000000 : (xi / uii));
  }
  return;
}"

example : LeanSlang.emit shader = expected := by native_decide

example : shader.entryPointName = "main" := by native_decide

end Curvenet.SlangCodegen.DenseLinAlg
