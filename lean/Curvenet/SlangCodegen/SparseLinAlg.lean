import LeanSlang
import Curvenet.SlangCodegen.Common

/-!
# `Curvenet.SlangCodegen.SparseLinAlg` — composite per-vertex CG step

Slang re-port of the inner-loop CG kernel from
`src/curvenet/sparse_linalg.h`. One iteration's per-vertex work
fused into a single shader: SpMV + SAXPY + residual update.

Production CG splits this into separate `Spmv`, `Saxpby`, and
`DotReduce` dispatches (already ported as their own modules) so
the host can chain them with explicit barriers between. This
module captures the per-vertex view for documentation + reference;
the production path uses the split kernels.

Bindings:
  0 — `ConstantBuffer<SparseLAParams> { uint n; float alpha; }`
  1-3 — CSR rowPtr/colIdx/values
  4 — `StructuredBuffer<float> p_old`  (search direction)
  5 — `StructuredBuffer<float> r_old`  (residual)
  6 — `StructuredBuffer<float> x_old`  (iterate)
  7-8 — RWStructuredBuffer<float> r_new, x_new
-/

namespace Curvenet.SlangCodegen.SparseLinAlg

open LeanSlang
open Curvenet.SlangCodegen.Common


def shader : SlangShaderModule :=
  { structs :=
      [ { name := "SparseLAParams"
        , fields :=
            [ ⟨"n",     .scalar .uint,  Semantic.none, none, none, .qIn⟩
            , ⟨"alpha", .scalar .float, Semantic.none, none, none, .qIn⟩ ] } ]
  , globals :=
      [ ⟨"params", .const "SparseLAParams",   Semantic.none, some 0, some 0, .qIn⟩
      , ⟨"rowPtr", .roBuf intTy,              Semantic.none, some 1, some 0, .qIn⟩
      , ⟨"colIdx", .roBuf intTy,              Semantic.none, some 2, some 0, .qIn⟩
      , ⟨"values", .roBuf floatTy,            Semantic.none, some 3, some 0, .qIn⟩
      , ⟨"p_old",  .roBuf floatTy,            Semantic.none, some 4, some 0, .qIn⟩
      , ⟨"r_old",  .roBuf floatTy,            Semantic.none, some 5, some 0, .qIn⟩
      , ⟨"x_old",  .roBuf floatTy,            Semantic.none, some 6, some 0, .qIn⟩
      , ⟨"r_new",  .rwBuf floatTy,            Semantic.none, some 7, some 0, .qIn⟩
      , ⟨"x_new",  .rwBuf floatTy,            Semantic.none, some 8, some 0, .qIn⟩ ]
  , functions := [{
      attrs  := [.shaderCompute, .numthreads 256 1 1]
      name   := "main"
      params := [⟨"tid", .vec .uint 3, .svDispatchThreadId, none, none, .qIn⟩]
      body   :=
        [ .declInit uintTy "i" (.member (.var "tid") "x")
        , .ifNoElse (.bin ">=" (.var "i") (.member (.var "params") "n"))
            [ .ret none ]
        , -- Compute (A·p)[i]
          .declInit uintTy "rs"
            (.call "uint" [.index (.var "rowPtr") (.var "i")])
        , .declInit uintTy "re"
            (.call "uint" [.index (.var "rowPtr")
              (.bin "+" (.var "i") (.litUint 1))])
        , .declInit floatTy "Api" (.litFloat 0.0)
        , .forCount "p" (.var "rs") (.var "re")
            [ .assign (.var "Api")
                (.call "fma"
                  [ .index (.var "values") (.var "p")
                  , .index (.var "p_old")
                      (.call "uint" [.index (.var "colIdx") (.var "p")])
                  , .var "Api" ]) ]
        , -- x_new[i] = x_old[i] + alpha * p_old[i]
          .assign (.index (.var "x_new") (.var "i"))
            (.call "fma"
              [ .member (.var "params") "alpha"
              , .index (.var "p_old") (.var "i")
              , .index (.var "x_old") (.var "i") ])
        , -- r_new[i] = r_old[i] - alpha * Api
          .assign (.index (.var "r_new") (.var "i"))
            (.bin "-" (.index (.var "r_old") (.var "i"))
              (.bin "*" (.member (.var "params") "alpha") (.var "Api")))
        , .ret none ] }] }

def expected : String :=
"struct SparseLAParams {
  uint n;
  float alpha;
};

[[vk::binding(0, 0)]]
ConstantBuffer<SparseLAParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<int> rowPtr;
[[vk::binding(2, 0)]]
StructuredBuffer<int> colIdx;
[[vk::binding(3, 0)]]
StructuredBuffer<float> values;
[[vk::binding(4, 0)]]
StructuredBuffer<float> p_old;
[[vk::binding(5, 0)]]
StructuredBuffer<float> r_old;
[[vk::binding(6, 0)]]
StructuredBuffer<float> x_old;
[[vk::binding(7, 0)]]
RWStructuredBuffer<float> r_new;
[[vk::binding(8, 0)]]
RWStructuredBuffer<float> x_new;

[shader(\"compute\")] [numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if ((i >= params.n)) {
    return;
  }
  uint rs = uint(rowPtr[i]);
  uint re = uint(rowPtr[(i + 1u)]);
  float Api = 0.000000;
  for (uint p = rs; p < re; ++p) {
    Api = fma(values[p], p_old[uint(colIdx[p])], Api);
  }
  x_new[i] = fma(params.alpha, p_old[i], x_old[i]);
  r_new[i] = (r_old[i] - (params.alpha * Api));
  return;
}"

example : LeanSlang.emit shader = expected := by native_decide

example : shader.entryPointName = "main" := by native_decide

end Curvenet.SlangCodegen.SparseLinAlg
