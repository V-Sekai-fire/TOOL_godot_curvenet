import LeanSlang

/-!
# `Curvenet.SlangCodegen.SpmvMulti` — multi-RHS sparse matvec

Slang re-port of `src/curvenet/shaders/spmv_multi.comp`. Computes
`Y = A · X` where X and Y are `nv × k` row-major matrices. Each row
of A is read once per (i) for k different output columns, so per-row
memory traffic on A is shared across columns — the main win when
k > 1 over k separate spmv calls.

`K_MAX = 16` is the compile-time bound on k that lets the per-thread
accumulator fit in a fixed register array. For our use (deformer
9-col Fv solve, 3-col Xv solve), 16 is comfortable.

Bindings (set 0):
  binding 0 — `ConstantBuffer<SpmvMultiParams> { uint rows; uint k; }`
  binding 1-3 — CSR rowPtr, colIdx, values
  binding 4 — `StructuredBuffer<float> x`   (length nv * k)
  binding 5 — `RWStructuredBuffer<float> y` (length nv * k)
-/

namespace Curvenet.SlangCodegen.SpmvMulti

open LeanSlang

def shader : SlangShaderModule :=
  { structs :=
      [ { name := "SpmvMultiParams"
        , fields :=
            [ ⟨"rows", .scalar .uint, Semantic.none, none, none, .qIn⟩
            , ⟨"k",    .scalar .uint, Semantic.none, none, none, .qIn⟩ ] } ]
  , globals :=
      [ ⟨"params", .const "SpmvMultiParams",  Semantic.none, some 0, some 0, .qIn⟩
      , ⟨"rowPtr", .roBuf (.scalar .int),     Semantic.none, some 1, some 0, .qIn⟩
      , ⟨"colIdx", .roBuf (.scalar .int),     Semantic.none, some 2, some 0, .qIn⟩
      , ⟨"values", .roBuf (.scalar .float),   Semantic.none, some 3, some 0, .qIn⟩
      , ⟨"x",      .roBuf (.scalar .float),   Semantic.none, some 4, some 0, .qIn⟩
      , ⟨"y",      .rwBuf (.scalar .float),   Semantic.none, some 5, some 0, .qIn⟩ ]
  , functions := [{
      attrs  := [.shaderCompute, .numthreads 256 1 1]
      name   := "main"
      params := [⟨"tid", .vec .uint 3, .svDispatchThreadId, none, none, .qIn⟩]
      body   :=
        [ .declInit (.scalar .uint) "i" (.member (.var "tid") "x")
        , .ifNoElse (.bin ">=" (.var "i") (.member (.var "params") "rows"))
            [ .ret none ]
        , .declInit (.scalar .uint) "rs"
            (.call "uint" [.index (.var "rowPtr") (.var "i")])
        , .declInit (.scalar .uint) "re"
            (.call "uint" [.index (.var "rowPtr")
              (.bin "+" (.var "i") (.litUint 1))])
        , .declareArray (.scalar .float) "acc" 16
        , .forCount "c" (.litUint 0) (.member (.var "params") "k")
            [ .assign (.index (.var "acc") (.var "c")) (.litFloat 0.0) ]
        , .forCount "p" (.var "rs") (.var "re")
            [ .declInit (.scalar .float) "aij"
                (.index (.var "values") (.var "p"))
            , .declInit (.scalar .uint) "j"
                (.call "uint" [.index (.var "colIdx") (.var "p")])
            , .forCount "c" (.litUint 0) (.member (.var "params") "k")
                [ .assign (.index (.var "acc") (.var "c"))
                    (.call "fma"
                      [ .var "aij"
                      , .index (.var "x")
                          (.bin "+"
                            (.bin "*" (.var "j") (.member (.var "params") "k"))
                            (.var "c"))
                      , .index (.var "acc") (.var "c") ]) ] ]
        , .forCount "c" (.litUint 0) (.member (.var "params") "k")
            [ .assign (.index (.var "y")
                (.bin "+"
                  (.bin "*" (.var "i") (.member (.var "params") "k"))
                  (.var "c")))
                (.index (.var "acc") (.var "c")) ]
        ] }] }

def expected : String :=
"struct SpmvMultiParams {
  uint rows;
  uint k;
};

[[vk::binding(0, 0)]]
ConstantBuffer<SpmvMultiParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<int> rowPtr;
[[vk::binding(2, 0)]]
StructuredBuffer<int> colIdx;
[[vk::binding(3, 0)]]
StructuredBuffer<float> values;
[[vk::binding(4, 0)]]
StructuredBuffer<float> x;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float> y;

[shader(\"compute\")] [numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if ((i >= params.rows)) {
    return;
  }
  uint rs = uint(rowPtr[i]);
  uint re = uint(rowPtr[(i + 1u)]);
  float acc[16];
  for (uint c = 0u; c < params.k; ++c) {
    acc[c] = 0.000000;
  }
  for (uint p = rs; p < re; ++p) {
    float aij = values[p];
    uint j = uint(colIdx[p]);
    for (uint c = 0u; c < params.k; ++c) {
      acc[c] = fma(aij, x[((j * params.k) + c)], acc[c]);
    }
  }
  for (uint c = 0u; c < params.k; ++c) {
    y[((i * params.k) + c)] = acc[c];
  }
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide

end Curvenet.SlangCodegen.SpmvMulti
