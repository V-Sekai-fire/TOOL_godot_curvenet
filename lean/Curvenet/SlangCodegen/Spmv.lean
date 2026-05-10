import LeanSlang

/-!
# `Curvenet.SlangCodegen.Spmv` — sparse matrix-vector product

Slang re-port of `src/curvenet/shaders/spmv.comp`. Computes
`y[i] = Σ over nz in row i  values[nz] * x[col_idx[nz]]` for the
DeGoes22 deformer's GPU CG path. One thread per row.

Bindings (set 0):
  binding 0 — `ConstantBuffer<SpmvParams> { uint rows; }`
  binding 1 — `StructuredBuffer<int> rowPtr`   (length rows + 1)
  binding 2 — `StructuredBuffer<int> colIdx`   (length nnz)
  binding 3 — `StructuredBuffer<float> values` (length nnz)
  binding 4 — `StructuredBuffer<float> x`      (length cols)
  binding 5 — `RWStructuredBuffer<float> y`    (length rows)

Float32 throughout. Per-row spmv has only ~7 nonzeros on a triangle
Laplacian, so the fp32 accumulation error is ~7·ε and well below
the §4.3 solve's tolerance. Higher-precision sums (df32) live in
`Curvenet.SlangCodegen.DotReduce`.
-/

namespace Curvenet.SlangCodegen.Spmv

open LeanSlang

def shader : SlangShaderModule :=
  { structs :=
      [ { name := "SpmvParams"
        , fields := [⟨"rows", .scalar .uint, Semantic.none, none, none, .qIn⟩] } ]
  , globals :=
      [ ⟨"params", .const "SpmvParams",       Semantic.none, some 0, some 0, .qIn⟩
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
        , .declInit (.scalar .float) "s" (.litFloat 0.0)
        , .forCount "p" (.var "rs") (.var "re")
            [ .assign (.var "s")
                (.call "fma"
                  [ .index (.var "values") (.var "p")
                  , .index (.var "x")
                      (.call "uint" [.index (.var "colIdx") (.var "p")])
                  , .var "s" ]) ]
        , .assign (.index (.var "y") (.var "i")) (.var "s")
        ] }] }

def expected : String :=
"struct SpmvParams {
  uint rows;
};

[[vk::binding(0, 0)]]
ConstantBuffer<SpmvParams> params;
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
  float s = 0.000000;
  for (uint p = rs; p < re; ++p) {
    s = fma(values[p], x[uint(colIdx[p])], s);
  }
  y[i] = s;
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide

end Curvenet.SlangCodegen.Spmv
