import LeanSlang

/-!
# `Curvenet.SlangCodegen.SgsColor` — multi-color symmetric Gauss-Seidel

Slang re-port of `src/curvenet/shaders/sgs_color.comp`. One thread
per row of the active color. Graph coloring guarantees no adjacent
same-color vertices, so workgroup-parallel execution is
mathematically equivalent to scalar lexical SGS within the color
group.

The host invokes once per color: forward sweep walks colors 0..C-1,
then a backward sweep walks C-1..0. Symmetry preserves the SPD
M^{-1} property required by the outer PCG.

Branchless row update: full row dot product, then subtract diagonal
contribution. Mirrors the CPU's branchless SGS to keep numerical
parity.

Bindings (set 0):
  0 — `ConstantBuffer<SgsParams> { uint n; uint num_color_rows; }`
  1-3 — CSR rowPtr, colIdx, values
  4 — `StructuredBuffer<float> diag` (cached A_ii, length n)
  5 — `StructuredBuffer<int> colorRows` (length num_color_rows)
  6 — `StructuredBuffer<float> b` (RHS, length n)
  7 — `RWStructuredBuffer<float> x` (iterate, length n)
-/

namespace Curvenet.SlangCodegen.SgsColor

open LeanSlang

private def floatTy : SlangType := .scalar .float
private def uintTy  : SlangType := .scalar .uint

def shader : SlangShaderModule :=
  { structs :=
      [ { name := "SgsParams"
        , fields :=
            [ ⟨"n",              .scalar .uint, Semantic.none, none, none, .qIn⟩
            , ⟨"num_color_rows", .scalar .uint, Semantic.none, none, none, .qIn⟩ ] } ]
  , globals :=
      [ ⟨"params",    .const "SgsParams",         Semantic.none, some 0, some 0, .qIn⟩
      , ⟨"rowPtr",    .roBuf (.scalar .int),      Semantic.none, some 1, some 0, .qIn⟩
      , ⟨"colIdx",    .roBuf (.scalar .int),      Semantic.none, some 2, some 0, .qIn⟩
      , ⟨"values",    .roBuf (.scalar .float),    Semantic.none, some 3, some 0, .qIn⟩
      , ⟨"diag",      .roBuf (.scalar .float),    Semantic.none, some 4, some 0, .qIn⟩
      , ⟨"colorRows", .roBuf (.scalar .int),      Semantic.none, some 5, some 0, .qIn⟩
      , ⟨"b",         .roBuf (.scalar .float),    Semantic.none, some 6, some 0, .qIn⟩
      , ⟨"x",         .rwBuf (.scalar .float),    Semantic.none, some 7, some 0, .qIn⟩ ]
  , functions := [{
      attrs  := [.shaderCompute, .numthreads 64 1 1]
      name   := "main"
      params := [⟨"tid", .vec .uint 3, .svDispatchThreadId, none, none, .qIn⟩]
      body   :=
        [ .declInit uintTy "t" (.member (.var "tid") "x")
        , .ifNoElse (.bin ">=" (.var "t") (.member (.var "params") "num_color_rows"))
            [ .ret none ]
        , .declInit uintTy "i"
            (.call "uint" [.index (.var "colorRows") (.var "t")])
        , .declInit floatTy "d" (.index (.var "diag") (.var "i"))
        , .ifNoElse (.bin "==" (.var "d") (.litFloat 0.0))
            [ .ret none ]
        , .declInit uintTy "rs"
            (.call "uint" [.index (.var "rowPtr") (.var "i")])
        , .declInit uintTy "re"
            (.call "uint" [.index (.var "rowPtr")
              (.bin "+" (.var "i") (.litUint 1))])
        , .declInit floatTy "s_full" (.litFloat 0.0)
        , .forCount "k" (.var "rs") (.var "re")
            [ .assign (.var "s_full")
                (.bin "+" (.var "s_full")
                  (.bin "*"
                    (.index (.var "values") (.var "k"))
                    (.index (.var "x")
                      (.call "uint" [.index (.var "colIdx") (.var "k")])))) ]
        , .declInit floatTy "s_off"
            (.bin "-" (.var "s_full")
                      (.bin "*" (.var "d") (.index (.var "x") (.var "i"))))
        , .assign (.index (.var "x") (.var "i"))
            (.bin "/"
              (.bin "-" (.index (.var "b") (.var "i")) (.var "s_off"))
              (.var "d"))
        , .ret none
        ] }] }

def expected : String :=
"struct SgsParams {
  uint n;
  uint num_color_rows;
};

[[vk::binding(0, 0)]]
ConstantBuffer<SgsParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<int> rowPtr;
[[vk::binding(2, 0)]]
StructuredBuffer<int> colIdx;
[[vk::binding(3, 0)]]
StructuredBuffer<float> values;
[[vk::binding(4, 0)]]
StructuredBuffer<float> diag;
[[vk::binding(5, 0)]]
StructuredBuffer<int> colorRows;
[[vk::binding(6, 0)]]
StructuredBuffer<float> b;
[[vk::binding(7, 0)]]
RWStructuredBuffer<float> x;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint t = tid.x;
  if ((t >= params.num_color_rows)) {
    return;
  }
  uint i = uint(colorRows[t]);
  float d = diag[i];
  if ((d == 0.000000)) {
    return;
  }
  uint rs = uint(rowPtr[i]);
  uint re = uint(rowPtr[(i + 1u)]);
  float s_full = 0.000000;
  for (uint k = rs; k < re; ++k) {
    s_full = (s_full + (values[k] * x[uint(colIdx[k])]));
  }
  float s_off = (s_full - (d * x[i]));
  x[i] = ((b[i] - s_off) / d);
  return;
}"

example : LeanSlang.emit shader = expected := by native_decide

example : shader.entryPointName = "main" := by native_decide

end Curvenet.SlangCodegen.SgsColor
