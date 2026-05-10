import LeanSlang

/-!
# `Curvenet.SlangCodegen.JacobiMulti` — multi-RHS Jacobi preconditioner

Slang re-port of `src/curvenet/shaders/jacobi_multi.comp`. Diagonal
`d` shared across all `k` RHS columns; thread `gid` handles vertex
`i = gid / k`, output column `c = gid % k`.
-/

namespace Curvenet.SlangCodegen.JacobiMulti

open LeanSlang

def shader : SlangShaderModule :=
  { structs :=
      [ { name := "JacobiMultiParams"
        , fields :=
            [ ⟨"n", .scalar .uint, Semantic.none, none, none, .qIn⟩
            , ⟨"k", .scalar .uint, Semantic.none, none, none, .qIn⟩ ] } ]
  , globals :=
      [ ⟨"params", .const "JacobiMultiParams", Semantic.none, some 0, some 0, .qIn⟩
      , ⟨"d",      .roBuf (.scalar .float),    Semantic.none, some 1, some 0, .qIn⟩
      , ⟨"b",      .roBuf (.scalar .float),    Semantic.none, some 2, some 0, .qIn⟩
      , ⟨"y",      .rwBuf (.scalar .float),    Semantic.none, some 3, some 0, .qIn⟩ ]
  , functions := [{
      attrs  := [.shaderCompute, .numthreads 256 1 1]
      name   := "main"
      params := [⟨"tid", .vec .uint 3, .svDispatchThreadId, none, none, .qIn⟩]
      body   :=
        [ .declInit (.scalar .uint) "gid" (.member (.var "tid") "x")
        , .declInit (.scalar .uint) "total"
            (.bin "*" (.member (.var "params") "n")
                      (.member (.var "params") "k"))
        , .ifNoElse (.bin ">=" (.var "gid") (.var "total"))
            [ .ret none ]
        , .declInit (.scalar .uint) "i"
            (.bin "/" (.var "gid") (.member (.var "params") "k"))
        , .declInit (.scalar .float) "dii" (.index (.var "d") (.var "i"))
        , .assign (.index (.var "y") (.var "gid"))
            (.ternary
              (.bin "==" (.var "dii") (.litFloat 0.0))
              (.litFloat 0.0)
              (.bin "/" (.index (.var "b") (.var "gid")) (.var "dii")))
        ] }] }

def expected : String :=
"struct JacobiMultiParams {
  uint n;
  uint k;
};

[[vk::binding(0, 0)]]
ConstantBuffer<JacobiMultiParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> d;
[[vk::binding(2, 0)]]
StructuredBuffer<float> b;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float> y;

[shader(\"compute\")] [numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint gid = tid.x;
  uint total = (params.n * params.k);
  if ((gid >= total)) {
    return;
  }
  uint i = (gid / params.k);
  float dii = d[i];
  y[gid] = ((dii == 0.000000) ? 0.000000 : (b[gid] / dii));
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide

end Curvenet.SlangCodegen.JacobiMulti
