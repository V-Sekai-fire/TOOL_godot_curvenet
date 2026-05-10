import LeanSlang

/-!
# `Curvenet.SlangCodegen.AxpyMulti` — multi-RHS AXPY

Slang re-port of `src/curvenet/shaders/axpy_multi.comp`. Computes
`y[gid] += alpha[c] * x[gid]` where `c = gid % k`, for `gid < n*k`.
One thread per (vertex, column) pair.
-/

namespace Curvenet.SlangCodegen.AxpyMulti

open LeanSlang

def shader : SlangShaderModule :=
  { structs :=
      [ { name := "AxpyMultiParams"
        , fields :=
            [ ⟨"n", .scalar .uint, Semantic.none, none, none, .qIn⟩
            , ⟨"k", .scalar .uint, Semantic.none, none, none, .qIn⟩ ] } ]
  , globals :=
      [ ⟨"params", .const "AxpyMultiParams",   Semantic.none, some 0, some 0, .qIn⟩
      , ⟨"alpha",  .roBuf (.scalar .float),    Semantic.none, some 1, some 0, .qIn⟩
      , ⟨"x",      .roBuf (.scalar .float),    Semantic.none, some 2, some 0, .qIn⟩
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
        , .declInit (.scalar .uint) "c"
            (.bin "%" (.var "gid") (.member (.var "params") "k"))
        , .assign (.index (.var "y") (.var "gid"))
            (.call "fma"
              [ .index (.var "alpha") (.var "c")
              , .index (.var "x") (.var "gid")
              , .index (.var "y") (.var "gid") ])
        ] }] }

def expected : String :=
"struct AxpyMultiParams {
  uint n;
  uint k;
};

[[vk::binding(0, 0)]]
ConstantBuffer<AxpyMultiParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> alpha;
[[vk::binding(2, 0)]]
StructuredBuffer<float> x;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float> y;

[shader(\"compute\")] [numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint gid = tid.x;
  uint total = (params.n * params.k);
  if ((gid >= total)) {
    return;
  }
  uint c = (gid % params.k);
  y[gid] = fma(alpha[c], x[gid], y[gid]);
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide

end Curvenet.SlangCodegen.AxpyMulti
