import LeanSlang

/-!
# `Curvenet.SlangCodegen.SaxpbyMulti` — multi-RHS saxpby

Slang re-port of `src/curvenet/shaders/saxpby_multi.comp`.
`dst[gid] = alpha[c] * x[gid] + beta[c] * y[gid]` where `c = gid % k`,
for `gid < n*k`.
-/

namespace Curvenet.SlangCodegen.SaxpbyMulti

open LeanSlang

def shader : SlangShaderModule :=
  { structs :=
      [ { name := "SaxpbyMultiParams"
        , fields :=
            [ ⟨"n", .scalar .uint, Semantic.none, none, none, .qIn⟩
            , ⟨"k", .scalar .uint, Semantic.none, none, none, .qIn⟩ ] } ]
  , globals :=
      [ ⟨"params", .const "SaxpbyMultiParams", Semantic.none, some 0, some 0, .qIn⟩
      , ⟨"alpha",  .roBuf (.scalar .float),    Semantic.none, some 1, some 0, .qIn⟩
      , ⟨"beta",   .roBuf (.scalar .float),    Semantic.none, some 2, some 0, .qIn⟩
      , ⟨"x",      .roBuf (.scalar .float),    Semantic.none, some 3, some 0, .qIn⟩
      , ⟨"y",      .roBuf (.scalar .float),    Semantic.none, some 4, some 0, .qIn⟩
      , ⟨"dst",    .rwBuf (.scalar .float),    Semantic.none, some 5, some 0, .qIn⟩ ]
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
        , .assign (.index (.var "dst") (.var "gid"))
            (.call "fma"
              [ .index (.var "alpha") (.var "c")
              , .index (.var "x") (.var "gid")
              , .bin "*" (.index (.var "beta") (.var "c"))
                         (.index (.var "y") (.var "gid")) ])
        ] }] }

def expected : String :=
"struct SaxpbyMultiParams {
  uint n;
  uint k;
};

[[vk::binding(0, 0)]]
ConstantBuffer<SaxpbyMultiParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> alpha;
[[vk::binding(2, 0)]]
StructuredBuffer<float> beta;
[[vk::binding(3, 0)]]
StructuredBuffer<float> x;
[[vk::binding(4, 0)]]
StructuredBuffer<float> y;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float> dst;

[shader(\"compute\")] [numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint gid = tid.x;
  uint total = (params.n * params.k);
  if ((gid >= total)) {
    return;
  }
  uint c = (gid % params.k);
  dst[gid] = fma(alpha[c], x[gid], (beta[c] * y[gid]));
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide

end Curvenet.SlangCodegen.SaxpbyMulti
