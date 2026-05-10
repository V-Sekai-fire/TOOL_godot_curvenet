import LeanSlang

/-!
# `Curvenet.SlangCodegen.Axpy` — in-place AXPY compute kernel

Slang re-port of `src/curvenet/shaders/axpy.comp`, codegen'd via
LeanSlang. Computes `y[i] += alpha * x[i]` for `i < n`. One thread
per element, fused multiply-add per write.

Bindings (set 0):
  binding 0 — `ConstantBuffer<AxpyParams> { uint n; float alpha; }`
  binding 1 — `StructuredBuffer<float> x`
  binding 2 — `RWStructuredBuffer<float> y`
-/

namespace Curvenet.SlangCodegen.Axpy

open LeanSlang

def shader : SlangShaderModule :=
  { structs :=
      [ { name := "AxpyParams"
        , fields :=
            [ ⟨"n",     .scalar .uint,  Semantic.none, none, none, .qIn⟩
            , ⟨"alpha", .scalar .float, Semantic.none, none, none, .qIn⟩ ] } ]
  , globals :=
      [ ⟨"params", .const "AxpyParams",       Semantic.none, some 0, some 0, .qIn⟩
      , ⟨"x",      .roBuf (.scalar .float),   Semantic.none, some 1, some 0, .qIn⟩
      , ⟨"y",      .rwBuf (.scalar .float),   Semantic.none, some 2, some 0, .qIn⟩ ]
  , functions := [{
      attrs  := [.shaderCompute, .numthreads 256 1 1]
      name   := "main"
      params := [⟨"tid", .vec .uint 3, .svDispatchThreadId, none, none, .qIn⟩]
      body   :=
        [ .declInit (.scalar .uint) "i" (.member (.var "tid") "x")
        , .ifNoElse (.bin ">=" (.var "i") (.member (.var "params") "n"))
            [ .ret none ]
        , .assign (.index (.var "y") (.var "i"))
            (.call "fma"
              [ .member (.var "params") "alpha"
              , .index (.var "x") (.var "i")
              , .index (.var "y") (.var "i") ])
        ] }] }

def expected : String :=
"struct AxpyParams {
  uint n;
  float alpha;
};

[[vk::binding(0, 0)]]
ConstantBuffer<AxpyParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> x;
[[vk::binding(2, 0)]]
RWStructuredBuffer<float> y;

[shader(\"compute\")] [numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if ((i >= params.n)) {
    return;
  }
  y[i] = fma(params.alpha, x[i], y[i]);
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide

end Curvenet.SlangCodegen.Axpy
