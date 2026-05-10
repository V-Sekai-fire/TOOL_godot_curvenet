import LeanSlang

/-!
# `Curvenet.SlangCodegen.Jacobi` — diagonal preconditioner application

Slang re-port of `src/curvenet/shaders/jacobi.comp`. Computes
`y[i] = (d[i] == 0) ? 0 : b[i] / d[i]`. Zero-diagonal maps to zero
output — same convention as the CPU spec, where promoted vertices
produce zero rows in `LhsM` and the caller overlays the constraint
value back.
-/

namespace Curvenet.SlangCodegen.Jacobi

open LeanSlang

def shader : SlangShaderModule :=
  { structs :=
      [ { name := "JacobiParams"
        , fields := [⟨"n", .scalar .uint, Semantic.none, none, none, .qIn⟩] } ]
  , globals :=
      [ ⟨"params", .const "JacobiParams",     Semantic.none, some 0, some 0, .qIn⟩
      , ⟨"d",      .roBuf (.scalar .float),   Semantic.none, some 1, some 0, .qIn⟩
      , ⟨"b",      .roBuf (.scalar .float),   Semantic.none, some 2, some 0, .qIn⟩
      , ⟨"y",      .rwBuf (.scalar .float),   Semantic.none, some 3, some 0, .qIn⟩ ]
  , functions := [{
      attrs  := [.shaderCompute, .numthreads 256 1 1]
      name   := "main"
      params := [⟨"tid", .vec .uint 3, .svDispatchThreadId, none, none, .qIn⟩]
      body   :=
        [ .declInit (.scalar .uint) "i" (.member (.var "tid") "x")
        , .ifNoElse (.bin ">=" (.var "i") (.member (.var "params") "n"))
            [ .ret none ]
        , .declInit (.scalar .float) "dii" (.index (.var "d") (.var "i"))
        , .assign (.index (.var "y") (.var "i"))
            (.ternary
              (.bin "==" (.var "dii") (.litFloat 0.0))
              (.litFloat 0.0)
              (.bin "/" (.index (.var "b") (.var "i")) (.var "dii")))
        ] }] }

def expected : String :=
"struct JacobiParams {
  uint n;
};

[[vk::binding(0, 0)]]
ConstantBuffer<JacobiParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> d;
[[vk::binding(2, 0)]]
StructuredBuffer<float> b;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float> y;

[shader(\"compute\")] [numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if ((i >= params.n)) {
    return;
  }
  float dii = d[i];
  y[i] = ((dii == 0.000000) ? 0.000000 : (b[i] / dii));
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide

end Curvenet.SlangCodegen.Jacobi
