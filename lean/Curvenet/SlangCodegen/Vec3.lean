import LeanSlang
import Curvenet.SlangCodegen.Common

/-!
# `Curvenet.SlangCodegen.Vec3` — float3 baseline shader

The C++ `curvenet::Vec3` (3 doubles + +, -, *, /, dot, length) maps
1-to-1 onto Slang's built-in `float3`. There's no algorithm worth
porting beyond the type-equivalence; the shader below is a
documenting smoke-test that exercises Slang's float3 ops the way
downstream kernels (PolygonLaplacian, ScaledFrames, …) rely on.

One thread per (a, b) pair. Computes
`out_dot[i] = dot(a[i], b[i])` and
`out_len[i] = length(a[i])`. Validates that Slang's `dot` and
`length` builtins are wired up the way the higher-level kernels
assume.
-/

namespace Curvenet.SlangCodegen.Vec3

open LeanSlang
open Curvenet.SlangCodegen.Common


def shader : SlangShaderModule :=
  { structs :=
      [ { name := "Vec3Params"
        , fields := [⟨"n", .scalar .uint, Semantic.none, none, none, .qIn⟩] } ]
  , globals :=
      [ ⟨"params",  .const "Vec3Params", Semantic.none, some 0, some 0, .qIn⟩
      , ⟨"a",       .roBuf f3Ty,         Semantic.none, some 1, some 0, .qIn⟩
      , ⟨"b",       .roBuf f3Ty,         Semantic.none, some 2, some 0, .qIn⟩
      , ⟨"out_dot", .rwBuf floatTy,      Semantic.none, some 3, some 0, .qIn⟩
      , ⟨"out_len", .rwBuf floatTy,      Semantic.none, some 4, some 0, .qIn⟩ ]
  , functions := [{
      attrs  := [.shaderCompute, .numthreads 256 1 1]
      name   := "main"
      params := [⟨"tid", .vec .uint 3, .svDispatchThreadId, none, none, .qIn⟩]
      body   :=
        [ .declInit uintTy "i" (.member (.var "tid") "x")
        , .ifNoElse (.bin ">=" (.var "i") (.member (.var "params") "n"))
            [ .ret none ]
        , .declInit f3Ty "av" (.index (.var "a") (.var "i"))
        , .declInit f3Ty "bv" (.index (.var "b") (.var "i"))
        , .assign (.index (.var "out_dot") (.var "i"))
            (.call "dot" [.var "av", .var "bv"])
        , .assign (.index (.var "out_len") (.var "i"))
            (.call "length" [.var "av"])
        , .ret none ] }] }

def expected : String :=
"struct Vec3Params {
  uint n;
};

[[vk::binding(0, 0)]]
ConstantBuffer<Vec3Params> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float3> a;
[[vk::binding(2, 0)]]
StructuredBuffer<float3> b;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float> out_dot;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> out_len;

[shader(\"compute\")] [numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if ((i >= params.n)) {
    return;
  }
  float3 av = a[i];
  float3 bv = b[i];
  out_dot[i] = dot(av, bv);
  out_len[i] = length(av);
  return;
}"

example : LeanSlang.emit shader = expected := by native_decide

example : shader.entryPointName = "main" := by native_decide

end Curvenet.SlangCodegen.Vec3
