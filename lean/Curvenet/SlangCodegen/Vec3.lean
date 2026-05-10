import LeanSlang

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

private def floatTy : SlangType := .scalar .float
private def f3Ty    : SlangType := .vec .float 3
private def uintTy  : SlangType := .scalar .uint

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

example : shader.entryPointName = "main" := by native_decide

end Curvenet.SlangCodegen.Vec3
