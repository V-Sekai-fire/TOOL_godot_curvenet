import LeanSlang

/-!
# `Curvenet.SlangCodegen.RobustLaplacian` — clamped cot Laplacian

Slang re-port of `src/curvenet/robust_laplacian.h`. Same per-triangle
3×3 cot Laplacian as `PolygonLaplacian` but with the cotangent values
clamped to a finite range to prevent NaN propagation when triangles
degenerate (very thin or zero-area).

Clamp rule: `cot ∈ [-cap, cap]` for `cap = 1 / sin(min_angle)` —
i.e. enforces a minimum interior angle. Default `cap = 1e3`
corresponds to ~0.06° minimum angle, well below mesh-quality floors.

One thread per triangle. Writes 9 floats per triangle (row-major
3×3) just like PolygonLaplacian.

Bindings (set 0):
  0 — `ConstantBuffer<RobustLapParams> { uint num_triangles; float cot_cap; }`
  1 — `StructuredBuffer<float3> positions`
  2 — `StructuredBuffer<int> tri_indices`
  3 — `RWStructuredBuffer<float> matrices`
-/

namespace Curvenet.SlangCodegen.RobustLaplacian

open LeanSlang

private def floatTy : SlangType := .scalar .float
private def f3Ty    : SlangType := .vec .float 3
private def uintTy  : SlangType := .scalar .uint

private def clamped_cot : SlangFunctionDecl :=
  { attrs := []
  , retType := floatTy
  , name := "clamped_cot"
  , params :=
      [ ⟨"apex", f3Ty,    Semantic.none, none, none, .qIn⟩
      , ⟨"o1",   f3Ty,    Semantic.none, none, none, .qIn⟩
      , ⟨"o2",   f3Ty,    Semantic.none, none, none, .qIn⟩
      , ⟨"cap",  floatTy, Semantic.none, none, none, .qIn⟩ ]
  , body :=
      [ .declInit f3Ty "a" (.bin "-" (.var "o1") (.var "apex"))
      , .declInit f3Ty "b" (.bin "-" (.var "o2") (.var "apex"))
      , .declInit floatTy "d" (.call "dot" [.var "a", .var "b"])
      , .declInit floatTy "cl"
          (.call "length" [.call "cross" [.var "a", .var "b"]])
      , .declInit floatTy "raw"
          (.ternary (.bin "<" (.var "cl") (.litFloat 1e-30))
            (.litFloat 0.0)
            (.bin "/" (.var "d") (.var "cl")))
      , .retExpr
          (.call "clamp"
            [.var "raw", .un "-" (.var "cap"), .var "cap"]) ] }

private def setM (base i j : SlangExpr) (v : SlangExpr) : SlangStmt :=
  .assign (.index (.var "matrices")
            (.bin "+" base (.bin "+" (.bin "*" i (.litUint 3)) j)))
          v

private def mainEntry : SlangFunctionDecl :=
  { attrs  := [.shaderCompute, .numthreads 64 1 1]
  , name   := "main"
  , params := [⟨"tid", .vec .uint 3, .svDispatchThreadId, none, none, .qIn⟩]
  , body   :=
      [ .declInit uintTy "t" (.member (.var "tid") "x")
      , .ifNoElse (.bin ">=" (.var "t")
            (.member (.var "params") "num_triangles"))
          [ .ret none ]
      , .declInit floatTy "cap" (.member (.var "params") "cot_cap")
      , .declInit uintTy "i0"
          (.call "uint" [.index (.var "tri_indices")
            (.bin "*" (.var "t") (.litUint 3))])
      , .declInit uintTy "i1"
          (.call "uint" [.index (.var "tri_indices")
            (.bin "+" (.bin "*" (.var "t") (.litUint 3)) (.litUint 1))])
      , .declInit uintTy "i2"
          (.call "uint" [.index (.var "tri_indices")
            (.bin "+" (.bin "*" (.var "t") (.litUint 3)) (.litUint 2))])
      , .declInit f3Ty "a" (.index (.var "positions") (.var "i0"))
      , .declInit f3Ty "b" (.index (.var "positions") (.var "i1"))
      , .declInit f3Ty "c" (.index (.var "positions") (.var "i2"))
      , .declInit floatTy "cot_a"
          (.call "clamped_cot" [.var "a", .var "b", .var "c", .var "cap"])
      , .declInit floatTy "cot_b"
          (.call "clamped_cot" [.var "b", .var "a", .var "c", .var "cap"])
      , .declInit floatTy "cot_c"
          (.call "clamped_cot" [.var "c", .var "a", .var "b", .var "cap"])
      , .declInit floatTy "w_bc"
          (.bin "*" (.var "cot_a") (.litFloat 0.5))
      , .declInit floatTy "w_ac"
          (.bin "*" (.var "cot_b") (.litFloat 0.5))
      , .declInit floatTy "w_ab"
          (.bin "*" (.var "cot_c") (.litFloat 0.5))
      , .declInit uintTy "base"
          (.bin "*" (.var "t") (.litUint 9))
      , setM (.var "base") (.litUint 0) (.litUint 0)
          (.bin "+" (.var "w_ab") (.var "w_ac"))
      , setM (.var "base") (.litUint 0) (.litUint 1) (.un "-" (.var "w_ab"))
      , setM (.var "base") (.litUint 0) (.litUint 2) (.un "-" (.var "w_ac"))
      , setM (.var "base") (.litUint 1) (.litUint 0) (.un "-" (.var "w_ab"))
      , setM (.var "base") (.litUint 1) (.litUint 1)
          (.bin "+" (.var "w_ab") (.var "w_bc"))
      , setM (.var "base") (.litUint 1) (.litUint 2) (.un "-" (.var "w_bc"))
      , setM (.var "base") (.litUint 2) (.litUint 0) (.un "-" (.var "w_ac"))
      , setM (.var "base") (.litUint 2) (.litUint 1) (.un "-" (.var "w_bc"))
      , setM (.var "base") (.litUint 2) (.litUint 2)
          (.bin "+" (.var "w_ac") (.var "w_bc"))
      , .ret none ] }

def shader : SlangShaderModule :=
  { structs :=
      [ { name := "RobustLapParams"
        , fields :=
            [ ⟨"num_triangles", .scalar .uint,  Semantic.none, none, none, .qIn⟩
            , ⟨"cot_cap",       .scalar .float, Semantic.none, none, none, .qIn⟩ ] } ]
  , globals :=
      [ ⟨"params",      .const "RobustLapParams",    Semantic.none, some 0, some 0, .qIn⟩
      , ⟨"positions",   .roBuf f3Ty,                 Semantic.none, some 1, some 0, .qIn⟩
      , ⟨"tri_indices", .roBuf (.scalar .int),       Semantic.none, some 2, some 0, .qIn⟩
      , ⟨"matrices",    .rwBuf (.scalar .float),     Semantic.none, some 3, some 0, .qIn⟩ ]
  , functions := [clamped_cot, mainEntry] }

example : shader.entryPointName = "main" := by native_decide

end Curvenet.SlangCodegen.RobustLaplacian
