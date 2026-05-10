import LeanSlang

/-!
# `Curvenet.SlangCodegen.PolygonLaplacian` — per-triangle cot Laplacian

Slang re-port of `src/curvenet/polygon_laplacian.h` (the
`triangle_cot_laplacian` core; n-gon fan triangulation is left to
the host driver).

One thread per triangle. For each triangle (a, b, c) computes the
three cotangents at apexes a, b, c and writes the 3×3 cot Laplacian:

      [  w_ab+w_ac     -w_ab        -w_ac      ]
      [    -w_ab     w_ab+w_bc      -w_bc      ]
      [    -w_ac      -w_bc       w_ac+w_bc    ]

where `w_xy = 0.5 · cot_at_apex_opposite_to_xy`. The output matrix
buffer is 9 floats per triangle, row-major.

Bindings (set 0):
  0 — `ConstantBuffer<PolyLapParams> { uint num_triangles; }`
  1 — `StructuredBuffer<float3> positions`           (per-vertex)
  2 — `StructuredBuffer<int> tri_indices`            (3 ints per triangle, length = 3 * num_triangles)
  3 — `RWStructuredBuffer<float> matrices`           (9 floats per triangle)
-/

namespace Curvenet.SlangCodegen.PolygonLaplacian

open LeanSlang

private def floatTy : SlangType := .scalar .float
private def f3Ty    : SlangType := .vec .float 3
private def uintTy  : SlangType := .scalar .uint

/-- `cot_at_apex(apex, o1, o2) = dot(a, b) / length(cross(a, b))`
    where a = o1 - apex, b = o2 - apex. -/
private def cot_at_apex : SlangFunctionDecl :=
  { attrs   := []
  , retType := floatTy
  , name    := "cot_at_apex"
  , params  :=
      [ ⟨"apex", f3Ty, Semantic.none, none, none, .qIn⟩
      , ⟨"o1",   f3Ty, Semantic.none, none, none, .qIn⟩
      , ⟨"o2",   f3Ty, Semantic.none, none, none, .qIn⟩ ]
  , body    :=
      [ .declInit f3Ty "a" (.bin "-" (.var "o1") (.var "apex"))
      , .declInit f3Ty "b" (.bin "-" (.var "o2") (.var "apex"))
      , .declInit floatTy "d" (.call "dot" [.var "a", .var "b"])
      , .declInit floatTy "cl"
          (.call "length" [.call "cross" [.var "a", .var "b"]])
      , .retExpr (.bin "/" (.var "d") (.var "cl")) ] }

/-- Helper: write `m[i, j] = v` for the per-triangle 3×3 row-major
    matrix at base offset `base = 9 * tri`. -/
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
          (.call "cot_at_apex" [.var "a", .var "b", .var "c"])
      , .declInit floatTy "cot_b"
          (.call "cot_at_apex" [.var "b", .var "a", .var "c"])
      , .declInit floatTy "cot_c"
          (.call "cot_at_apex" [.var "c", .var "a", .var "b"])
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
      [ { name := "PolyLapParams"
        , fields :=
            [ ⟨"num_triangles", .scalar .uint, Semantic.none, none, none, .qIn⟩ ] } ]
  , globals :=
      [ ⟨"params",      .const "PolyLapParams",      Semantic.none, some 0, some 0, .qIn⟩
      , ⟨"positions",   .roBuf f3Ty,                 Semantic.none, some 1, some 0, .qIn⟩
      , ⟨"tri_indices", .roBuf (.scalar .int),       Semantic.none, some 2, some 0, .qIn⟩
      , ⟨"matrices",    .rwBuf (.scalar .float),     Semantic.none, some 3, some 0, .qIn⟩ ]
  , functions := [cot_at_apex, mainEntry] }

example : shader.entryPointName = "main" := by native_decide

end Curvenet.SlangCodegen.PolygonLaplacian
