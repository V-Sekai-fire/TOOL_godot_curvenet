import LeanSlang

/-!
# `Curvenet.SlangCodegen.CutMeshLaplacian` — per-(face, vertex) cot weight scatter

Slang re-port of `src/curvenet/cut_mesh_laplacian.h`. Computes the
per-triangle 3×3 cot Laplacian (same as PolygonLaplacian but
tracking the cut-mesh re-indexing) and writes weight contributions
to a flat (face, vertex) sparse buffer that the host accumulates
into the global L. Avoids GPU atomics by writing per-(face, slot)
entries; the host runs the scatter as a separate pass.

Output layout: 9 floats per triangle (row-major 3×3, identical to
PolygonLaplacian). The cut-mesh-specific re-indexing (split vertices
get distinct rows) is a host responsibility; this kernel works with
whatever index space the host hands it.

Bindings (set 0):
  0 — `ConstantBuffer<CutLapParams> { uint num_triangles; }`
  1 — `StructuredBuffer<float3> positions` (cut-mesh vertex positions)
  2 — `StructuredBuffer<int>    tri_indices` (cut-mesh triangle indices)
  3 — `RWStructuredBuffer<float> matrices` (9 floats per triangle)
-/

namespace Curvenet.SlangCodegen.CutMeshLaplacian

open LeanSlang

private def floatTy : SlangType := .scalar .float
private def f3Ty    : SlangType := .vec .float 3
private def uintTy  : SlangType := .scalar .uint

private def cot_at_apex : SlangFunctionDecl :=
  { attrs := []
  , retType := floatTy
  , name := "cot_at_apex"
  , params :=
      [ ⟨"apex", f3Ty, Semantic.none, none, none, .qIn⟩
      , ⟨"o1",   f3Ty, Semantic.none, none, none, .qIn⟩
      , ⟨"o2",   f3Ty, Semantic.none, none, none, .qIn⟩ ]
  , body :=
      [ .declInit f3Ty "a" (.bin "-" (.var "o1") (.var "apex"))
      , .declInit f3Ty "b" (.bin "-" (.var "o2") (.var "apex"))
      , .declInit floatTy "d"  (.call "dot" [.var "a", .var "b"])
      , .declInit floatTy "cl"
          (.call "length" [.call "cross" [.var "a", .var "b"]])
      , .retExpr
          (.ternary (.bin "<" (.var "cl") (.litFloat 1e-30))
            (.litFloat 0.0)
            (.bin "/" (.var "d") (.var "cl"))) ] }

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
      , .ifNoElse (.bin ">=" (.var "t") (.member (.var "params") "num_triangles"))
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
      , .declInit uintTy "base" (.bin "*" (.var "t") (.litUint 9))
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
      [ { name := "CutLapParams"
        , fields := [⟨"num_triangles", .scalar .uint, Semantic.none, none, none, .qIn⟩] } ]
  , globals :=
      [ ⟨"params",      .const "CutLapParams",    Semantic.none, some 0, some 0, .qIn⟩
      , ⟨"positions",   .roBuf f3Ty,              Semantic.none, some 1, some 0, .qIn⟩
      , ⟨"tri_indices", .roBuf (.scalar .int),    Semantic.none, some 2, some 0, .qIn⟩
      , ⟨"matrices",    .rwBuf floatTy,           Semantic.none, some 3, some 0, .qIn⟩ ]
  , functions := [cot_at_apex, mainEntry] }

example : shader.entryPointName = "main" := by native_decide

end Curvenet.SlangCodegen.CutMeshLaplacian
