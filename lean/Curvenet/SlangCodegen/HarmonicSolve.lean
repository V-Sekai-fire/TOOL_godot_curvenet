import LeanSlang

/-!
# `Curvenet.SlangCodegen.HarmonicSolve` — per-vertex harmonic residual

Slang re-port of the harmonic-solve setup in
`src/curvenet/harmonic_solve.h`. The full §4.3 Stage 1 solve is
many CG iterations of `L · phi = b` orchestrated by the host
(dispatch SpMV → DotReduce → SAXPBY → repeat). This kernel models
the per-vertex residual computation that bookends each iteration:
  r[i] = b[i] - (L · phi)[i]

The harmonic L matrix is the cot Laplacian assembled by
PolygonLaplacian / RobustLaplacian / CutMeshLaplacian.

Bindings (set 0):
  0 — `ConstantBuffer<HarmonicParams> { uint n; }`
  1-3 — CSR L: rowPtr, colIdx, values
  4 — `StructuredBuffer<float> phi`   (current iterate)
  5 — `StructuredBuffer<float> b`     (RHS)
  6 — `RWStructuredBuffer<float> r`   (residual output)
-/

namespace Curvenet.SlangCodegen.HarmonicSolve

open LeanSlang

private def floatTy : SlangType := .scalar .float
private def intTy   : SlangType := .scalar .int
private def uintTy  : SlangType := .scalar .uint

def shader : SlangShaderModule :=
  { structs :=
      [ { name := "HarmonicParams"
        , fields := [⟨"n", .scalar .uint, Semantic.none, none, none, .qIn⟩] } ]
  , globals :=
      [ ⟨"params", .const "HarmonicParams",  Semantic.none, some 0, some 0, .qIn⟩
      , ⟨"rowPtr", .roBuf intTy,             Semantic.none, some 1, some 0, .qIn⟩
      , ⟨"colIdx", .roBuf intTy,             Semantic.none, some 2, some 0, .qIn⟩
      , ⟨"values", .roBuf floatTy,           Semantic.none, some 3, some 0, .qIn⟩
      , ⟨"phi",    .roBuf floatTy,           Semantic.none, some 4, some 0, .qIn⟩
      , ⟨"b",      .roBuf floatTy,           Semantic.none, some 5, some 0, .qIn⟩
      , ⟨"r",      .rwBuf floatTy,           Semantic.none, some 6, some 0, .qIn⟩ ]
  , functions := [{
      attrs  := [.shaderCompute, .numthreads 256 1 1]
      name   := "main"
      params := [⟨"tid", .vec .uint 3, .svDispatchThreadId, none, none, .qIn⟩]
      body   :=
        [ .declInit uintTy "i" (.member (.var "tid") "x")
        , .ifNoElse (.bin ">=" (.var "i") (.member (.var "params") "n"))
            [ .ret none ]
        , .declInit uintTy "rs"
            (.call "uint" [.index (.var "rowPtr") (.var "i")])
        , .declInit uintTy "re"
            (.call "uint" [.index (.var "rowPtr")
              (.bin "+" (.var "i") (.litUint 1))])
        , .declInit floatTy "Lphi" (.litFloat 0.0)
        , .forCount "p" (.var "rs") (.var "re")
            [ .assign (.var "Lphi")
                (.call "fma"
                  [ .index (.var "values") (.var "p")
                  , .index (.var "phi")
                      (.call "uint" [.index (.var "colIdx") (.var "p")])
                  , .var "Lphi" ]) ]
        , .assign (.index (.var "r") (.var "i"))
            (.bin "-" (.index (.var "b") (.var "i")) (.var "Lphi"))
        , .ret none ] }] }

example : shader.entryPointName = "main" := by native_decide

end Curvenet.SlangCodegen.HarmonicSolve
