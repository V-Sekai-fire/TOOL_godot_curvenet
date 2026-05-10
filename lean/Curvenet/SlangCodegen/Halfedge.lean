import LeanSlang

/-!
# `Curvenet.SlangCodegen.Halfedge` — per-halfedge bounds check

Slang re-port of the per-halfedge `indices_in_range` invariant from
`src/curvenet/halfedge.h`. One thread per halfedge: reads (target,
twin, next, face) and writes 1 if all four indices fit their ranges,
else 0. Host OR-reduces the results to get a global manifold flag
(or scans for the first violation).

The other manifold predicates (twin_involutive, twins_are_opposite,
face_loops_close) need cross-halfedge state and graph traversal —
they live in their own modules where the work decomposes more
naturally.

Halfedge SoA layout:
  binding 1 — `StructuredBuffer<int> targets`     (length he_count)
  binding 2 — `StructuredBuffer<int> twins`       (length he_count, -1 = none)
  binding 3 — `StructuredBuffer<int> nexts`       (length he_count)
  binding 4 — `StructuredBuffer<int> faces`       (length he_count, -1 = none)
  binding 5 — `RWStructuredBuffer<uint> ok`       (1 = in-range, 0 = violation)
-/

namespace Curvenet.SlangCodegen.Halfedge

open LeanSlang

private def floatTy : SlangType := .scalar .float
private def intTy   : SlangType := .scalar .int
private def uintTy  : SlangType := .scalar .uint

def shader : SlangShaderModule :=
  { structs :=
      [ { name := "HalfedgeParams"
        , fields :=
            [ ⟨"he_count",     .scalar .uint, Semantic.none, none, none, .qIn⟩
            , ⟨"vertex_count", .scalar .uint, Semantic.none, none, none, .qIn⟩
            , ⟨"face_count",   .scalar .uint, Semantic.none, none, none, .qIn⟩ ] } ]
  , globals :=
      [ ⟨"params",  .const "HalfedgeParams", Semantic.none, some 0, some 0, .qIn⟩
      , ⟨"targets", .roBuf intTy,            Semantic.none, some 1, some 0, .qIn⟩
      , ⟨"twins",   .roBuf intTy,            Semantic.none, some 2, some 0, .qIn⟩
      , ⟨"nexts",   .roBuf intTy,            Semantic.none, some 3, some 0, .qIn⟩
      , ⟨"faces",   .roBuf intTy,            Semantic.none, some 4, some 0, .qIn⟩
      , ⟨"ok",      .rwBuf uintTy,           Semantic.none, some 5, some 0, .qIn⟩ ]
  , functions := [{
      attrs  := [.shaderCompute, .numthreads 256 1 1]
      name   := "main"
      params := [⟨"tid", .vec .uint 3, .svDispatchThreadId, none, none, .qIn⟩]
      body   :=
        [ .declInit uintTy "i" (.member (.var "tid") "x")
        , .ifNoElse (.bin ">=" (.var "i") (.member (.var "params") "he_count"))
            [ .ret none ]
        , .declInit intTy "tgt" (.index (.var "targets") (.var "i"))
        , .declInit intTy "tw"  (.index (.var "twins")   (.var "i"))
        , .declInit intTy "nx"  (.index (.var "nexts")   (.var "i"))
        , .declInit intTy "fc"  (.index (.var "faces")   (.var "i"))
        , .declInit uintTy "valid" (.litUint 1)
        , -- target in [0, vertex_count)
          .ifNoElse
            (.bin "||" (.bin "<" (.var "tgt") (.litUint 0))
                       (.bin ">=" (.call "uint" [.var "tgt"])
                                  (.member (.var "params") "vertex_count")))
            [ .assign (.var "valid") (.litUint 0) ]
        , -- next in [0, he_count)
          .ifNoElse
            (.bin "||" (.bin "<" (.var "nx") (.litUint 0))
                       (.bin ">=" (.call "uint" [.var "nx"])
                                  (.member (.var "params") "he_count")))
            [ .assign (.var "valid") (.litUint 0) ]
        , -- twin: -1 (none) OK; otherwise in [0, he_count)
          .ifNoElse
            (.bin "&&" (.bin ">=" (.var "tw") (.litUint 0))
                       (.bin ">=" (.call "uint" [.var "tw"])
                                  (.member (.var "params") "he_count")))
            [ .assign (.var "valid") (.litUint 0) ]
        , -- face: -1 (none) OK; otherwise in [0, face_count)
          .ifNoElse
            (.bin "&&" (.bin ">=" (.var "fc") (.litUint 0))
                       (.bin ">=" (.call "uint" [.var "fc"])
                                  (.member (.var "params") "face_count")))
            [ .assign (.var "valid") (.litUint 0) ]
        , .assign (.index (.var "ok") (.var "i")) (.var "valid")
        , .ret none ] }] }

example : shader.entryPointName = "main" := by native_decide

end Curvenet.SlangCodegen.Halfedge
