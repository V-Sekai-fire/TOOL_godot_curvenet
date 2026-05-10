/-!
# `Curvenet.Common` — shared algorithm-side types + helpers

Tiny abbreviations and predicates that were previously copy-pasted
across module-private namespaces (`fclose` in 4 modules, `Mat3` in
3, `Adjacency` in 2). Mirrors the dedup pattern that
`Curvenet.SlangCodegen.Common` applied to the codegen side.

Each consumer `import`s + `open`s `Curvenet.Common` and drops its
own local copy.
-/

namespace Curvenet.Common

/-- Tolerance-aware Float compare: `|a - b| < eps`. -/
@[inline] def fclose (a b eps : Float) : Bool := (a - b).abs < eps

/-- Dense small matrix as a row-major Float array. Used for §3 frame
    matrices (3×3) in `ScaledFrames` / `IntersectionFrames` and for
    the IC(0) factorisation working block in `IncompleteCholesky`. -/
abbrev Mat3 := Array Float

/-- Adjacency-list graph: `nbrs[v]` is the neighbour set of vertex
    `v`. Used by `GraphColoring` and `DirectDeltaMushBind`. -/
abbrev Adjacency := Array (Array Nat)

end Curvenet.Common
