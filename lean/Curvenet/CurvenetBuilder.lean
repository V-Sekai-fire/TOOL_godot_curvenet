/-
MIT License — Copyright (c) 2026 K. S. Ernest (iFire) Lee.

Curvenet graph builder. Mirrors `src/curvenet/curvenet_builder.h`.

Given a list of curves (each curve is a polyline of `Vec3` knots), we

  1. ε-merge knot positions across curves so shared endpoints (or
     near-coincident interior knots) become a single graph knot;
  2. record incidence: for each merged knot, which (curve_id, knot_idx)
     entries collapsed onto it;
  3. classify each merged knot per DeGoes22 §3:
       anchor       — 1 incident segment
       regular      — 2 incident segments
       intersection — ≥3 incident segments

Tangent extraction is omitted from the Lean spec — it relies on
non-decidable Float arithmetic (`length`, division) and the structural
properties we want to verify don't depend on the unit normalisation.
-/

import Curvenet.Vec3

namespace Curvenet
namespace CurvenetBuilder

inductive KnotKind
  | anchor
  | regular
  | intersection
deriving Repr, Inhabited, BEq, DecidableEq

structure KnotRef where
  curveId  : Nat
  knotIdx  : Nat
deriving Repr, Inhabited, BEq

structure CurvenetGraph where
  knotPositions : Array Vec3
  /-- For each input curve, the merged-knot indices for its knots in
     order along the curve. -/
  curves        : Array (Array Nat)
  /-- For each merged knot, the incidence list (curve, knot-index pairs
     that collapsed onto it). -/
  incidence     : Array (Array KnotRef)
deriving Repr, Inhabited

/-- Square-distance between two `Vec3`s. -/
def distSq (a b : Vec3) : Float :=
  let dx := a.x - b.x
  let dy := a.y - b.y
  let dz := a.z - b.z
  dx * dx + dy * dy + dz * dz

/-- Build the curvenet graph by ε-merging knot positions across curves. -/
def build (curvesIn : Array (Array Vec3)) (eps : Float) : CurvenetGraph := Id.run do
  let epsSq := eps * eps
  let mut knotPositions : Array Vec3 := #[]
  let mut curves : Array (Array Nat) := #[]
  let mut incidence : Array (Array KnotRef) := #[]
  for c in [0:curvesIn.size] do
    let curve := curvesIn[c]!
    let mut mergedRow : Array Nat := #[]
    for k in [0:curve.size] do
      let p := curve[k]!
      let mut found : Option Nat := none
      for q in [0:knotPositions.size] do
        if distSq knotPositions[q]! p ≤ epsSq then
          found := some q
          break
      let q : Nat ←
        match found with
        | some q => pure q
        | none => do
            let qi := knotPositions.size
            knotPositions := knotPositions.push p
            incidence := incidence.push #[]
            pure qi
      mergedRow := mergedRow.push q
      let inc := incidence[q]!
      incidence := incidence.set! q (inc.push ⟨c, k⟩)
    curves := curves.push mergedRow
  return ⟨knotPositions, curves, incidence⟩

/-- Number of segment endpoints incident to merged knot `k`. -/
def incidentSegmentCount (g : CurvenetGraph) (k : Nat) : Nat := Id.run do
  let mut n : Nat := 0
  let inc := g.incidence[k]!
  for r in inc do
    let cn := g.curves[r.curveId]!.size
    if r.knotIdx > 0 then n := n + 1
    if r.knotIdx + 1 < cn then n := n + 1
  return n

/-- Classify each merged knot as anchor / regular / intersection. -/
def classify (g : CurvenetGraph) : Array KnotKind := Id.run do
  let mut out : Array KnotKind := Array.mkEmpty g.knotPositions.size
  for k in [0:g.knotPositions.size] do
    let n := incidentSegmentCount g k
    if n ≥ 3 then
      out := out.push KnotKind.intersection
    else if n == 1 then
      out := out.push KnotKind.anchor
    else
      out := out.push KnotKind.regular
  return out

end CurvenetBuilder

/- ============================================================ -/
/- Concrete cases.                                              -/
/- ============================================================ -/

namespace CurvenetBuilderExamples

open CurvenetBuilder

private def A : Vec3 := ⟨0.0, 0.0, 0.0⟩
private def B : Vec3 := ⟨1.0, 0.0, 0.0⟩
private def C : Vec3 := ⟨2.0, 0.0, 0.0⟩
private def Bp : Vec3 := ⟨1.0 + 1e-10, 0.0, 0.0⟩  -- B within ε
private def Bfar : Vec3 := ⟨1.0 + 1e-3, 0.0, 0.0⟩  -- B beyond ε
private def origin : Vec3 := ⟨0.0, 0.0, 0.0⟩
private def xUnit : Vec3 := ⟨1.0, 0.0, 0.0⟩
private def yUnit : Vec3 := ⟨0.0, 1.0, 0.0⟩
private def negXUnit : Vec3 := ⟨-1.0, 0.0, 0.0⟩
private def negYUnit : Vec3 := ⟨0.0, -1.0, 0.0⟩

/-- Two-knot open curve A→B: 2 knots, both anchors, 1 segment. -/
example :
    let g := build #[#[A, B]] 1e-6
    g.knotPositions.size = 2 := by native_decide

example :
    let g := build #[#[A, B]] 1e-6
    classify g = #[KnotKind.anchor, KnotKind.anchor] := by native_decide

/-- Two curves sharing endpoint B (within ε): 3 distinct knots, B is
   regular (deg=2 — one segment ending, one starting). -/
example :
    let g := build #[#[A, B], #[Bp, C]] 1e-6
    g.knotPositions.size = 3 := by native_decide

example :
    let g := build #[#[A, B], #[Bp, C]] 1e-6
    classify g = #[KnotKind.anchor, KnotKind.regular, KnotKind.anchor]
    := by native_decide

/-- Two curves with endpoints separated by 1e-3 > ε = 1e-6: no merge,
   4 distinct knots, all anchors. -/
example :
    let g := build #[#[A, B], #[Bfar, C]] 1e-6
    g.knotPositions.size = 4 := by native_decide

example :
    let g := build #[#[A, B], #[Bfar, C]] 1e-6
    classify g = #[KnotKind.anchor, KnotKind.anchor, KnotKind.anchor, KnotKind.anchor]
    := by native_decide

/-- "+"-pattern: two curves crossing at the origin. Curve 0 goes
   (-x → origin → +x); curve 1 goes (-y → origin → +y). The origin is
   shared by both curves (incidence list length 2), and each curve has
   2 segments at the origin → total 4 incident segments → intersection. -/
example :
    let g := build #[#[negXUnit, origin, xUnit], #[negYUnit, origin, yUnit]] 1e-6
    g.knotPositions.size = 5 := by native_decide

example :
    let g := build #[#[negXUnit, origin, xUnit], #[negYUnit, origin, yUnit]] 1e-6
    -- Knots 0..3 are the 4 endpoints (anchors); knot 4 (origin) is the
    -- intersection. Build emits the origin first, then one curve's tail
    -- (xUnit), then negYUnit, then yUnit. So the actual order is:
    --   0: negXUnit (anchor)
    --   1: origin   (intersection)
    --   2: xUnit    (anchor)
    --   3: negYUnit (anchor)
    --   4: yUnit    (anchor)
    classify g = #[KnotKind.anchor, KnotKind.intersection, KnotKind.anchor,
                    KnotKind.anchor, KnotKind.anchor] := by native_decide

/-- Single 3-knot curve A→B→C: B is interior of one curve (deg=2 →
   regular), A and C are anchors. -/
example :
    let g := build #[#[A, B, C]] 1e-6
    classify g = #[KnotKind.anchor, KnotKind.regular, KnotKind.anchor]
    := by native_decide

/-- Knot count after ε-merge equals the number of distinct positions. -/
example :
    let g := build #[#[A, B], #[Bp, C]] 1e-6
    g.knotPositions.size = 3 := by native_decide

end CurvenetBuilderExamples

end Curvenet
