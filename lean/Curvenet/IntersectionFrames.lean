/-
MIT License — Copyright (c) 2026 K. S. Ernest (iFire) Lee.

Per-side scaled frames at curvenet intersections — slice 13 of the
DeGoes22 rewrite.

DeGoes22 §3 ("Normal & width at intersections"):
  Given the ordered list of outgoing tangents `(t_i, l_i)` at an
  intersection, define for each pair of consecutive tangents the
  corner vector

      c_i = t_i × t_{i+1}       (cross product)
      m_i = c_i / |c_i|         (corner normal)

  Each segment outgoing the intersection then has two sides:

      n_i^+ = m_i           (left-side normal)
      n_i^- = m_{i-1}       (right-side normal)

  The corresponding widths are

      w_i^+ = l_i + |c_i|     (l_{i+1} − l_i)
      w_i^- = l_i + |c_{i-1}| (l_{i-1} − l_i)

This file formalizes the corner-normal + per-side-width computation
exactly as written. The slice-4 isolated-segment formulation handles
samples not adjacent to any intersection; slice 9 used identity scaled
frames in tests. A future slice will weave these per-side frames into
the deformation gradient `F_i = (B_i S_i)(B̆_i S̆_i)⁻¹` for the full §3
reduction.

T-junction degeneracy (parallel consecutive tangents, |c_i| = 0) is
explicitly NOT handled here — the paper applies a special-case fallback
`m_i = (c_{i+1} + c_{i−1}) / |c_{i+1} + c_{i−1}|`. Adding that fallback
is left as a follow-up; tests in this slice avoid the degenerate case.
-/

import Curvenet.Vec3

namespace Curvenet
namespace IntersectionFrames

/-- Outgoing segment from an intersection: a unit tangent and a length. -/
structure OutgoingSegment where
  tangent : Vec3
  length  : Float
deriving Repr, Inhabited

@[inline] def cross (a b : Vec3) : Vec3 :=
  ⟨ a.y * b.z - a.z * b.y
  , a.z * b.x - a.x * b.z
  , a.x * b.y - a.y * b.x ⟩

@[inline] def vec3Norm (v : Vec3) : Float :=
  (v.x * v.x + v.y * v.y + v.z * v.z).sqrt

@[inline] def vec3Scale (s : Float) (v : Vec3) : Vec3 :=
  ⟨ s * v.x, s * v.y, s * v.z ⟩

/-- Corner vectors `c_i = t_i × t_{i+1}` for an intersection's CCW-ordered
   outgoing tangents (cyclic). Output length matches the input length. -/
def cornerVectors (segs : Array OutgoingSegment) : Array Vec3 := Id.run do
  let n := segs.size
  let mut acc : Array Vec3 := Array.replicate n ⟨0.0, 0.0, 0.0⟩
  for i in [0:n] do
    let ti  := segs[i]!.tangent
    let tiP := segs[(i + 1) % n]!.tangent
    acc := acc.set! i (cross ti tiP)
  return acc

/-- Eps below which a corner-vector magnitude triggers the §3 T-junction
   fallback instead of the direct `c_i / |c_i|` formula. -/
private def degenerateEps : Float := 1.0e-12

/-- Corner normals `m_i = c_i / |c_i|`, with the §3 T-junction fallback
   `m_i = (c_{i+1} + c_{i−1}) / |c_{i+1} + c_{i−1}|` when |c_i| ≤ eps
   (parallel consecutive tangents). If the fallback denominator is also
   zero (degenerate triple-collinear configuration), returns the zero
   vector at that index — an explicit signal of unrecoverable
   degeneracy. -/
def cornerNormals (segs : Array OutgoingSegment) : Array Vec3 := Id.run do
  let cs := cornerVectors segs
  let n := cs.size
  let mut acc : Array Vec3 := Array.replicate n ⟨0.0, 0.0, 0.0⟩
  for i in [0:n] do
    let c := cs[i]!
    let nrm := vec3Norm c
    if nrm > degenerateEps then
      acc := acc.set! i (vec3Scale (1.0 / nrm) c)
    else
      let cP := cs[(i + 1) % n]!
      let cM := cs[(i + n - 1) % n]!
      let sum : Vec3 := ⟨ cP.x + cM.x, cP.y + cM.y, cP.z + cM.z ⟩
      let sNorm := vec3Norm sum
      if sNorm > degenerateEps then
        acc := acc.set! i (vec3Scale (1.0 / sNorm) sum)
      -- else leave zero
  return acc

/-- Per-side normals for each outgoing segment.
   For segment `i`: `(n_i^+, n_i^-) = (m_i, m_{i−1})`. -/
def perSideNormals (segs : Array OutgoingSegment) :
    Array (Vec3 × Vec3) := Id.run do
  let ms := cornerNormals segs
  let n := ms.size
  let mut acc : Array (Vec3 × Vec3) := Array.replicate n (⟨0.0, 0.0, 0.0⟩, ⟨0.0, 0.0, 0.0⟩)
  for i in [0:n] do
    let mi   := ms[i]!
    let miM  := ms[(i + n - 1) % n]!
    acc := acc.set! i (mi, miM)
  return acc

/-- Per-side widths for each outgoing segment.
   `w_i^+ = l_i + |c_i| (l_{i+1} − l_i)`,
   `w_i^- = l_i + |c_{i−1}| (l_{i−1} − l_i)`. -/
def perSideWidths (segs : Array OutgoingSegment) :
    Array (Float × Float) := Id.run do
  let cs := cornerVectors segs
  let n := segs.size
  let mut acc : Array (Float × Float) := Array.replicate n (0.0, 0.0)
  for i in [0:n] do
    let li   := segs[i]!.length
    let liP  := segs[(i + 1) % n]!.length
    let liM  := segs[(i + n - 1) % n]!.length
    let cMag  := vec3Norm cs[i]!
    let cMagM := vec3Norm cs[(i + n - 1) % n]!
    let wPlus  := li + cMag  * (liP - li)
    let wMinus := li + cMagM * (liM - li)
    acc := acc.set! i (wPlus, wMinus)
  return acc

/-- 3×3 row-major matrix stored as a 9-element Float array. -/
abbrev Mat3 := Array Float

/-- DeGoes22 §3 per-side scaled frame
       B_i S_i = [ t_i · l_i  |  b_i · w_i  |  n_i · h_i ]
   where
       b_i = n_i × t_i        (binormal, right-handed)
       h_i = sqrt(l_i · w_i)  (geometric mean of length and width)

   Columns of the resulting 3×3 are scaled basis vectors; rows are
   stored consecutively (row-major).

   At rest tangent + identity normal + unit length + unit width, the
   scaled frame is the identity matrix. -/
def scaledFrame (t n : Vec3) (l w : Float) : Mat3 :=
  let binX := n.y * t.z - n.z * t.y
  let binY := n.z * t.x - n.x * t.z
  let binZ := n.x * t.y - n.y * t.x
  let h    := (l * w).sqrt
  #[ t.x * l, binX * w, n.x * h
   , t.y * l, binY * w, n.y * h
   , t.z * l, binZ * w, n.z * h ]

/-- Per-side scaled frames for every outgoing segment of an intersection.
   Output array element `i` is the pair (B_i^+ · S_i^+, B_i^- · S_i^-). -/
def perSideScaledFrames (segs : Array OutgoingSegment) :
    Array (Mat3 × Mat3) := Id.run do
  let n       := segs.size
  let normals := perSideNormals segs
  let widths  := perSideWidths  segs
  let mut acc : Array (Mat3 × Mat3) :=
    Array.replicate n (#[], #[])
  for i in [0:n] do
    let t := segs[i]!.tangent
    let l := segs[i]!.length
    let (np, nm) := normals[i]!
    let (wp, wm) := widths[i]!
    acc := acc.set! i (scaledFrame t np l wp, scaledFrame t nm l wm)
  return acc

end IntersectionFrames

/- ============================================================ -/
/- Concrete intersections.                                      -/
/- ============================================================ -/

namespace IntersectionFramesExamples

open IntersectionFrames

/-- Two perpendicular tangents in the XY plane: t_0 = +x, t_1 = +y. -/
private def perpendicular : Array OutgoingSegment :=
  #[ ⟨⟨1.0, 0.0, 0.0⟩, 1.0⟩
   , ⟨⟨0.0, 1.0, 0.0⟩, 1.0⟩ ]

/-- For a 2-tangent intersection, c_0 = t_0 × t_1 = +z, c_1 = t_1 × t_0 = −z.
   Corner normals are unit vectors along ±z. -/
example :
    let ms := cornerNormals perpendicular
    let m0 := ms[0]!
    let m1 := ms[1]!
    ((m0.x).abs < 1e-12 ∧ (m0.y).abs < 1e-12 ∧ (m0.z - 1.0).abs < 1e-12 ∧
     (m1.x).abs < 1e-12 ∧ (m1.y).abs < 1e-12 ∧ (m1.z + 1.0).abs < 1e-12) := by
  native_decide

/-- Per-side normals: segment 0's left side is m_0 = +z; right side is
   m_{-1} = m_1 = −z. -/
example :
    let pairs := perSideNormals perpendicular
    let (np0, nm0) := pairs[0]!
    ((np0.z - 1.0).abs < 1e-12 ∧ (nm0.z + 1.0).abs < 1e-12) := by
  native_decide

/-- T-junction-style three tangents at 120° in the XY plane: a
   non-degenerate intersection where every corner normal points along
   +z (CCW configuration). -/
private def triJunction : Array OutgoingSegment :=
  let s : Float := 0.8660254037844386 -- sin 60°
  let c : Float := 0.5                 -- cos 60° = 0.5
  #[ ⟨⟨1.0, 0.0, 0.0⟩, 1.0⟩
   , ⟨⟨-c, s, 0.0⟩, 1.0⟩
   , ⟨⟨-c, -s, 0.0⟩, 1.0⟩ ]

example :
    let ms := cornerNormals triJunction
    -- All three corner normals point along +z.
    ((ms[0]!.z - 1.0).abs < 1e-12 ∧
     (ms[1]!.z - 1.0).abs < 1e-12 ∧
     (ms[2]!.z - 1.0).abs < 1e-12) := by
  native_decide

/-- For uniform segment lengths (all = 1) the width formula collapses to
   `w_i^± = l_i = 1` regardless of the corner geometry, since
   (l_{i±1} − l_i) = 0. -/
example :
    let widths := perSideWidths triJunction
    let (wp0, wm0) := widths[0]!
    ((wp0 - 1.0).abs < 1e-12 ∧ (wm0 - 1.0).abs < 1e-12) := by
  native_decide

/-- Anisotropic widths kick in when adjacent lengths differ.
   Two perpendicular tangents with l_0 = 1, l_1 = 3:
   w_0^+ = 1 + 1·(3 − 1) = 3,  w_0^- = 1 + 1·(3 − 1) = 3 (cyclic
   pre-segment is segment 1 itself in a 2-arity intersection). -/
private def anisoPerp : Array OutgoingSegment :=
  #[ ⟨⟨1.0, 0.0, 0.0⟩, 1.0⟩
   , ⟨⟨0.0, 1.0, 0.0⟩, 3.0⟩ ]

example :
    let widths := perSideWidths anisoPerp
    let (wp0, wm0) := widths[0]!
    ((wp0 - 3.0).abs < 1e-12 ∧ (wm0 - 3.0).abs < 1e-12) := by
  native_decide

/- ============================================================ -/
/- Scaled-frame matrix B_i S_i.                                 -/
/- ============================================================ -/

private def matCloseToIdentity (m : Mat3) (eps : Float) : Bool := Id.run do
  let id : Mat3 := #[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
  for k in [0:9] do
    if (m[k]! - id[k]!).abs ≥ eps then return false
  return true

/-- Identity-like input: t = +x, n = +z, l = w = 1 ⇒ B S = identity. -/
example :
    matCloseToIdentity (scaledFrame ⟨1.0, 0.0, 0.0⟩ ⟨0.0, 0.0, 1.0⟩ 1.0 1.0) 1e-12 = true := by
  native_decide

/-- Anisotropic: l = 2, w = 3 stretches columns 0 and 1 only.
   B S column 0 = (2, 0, 0), column 1 = (0, 3, 0), column 2 = (0, 0, sqrt(6)). -/
example :
    let M := scaledFrame ⟨1.0, 0.0, 0.0⟩ ⟨0.0, 0.0, 1.0⟩ 2.0 3.0
    let h := (2.0 * 3.0).sqrt
    -- Row 0: (2, 0, 0). Row 1: (0, 3, 0). Row 2: (0, 0, h).
    ((M[0]! - 2.0).abs < 1e-12 ∧ M[1]!.abs < 1e-12 ∧ M[2]!.abs < 1e-12 ∧
     M[3]!.abs < 1e-12 ∧ (M[4]! - 3.0).abs < 1e-12 ∧ M[5]!.abs < 1e-12 ∧
     M[6]!.abs < 1e-12 ∧ M[7]!.abs < 1e-12 ∧ (M[8]! - h).abs < 1e-12) := by
  native_decide

/-- Per-side scaled frames at the perpendicular intersection: segment 0
   (t = +x, l = 1) with both sides having unit width and ±z normals
   gives identity-with-z-flip on one side, identity on the other. -/
example :
    let frames := perSideScaledFrames perpendicular
    let (Bp, Bm) := frames[0]!
    -- Both B^+ and B^- have row 0 column 0 = 1 (t·l), row 2 col 2 = ±1 (n·h).
    ((Bp[0]! - 1.0).abs < 1e-12 ∧ (Bp[8]! - 1.0).abs < 1e-12 ∧
     (Bm[0]! - 1.0).abs < 1e-12 ∧ (Bm[8]! + 1.0).abs < 1e-12) := by
  native_decide

/- ============================================================ -/
/- T-junction degenerate fallback: at indices where consecutive  -/
/- tangents are parallel, |c_i| = 0; the §3 fallback              -/
/- m_i = (c_{i+1} + c_{i−1}) / |·| keeps the corner normal       -/
/- well-defined and unit-length.                                  -/
/- ============================================================ -/

/-- T-junction shape: t_0 = +x and t_2 = −x are anti-parallel
   (`c_2 = t_2 × t_0 = −x × +x = 0`); t_1 = +y branches off. The
   degenerate corner is index 2; its fallback uses c_0 + c_1 = +z + +z
   = 2·+z, normalised to +z. -/
private def tJunction : Array OutgoingSegment :=
  #[ ⟨⟨1.0, 0.0, 0.0⟩, 1.0⟩
   , ⟨⟨0.0, 1.0, 0.0⟩, 1.0⟩
   , ⟨⟨-1.0, 0.0, 0.0⟩, 1.0⟩ ]

/-- |c_2| = 0 in the T-junction; the direct formula would NaN, but the
   fallback keeps the normal at +z. -/
example :
    let cs := cornerVectors tJunction
    -- c_2 = t_2 × t_0 = (−1,0,0) × (1,0,0) = (0,0,0).
    cs[2]!.x.abs < 1e-12 ∧ cs[2]!.y.abs < 1e-12 ∧ cs[2]!.z.abs < 1e-12 := by
  native_decide

/-- All three corner normals come out as +z (the fallback at index 2
   matches the directly-computed normals at indices 0 and 1). -/
example :
    let ms := cornerNormals tJunction
    ((ms[0]!.z - 1.0).abs < 1e-12 ∧
     (ms[1]!.z - 1.0).abs < 1e-12 ∧
     (ms[2]!.z - 1.0).abs < 1e-12) := by
  native_decide

/-- The fallback returns a unit vector. -/
example :
    let ms := cornerNormals tJunction
    let m  := ms[2]!
    let nrm := vec3Norm m
    (nrm - 1.0).abs < 1e-12 := by
  native_decide

-- Reflection / symmetry invariants on per-side data.

/-- At a perpendicular 2-tangent intersection, every segment's plus-side
   normal is the reflection of its minus-side normal across the segment's
   tangent plane. -/
example :
    let pairs := perSideNormals perpendicular
    let (np0, nm0) := pairs[0]!
    let (np1, nm1) := pairs[1]!
    ((np0.x + nm0.x).abs < 1e-12 ∧
     (np0.y + nm0.y).abs < 1e-12 ∧
     (np0.z + nm0.z).abs < 1e-12 ∧
     (np1.x + nm1.x).abs < 1e-12 ∧
     (np1.y + nm1.y).abs < 1e-12 ∧
     (np1.z + nm1.z).abs < 1e-12) := by
  native_decide

/-- At the symmetric triJunction (3 unit segments), per-side widths are
   equal across plus/minus for every segment under uniform lengths. -/
example :
    let widths := perSideWidths triJunction
    let (wp0, wm0) := widths[0]!
    let (wp1, wm1) := widths[1]!
    let (wp2, wm2) := widths[2]!
    ((wp0 - wm0).abs < 1e-12 ∧
     (wp1 - wm1).abs < 1e-12 ∧
     (wp2 - wm2).abs < 1e-12) := by
  native_decide

/-- scaledFrame is homogeneous when l = w. -/
example :
    let unitF  := scaledFrame ⟨1.0, 0.0, 0.0⟩ ⟨0.0, 0.0, 1.0⟩ 1.0 1.0
    let scaled := scaledFrame ⟨1.0, 0.0, 0.0⟩ ⟨0.0, 0.0, 1.0⟩ 2.0 2.0
    ((scaled[0]! - 2.0 * unitF[0]!).abs < 1e-12 ∧
     (scaled[1]! - 2.0 * unitF[1]!).abs < 1e-12 ∧
     (scaled[2]! - 2.0 * unitF[2]!).abs < 1e-12 ∧
     (scaled[3]! - 2.0 * unitF[3]!).abs < 1e-12 ∧
     (scaled[4]! - 2.0 * unitF[4]!).abs < 1e-12 ∧
     (scaled[5]! - 2.0 * unitF[5]!).abs < 1e-12 ∧
     (scaled[6]! - 2.0 * unitF[6]!).abs < 1e-12 ∧
     (scaled[7]! - 2.0 * unitF[7]!).abs < 1e-12 ∧
     (scaled[8]! - 2.0 * unitF[8]!).abs < 1e-12) := by
  native_decide

end IntersectionFramesExamples

end Curvenet
