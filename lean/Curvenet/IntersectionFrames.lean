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

/-- Corner normals `m_i = c_i / |c_i|`. Diverges if |c_i| = 0 (parallel
   consecutive tangents — see file docstring). -/
def cornerNormals (segs : Array OutgoingSegment) : Array Vec3 := Id.run do
  let cs := cornerVectors segs
  let n := cs.size
  let mut acc : Array Vec3 := Array.replicate n ⟨0.0, 0.0, 0.0⟩
  for i in [0:n] do
    let c := cs[i]!
    let nrm := vec3Norm c
    let inv := if nrm == 0.0 then 0.0 else 1.0 / nrm
    acc := acc.set! i (vec3Scale inv c)
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

end IntersectionFramesExamples

end Curvenet
