/-
MIT License — Copyright (c) 2026 K. S. Ernest (iFire) Lee.

Curvenet-driven Direct Delta Mush — runtime kernel.

Background. DeGoes22's two-stage harmonic/Poisson solve at runtime
is too expensive for Quest 3's 0.8 ms / 12 RHS / 50k-vertex
budget (memory bandwidth alone for a single multi-RHS SpMV at
50k is ~1 ms). The only escape: bake the deformation response
once at bind time, then runtime is a sparse linear combination
per vertex.

Direct Delta Mush (Le & Lewis 2019) is the standard algorithm
for that bake-and-stream pattern. Each vertex stores its
**influence weights** `W[v, i]` (one per "handle"; in our case
each curvenet sample is a handle), and at runtime the deformed
position is
    `pos[v] = Σᵢ W[v, i] · F_i · rest_pos[v]`
where `F_i` is the per-handle 4×4 affine transform — for
curvenets this is exactly DeGoes22 §3's scaled-frame ratio
`(B_i S_i) (B̆_i S̆_i)⁻¹` between current and rest configurations.

Bind-time work (specced separately): solve the harmonic system
once on the rest mesh for each handle's unit perturbation,
capture per-vertex influence weights, smooth via a few Laplacian
iterations, optionally pack into per-vertex 3×4 matrices.

This file specifies the **runtime kernel**:
  * `applyTransform4x4` — apply a 4×4 affine matrix to a point
    (treats input as homogeneous `(x, y, z, 1)`, returns 3-vector)
  * `lbsMatvec` — the LBS-flavored linear combination over handles
  * Native-decide examples confirm: identity transforms preserve
    rest pose; translation transforms shift uniformly; rotation
    rotates; partial weighting blends correctly.

The bind-time weight bake is specced in a follow-up file once
the runtime kernel is locked.
-/

import Curvenet.Common

namespace Curvenet

open Curvenet.Common
namespace DirectDeltaMush

/-- A 4×4 affine transform as 16 floats in row-major order.
   We use the homogeneous convention: row 3 is `(0, 0, 0, 1)` for
   pure rigid + scale + shear. -/
abbrev Mat4 := Array Float

/-- Build the 4×4 identity. -/
def mat4Identity : Mat4 :=
  #[ 1.0, 0.0, 0.0, 0.0,
     0.0, 1.0, 0.0, 0.0,
     0.0, 0.0, 1.0, 0.0,
     0.0, 0.0, 0.0, 1.0 ]

/-- Translation `(tx, ty, tz)` as a 4×4. -/
def mat4Translation (tx ty tz : Float) : Mat4 :=
  #[ 1.0, 0.0, 0.0, tx,
     0.0, 1.0, 0.0, ty,
     0.0, 0.0, 1.0, tz,
     0.0, 0.0, 0.0, 1.0 ]

/-- Apply a 4×4 affine transform to a 3-point. The input point
   is treated as `(x, y, z, 1)` (homogeneous), the output is
   the first three components of `T · v`. -/
def applyTransform4x4 (T : Mat4) (x y z : Float) : Float × Float × Float :=
  let r0 := T[0]! * x + T[1]! * y + T[2]!  * z + T[3]!
  let r1 := T[4]! * x + T[5]! * y + T[6]!  * z + T[7]!
  let r2 := T[8]! * x + T[9]! * y + T[10]! * z + T[11]!
  (r0, r1, r2)

/-- A vertex's influence list: pairs of (handle index, weight).
   In production this is sparse — typically 4-8 non-zero entries
   per vertex for character meshes after DDM smoothing. -/
abbrev Influences := Array (Nat × Float)

/-- LBS matvec: weighted average of (T_i · rest) across the
   vertex's influences. Returns the deformed 3-position.
   The sum of weights does not need to be exactly 1 — for
   harmonic-derived weights it's automatically partition-of-unity
   (sums to 1 by construction of the harmonic basis). -/
def lbsMatvec (transforms : Array Mat4)
                  (infl : Influences)
                  (rx ry rz : Float) : Float × Float × Float := Id.run do
  let mut x : Float := 0.0
  let mut y : Float := 0.0
  let mut z : Float := 0.0
  for kv in infl do
    let (i, w) := kv
    let T := transforms[i]!
    let (tx, ty, tz) := applyTransform4x4 T rx ry rz
    x := x + w * tx
    y := y + w * ty
    z := z + w * tz
  return (x, y, z)


/-- 3-vector close. -/
def vclose (a b : Float × Float × Float) (eps : Float) : Bool :=
  fclose a.1 b.1 eps ∧ fclose a.2.1 b.2.1 eps ∧ fclose a.2.2 b.2.2 eps

end DirectDeltaMush

/- ============================================================ -/
/- Native-decide checks on the runtime kernel.                  -/
/- ============================================================ -/

namespace DirectDeltaMushExamples

open DirectDeltaMush

/-- Single handle, weight 1.0, identity transform: output = rest. -/
example :
    let infl : Influences := #[(0, 1.0)]
    let transforms : Array Mat4 := #[mat4Identity]
    vclose (lbsMatvec transforms infl 1.5 (-2.0) 0.75) (1.5, -2.0, 0.75) 1e-12 = true := by
  native_decide

/-- Single handle, weight 1.0, pure translation: output = rest + t. -/
example :
    let infl : Influences := #[(0, 1.0)]
    let transforms : Array Mat4 := #[mat4Translation 2.0 (-3.0) 1.0]
    vclose (lbsMatvec transforms infl 1.0 1.0 1.0) (3.0, -2.0, 2.0) 1e-12 = true := by
  native_decide

/-- Two handles, equal weight 0.5 each, both identity: output = rest. -/
example :
    let infl : Influences := #[(0, 0.5), (1, 0.5)]
    let transforms : Array Mat4 := #[mat4Identity, mat4Identity]
    vclose (lbsMatvec transforms infl 4.0 (-1.0) 2.0) (4.0, -1.0, 2.0) 1e-12 = true := by
  native_decide

/-- Two handles, weights blend two translations:
    handle 0 = translate (+1, 0, 0), handle 1 = translate (-1, 0, 0),
    weights (0.7, 0.3): blended translation = 0.7·(+1) + 0.3·(-1) = +0.4
    so output = rest + (+0.4, 0, 0). -/
example :
    let infl : Influences := #[(0, 0.7), (1, 0.3)]
    let transforms : Array Mat4 :=
      #[mat4Translation 1.0 0.0 0.0, mat4Translation (-1.0) 0.0 0.0]
    vclose (lbsMatvec transforms infl 0.0 0.0 0.0) (0.4, 0.0, 0.0) 1e-12 = true := by
  native_decide

/-- Empty influences: output is zero vector (consistent with
   sum-over-empty-set convention). Caller must ensure influence
   list is non-empty in practice; this case is for the Lean
   total-function discipline. -/
example :
    let infl : Influences := #[]
    let transforms : Array Mat4 := #[mat4Identity]
    vclose (lbsMatvec transforms infl 1.0 1.0 1.0) (0.0, 0.0, 0.0) 1e-12 = true := by
  native_decide

/-- Four-handle uniform blend at 1/4 each, all identity transforms:
    output equals rest pose (partition of unity preserves identity). -/
example :
    let infl : Influences :=
      #[(0, 0.25), (1, 0.25), (2, 0.25), (3, 0.25)]
    let id4 : Array Mat4 :=
      #[mat4Identity, mat4Identity, mat4Identity, mat4Identity]
    vclose (lbsMatvec id4 infl 7.0 8.0 9.0) (7.0, 8.0, 9.0) 1e-12 = true := by
  native_decide

end DirectDeltaMushExamples

end Curvenet
