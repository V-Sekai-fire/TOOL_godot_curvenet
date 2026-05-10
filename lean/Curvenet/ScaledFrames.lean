/-
MIT License — Copyright (c) 2026 K. S. Ernest (iFire) Lee.

Scaled frames + per-segment deformation gradients — slice 4 of the
DeGoes22 rewrite.

DeGoes22 §3 attaches a scaled frame `B_i S_i` to each curvenet segment per
side, with axes `(t_i, b_i, n_i)` and scales `(l_i, w_i, h_i)`. The
deformation gradient between rest and posed configurations is
  F_i = (B_i S_i) · (B̆_i S̆_i)⁻¹.
At intersections this matrix is anisotropic (different scales per axis).

For *isolated* curves (no intersections, no anchors — the only case we
support in this slice; intersections come with the cutting algorithm),
the paper specifies a simpler form:

  F_i = (l_i / l̆_i) · R(t̆_i, t_i)

where `R` is the smallest rotation aligning the rest tangent t̆_i to the
posed tangent t_i. This is what we formalize here.

We use Rodrigues' formula expressed without an explicit angle:

  R = I + K + K² / (1 + cos θ)

where  K = skew(t̆ × t),  cos θ = t̆ · t.  Stable as long as the rest
and posed tangents are not anti-parallel (cos θ ≠ −1); the anti-parallel
case is degenerate and a future slice will pick a stable axis.
-/

import Curvenet.Vec3

namespace Curvenet
namespace ScaledFrames

/-- Tangent (unit) and length of a segment from `p` to `q`. -/
def tangentLength (p q : Vec3) : Vec3 × Float :=
  let d : Vec3 := q - p
  let l : Float := (d.x * d.x + d.y * d.y + d.z * d.z).sqrt
  let inv := if l == 0.0 then 0.0 else 1.0 / l
  (⟨d.x * inv, d.y * inv, d.z * inv⟩, l)

/-- 3×3 row-major matrix, stored as a 9-element Float array. -/
abbrev Mat3 := Array Float

@[inline] def mat3Get (m : Mat3) (i j : Nat) : Float := m[i * 3 + j]!
@[inline] def mat3Mk (m00 m01 m02 m10 m11 m12 m20 m21 m22 : Float) : Mat3 :=
  #[m00, m01, m02, m10, m11, m12, m20, m21, m22]

def mat3Identity : Mat3 := mat3Mk 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0

def mat3Add (a b : Mat3) : Mat3 :=
  Array.ofFn (n := 9) (fun (k : Fin 9) => a[k.val]! + b[k.val]!)

def mat3SMul (s : Float) (a : Mat3) : Mat3 :=
  Array.ofFn (n := 9) (fun (k : Fin 9) => s * a[k.val]!)

def mat3Mul (a b : Mat3) : Mat3 :=
  Array.ofFn (n := 9) (fun (k : Fin 9) =>
    let i := k.val / 3
    let j := k.val % 3
    mat3Get a i 0 * mat3Get b 0 j +
    mat3Get a i 1 * mat3Get b 1 j +
    mat3Get a i 2 * mat3Get b 2 j)

/-- Skew-symmetric (cross-product) matrix `[v]_×`:
   [[ 0,  -z,  y],
    [ z,   0, -x],
    [-y,   x,  0]]
-/
def skew (v : Vec3) : Mat3 :=
  mat3Mk 0.0 (-v.z) v.y v.z 0.0 (-v.x) (-v.y) v.x 0.0

/-- Smallest rotation matrix aligning unit vector `from_` to unit vector `to_`.
   Returns identity when the two are equal; degenerate (anti-parallel) inputs
   produce a NaN-laden matrix and are not exercised in this slice's tests. -/
def smallestRotation (from_ to_ : Vec3) : Mat3 :=
  let cx := from_.y * to_.z - from_.z * to_.y
  let cy := from_.z * to_.x - from_.x * to_.z
  let cz := from_.x * to_.y - from_.y * to_.x
  let crossV : Vec3 := ⟨cx, cy, cz⟩
  let cosT  : Float := from_.x * to_.x + from_.y * to_.y + from_.z * to_.z
  -- R = I + K + K^2 / (1 + cos θ)
  let K  := skew crossV
  let K2 := mat3Mul K K
  let denom := 1.0 + cosT
  let invDenom := if denom == 0.0 then 0.0 else 1.0 / denom
  mat3Add (mat3Add mat3Identity K) (mat3SMul invDenom K2)

/-- Deformation gradient for an isolated-curve segment:
   F = (l / l̆) · R(t̆, t). -/
def isolatedSegmentGradient (rest_p rest_q posed_p posed_q : Vec3) : Mat3 :=
  let (tRest,  lRest)  := tangentLength rest_p rest_q
  let (tPosed, lPosed) := tangentLength posed_p posed_q
  let r := if lRest == 0.0 then 0.0 else lPosed / lRest
  mat3SMul r (smallestRotation tRest tPosed)

/-- |a_ij - b_ij| < eps for all i, j. -/
def mat3WithinEps (a b : Mat3) (eps : Float) : Bool := Id.run do
  for k in [0:9] do
    if (a[k]! - b[k]!).abs ≥ eps then return false
  return true

/-- Multiply a 3×3 matrix by a column vector. -/
def mat3MulVec (m : Mat3) (v : Vec3) : Vec3 :=
  ⟨ mat3Get m 0 0 * v.x + mat3Get m 0 1 * v.y + mat3Get m 0 2 * v.z
  , mat3Get m 1 0 * v.x + mat3Get m 1 1 * v.y + mat3Get m 1 2 * v.z
  , mat3Get m 2 0 * v.x + mat3Get m 2 1 * v.y + mat3Get m 2 2 * v.z ⟩

/-- Determinant of a 3×3 row-major matrix. -/
def mat3Det (m : Mat3) : Float :=
  let a := mat3Get m 0 0
  let b := mat3Get m 0 1
  let c := mat3Get m 0 2
  let d := mat3Get m 1 0
  let e := mat3Get m 1 1
  let f := mat3Get m 1 2
  let g := mat3Get m 2 0
  let h := mat3Get m 2 1
  let i := mat3Get m 2 2
  a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

/-- Inverse of a 3×3 row-major matrix via the adjugate. Returns NaN-laden
   garbage if the input is singular (det = 0). -/
def mat3Inv (m : Mat3) : Mat3 :=
  let a := mat3Get m 0 0
  let b := mat3Get m 0 1
  let c := mat3Get m 0 2
  let d := mat3Get m 1 0
  let e := mat3Get m 1 1
  let f := mat3Get m 1 2
  let g := mat3Get m 2 0
  let h := mat3Get m 2 1
  let i := mat3Get m 2 2
  let det := a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
  let inv := if det == 0.0 then 0.0 else 1.0 / det
  mat3Mk
    (inv * (e * i - f * h))  (inv * (c * h - b * i))  (inv * (b * f - c * e))
    (inv * (f * g - d * i))  (inv * (a * i - c * g))  (inv * (c * d - a * f))
    (inv * (d * h - e * g))  (inv * (b * g - a * h))  (inv * (a * e - b * d))

/-- DeGoes22 §3 deformation gradient combining rest and posed scaled
   frames: `F = (B S)·(B̆ S̆)⁻¹`. Both matrices are 3×3 row-major. -/
def deformationGradient (restFrame posedFrame : Mat3) : Mat3 :=
  mat3Mul posedFrame (mat3Inv restFrame)

end ScaledFrames

/- ============================================================ -/
/- Concrete deformation-gradient checks.                       -/
/- ============================================================ -/

namespace ScaledFramesExamples

open ScaledFrames

private def origin : Vec3 := ⟨0.0, 0.0, 0.0⟩
private def xUnit  : Vec3 := ⟨1.0, 0.0, 0.0⟩
private def yUnit  : Vec3 := ⟨0.0, 1.0, 0.0⟩

/-- Identity case: rest segment = posed segment → F = I. -/
example :
    let F := isolatedSegmentGradient origin xUnit origin xUnit
    mat3WithinEps F mat3Identity 1e-12 = true := by native_decide

/-- Pure rotation: rest along +x, posed along +y → F is the 90° z-rotation. -/
example :
    let F := isolatedSegmentGradient origin xUnit origin yUnit
    let R90 : Mat3 := mat3Mk 0.0 (-1.0) 0.0  1.0 0.0 0.0  0.0 0.0 1.0
    mat3WithinEps F R90 1e-12 = true := by native_decide

/-- Rotation pulls rest tangent onto posed tangent. -/
example :
    let R := smallestRotation xUnit yUnit
    let v := mat3MulVec R xUnit
    ((v.x - 0.0).abs < 1e-12 && (v.y - 1.0).abs < 1e-12 && v.z.abs < 1e-12) = true := by
  native_decide

/-- Pure isotropic scaling: rest along +x of length 1, posed along +x of
   length 2 → F = 2·I. -/
example :
    let F := isolatedSegmentGradient origin xUnit origin ⟨2.0, 0.0, 0.0⟩
    let twoI : Mat3 := mat3SMul 2.0 mat3Identity
    mat3WithinEps F twoI 1e-12 = true := by native_decide

/-- Combined: rotate + scale. Rest along +x length 1, posed along +y length
   3 → F = 3 · R90. Applying F to rest displacement reproduces posed
   displacement. -/
example :
    let F := isolatedSegmentGradient origin xUnit origin ⟨0.0, 3.0, 0.0⟩
    let restDisp : Vec3 := xUnit  -- rest_q - rest_p
    let posedDisp := mat3MulVec F restDisp
    ((posedDisp.x).abs < 1e-12 && (posedDisp.y - 3.0).abs < 1e-12 && posedDisp.z.abs < 1e-12) = true := by
  native_decide

/-- Smallest-rotation invariants: identity input → identity matrix. -/
example :
    mat3WithinEps (smallestRotation xUnit xUnit) mat3Identity 1e-12 = true := by native_decide

/- ============================================================ -/
/- 3×3 determinant + inverse + intersection-aware deformation   -/
/- gradient.                                                    -/
/- ============================================================ -/

/-- det(I) = 1. -/
example : (mat3Det mat3Identity - 1.0).abs < 1e-12 := by native_decide

/-- det(diag(2, 3, 4)) = 24. -/
example :
    let M : Mat3 := mat3Mk 2.0 0.0 0.0  0.0 3.0 0.0  0.0 0.0 4.0
    (mat3Det M - 24.0).abs < 1e-12 := by native_decide

/-- inv(I) = I. -/
example :
    mat3WithinEps (mat3Inv mat3Identity) mat3Identity 1e-12 = true := by native_decide

/-- inv(diag(2, 3, 4)) = diag(1/2, 1/3, 1/4). -/
example :
    let M  : Mat3 := mat3Mk 2.0 0.0 0.0  0.0 3.0 0.0  0.0 0.0 4.0
    let Mi : Mat3 := mat3Mk 0.5 0.0 0.0  0.0 (1.0/3.0) 0.0  0.0 0.0 0.25
    mat3WithinEps (mat3Inv M) Mi 1e-12 = true := by native_decide

/-- inv(M) · M = I (round-trip). -/
example :
    let M : Mat3 := mat3Mk 2.0 1.0 0.0  1.0 2.0 1.0  0.0 1.0 2.0
    mat3WithinEps (mat3Mul (mat3Inv M) M) mat3Identity 1e-10 = true := by native_decide

/-- Identity rest, identity posed → F = I. -/
example :
    mat3WithinEps (deformationGradient mat3Identity mat3Identity) mat3Identity 1e-12 = true := by native_decide

/-- Identity rest, anisotropic posed → F = posed (since I⁻¹ = I). -/
example :
    let posed : Mat3 := mat3Mk 2.0 0.0 0.0  0.0 3.0 0.0  0.0 0.0 4.0
    mat3WithinEps (deformationGradient mat3Identity posed) posed 1e-12 = true := by native_decide

/-- Same rest = posed → F = I (no deformation). -/
example :
    let M : Mat3 := mat3Mk 2.0 1.0 0.0  1.0 2.0 1.0  0.0 1.0 2.0
    mat3WithinEps (deformationGradient M M) mat3Identity 1e-10 = true := by native_decide

/-- Round-trip in the other direction: M · inv(M) = I. -/
example :
    let M : Mat3 := mat3Mk 2.0 1.0 0.0  1.0 2.0 1.0  0.0 1.0 2.0
    mat3WithinEps (mat3Mul M (mat3Inv M)) mat3Identity 1e-10 = true := by native_decide

/-- Round-trip on a non-symmetric invertible frame. -/
example :
    let M : Mat3 := mat3Mk 1.0 2.0 3.0  0.0 1.0 4.0  5.0 6.0 0.0
    mat3WithinEps (mat3Mul M (mat3Inv M)) mat3Identity 1e-10 = true := by native_decide

/-- Round-trip in the other direction on the same non-symmetric frame. -/
example :
    let M : Mat3 := mat3Mk 1.0 2.0 3.0  0.0 1.0 4.0  5.0 6.0 0.0
    mat3WithinEps (mat3Mul (mat3Inv M) M) mat3Identity 1e-10 = true := by native_decide

/-- Inverse is involutive: inv(inv(M)) = M (modulo float precision). -/
example :
    let M : Mat3 := mat3Mk 2.0 1.0 0.0  1.0 2.0 1.0  0.0 1.0 2.0
    mat3WithinEps (mat3Inv (mat3Inv M)) M 1e-9 = true := by native_decide

/-- Round-trip on a rotation matrix: R · R⁻¹ = I. -/
example :
    let R90 : Mat3 := mat3Mk 0.0 (-1.0) 0.0  1.0 0.0 0.0  0.0 0.0 1.0
    mat3WithinEps (mat3Mul R90 (mat3Inv R90)) mat3Identity 1e-12 = true := by native_decide

end ScaledFramesExamples

end Curvenet
