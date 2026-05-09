/-
MIT License — Copyright (c) 2026 K. S. Ernest (iFire) Lee.

Mean-value coordinates (Floater 2003) for an N-sided convex polygon, used as
the parametric domain of an N≥5 patch. Mirrors the C++ specification that
will appear in src/curvenet/mean_value.h.

Tangent half-angle form (no atan2 needed):
  d_k       = V_k − p
  r_k       = |d_k|
  τ_k       = cross(d_k, d_{k+1}) / (r_k · r_{k+1} + dot(d_k, d_{k+1}))
              -- equals tan(α_k / 2) where α_k = ∠(V_k, p, V_{k+1})
  raw_k     = (τ_{k−1} + τ_k) / r_k
  λ_k       = raw_k / Σⱼ raw_j

At a polygon vertex r_k = 0, so we special-case: if `p` bit-equals any V_k,
return the indicator vector (1.0 at k, 0.0 elsewhere).
-/

namespace Curvenet

structure Vec2 where
  x : Float
  y : Float
deriving Repr, Inhabited

@[inline] def Vec2.beq (a b : Vec2) : Bool := a.x == b.x && a.y == b.y
instance : BEq Vec2 := ⟨Vec2.beq⟩

@[inline] def Vec2.sub (a b : Vec2) : Vec2 := ⟨a.x - b.x, a.y - b.y⟩
instance : Sub Vec2 := ⟨Vec2.sub⟩

@[inline] def Vec2.norm (v : Vec2) : Float := (v.x * v.x + v.y * v.y).sqrt
@[inline] def Vec2.dot (a b : Vec2) : Float := a.x * b.x + a.y * b.y
@[inline] def Vec2.cross (a b : Vec2) : Float := a.x * b.y - a.y * b.x

/-- Regular N-gon on the unit circle. Vertex k at angle 2πk/N (k = 0 at +x). -/
def regularNgon (n : Nat) : Array Vec2 :=
  let twoPi : Float := 6.283185307179586
  Array.ofFn (n := n) (fun (k : Fin n) =>
    let θ := twoPi * Float.ofNat k.val / Float.ofNat n
    ⟨θ.cos, θ.sin⟩)

/-- Indicator vector of length n with 1.0 at index i, 0.0 elsewhere. -/
def indicator (n i : Nat) : Array Float :=
  Array.ofFn (n := n) (fun (k : Fin n) => if k.val = i then 1.0 else 0.0)

/-- Mean-value coordinates of `p` against polygon `poly`. -/
def meanValueWeights (poly : Array Vec2) (p : Vec2) : Array Float :=
  let n := poly.size
  match poly.findIdx? (fun v => v == p) with
  | some i => indicator n i
  | none =>
    let d := Array.ofFn (n := n) (fun (k : Fin n) => poly[k.val]! - p)
    let r := d.map Vec2.norm
    let tanHalf := Array.ofFn (n := n) (fun (k : Fin n) =>
      let kp := (k.val + 1) % n
      let c  := Vec2.cross d[k.val]! d[kp]!
      let dt := Vec2.dot   d[k.val]! d[kp]!
      c / (r[k.val]! * r[kp]! + dt))
    let raw := Array.ofFn (n := n) (fun (k : Fin n) =>
      let km := (k.val + n - 1) % n
      (tanHalf[km]! + tanHalf[k.val]!) / r[k.val]!)
    let total := raw.foldl (· + ·) 0.0
    raw.map (· / total)

/- ============================================================ -/
/- Concrete corner-Lagrange checks: λ_k(V_k) = 1, others = 0.  -/
/- ============================================================ -/

private def pentagon : Array Vec2 := regularNgon 5

example : meanValueWeights pentagon pentagon[0]! == #[1.0, 0.0, 0.0, 0.0, 0.0] := by
  native_decide

example : meanValueWeights pentagon pentagon[1]! == #[0.0, 1.0, 0.0, 0.0, 0.0] := by
  native_decide

example : meanValueWeights pentagon pentagon[2]! == #[0.0, 0.0, 1.0, 0.0, 0.0] := by
  native_decide

example : meanValueWeights pentagon pentagon[3]! == #[0.0, 0.0, 0.0, 1.0, 0.0] := by
  native_decide

example : meanValueWeights pentagon pentagon[4]! == #[0.0, 0.0, 0.0, 0.0, 1.0] := by
  native_decide

/- ============================================================ -/
/- Partition-of-unity at a non-vertex interior point (origin). -/
/- The pentagon centroid is (0, 0); by symmetry every λ_k = 1/5. -/
/- ============================================================ -/

/-- At the regular-pentagon centroid, weights sum to 1 (within tight float eps). -/
example :
    let w := meanValueWeights pentagon ⟨0.0, 0.0⟩
    let s := w.foldl (· + ·) 0.0
    ((s - 1.0).abs < 1e-12) = true := by
  native_decide

/-- At the centroid, by symmetry every weight is within tight eps of 1/5. -/
example :
    let w := meanValueWeights pentagon ⟨0.0, 0.0⟩
    w.all (fun x => (x - 0.2).abs < 1e-12) = true := by
  native_decide

end Curvenet
