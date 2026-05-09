/-
MIT License — Copyright (c) 2026 K. S. Ernest (iFire) Lee.

Real-valued 3-vector matching the C++ struct in src/curvenet/vec3.h.

Note on equality: Lean core provides `BEq Float` (bit-pattern comparison) but
not `DecidableEq Float` (because IEEE-754 makes propositional equality on
floats ill-behaved — NaN ≠ NaN, ±0 collisions, etc.). We use Boolean
equality for `Vec3` so `native_decide` can close concrete instance checks.
-/

namespace Curvenet

structure Vec3 where
  x : Float
  y : Float
  z : Float
deriving Repr

namespace Vec3

@[inline] def add (a b : Vec3) : Vec3 := ⟨a.x + b.x, a.y + b.y, a.z + b.z⟩
@[inline] def sub (a b : Vec3) : Vec3 := ⟨a.x - b.x, a.y - b.y, a.z - b.z⟩
@[inline] def smul (s : Float) (v : Vec3) : Vec3 := ⟨s * v.x, s * v.y, s * v.z⟩

@[inline] def beq (a b : Vec3) : Bool :=
  a.x == b.x && a.y == b.y && a.z == b.z

instance : Add Vec3 := ⟨Vec3.add⟩
instance : Sub Vec3 := ⟨Vec3.sub⟩
instance : HMul Float Vec3 Vec3 := ⟨Vec3.smul⟩
instance : BEq Vec3 := ⟨Vec3.beq⟩

end Vec3

end Curvenet
