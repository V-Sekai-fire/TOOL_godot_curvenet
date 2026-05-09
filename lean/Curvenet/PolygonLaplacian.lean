/-
MIT License — Copyright (c) 2026 K. S. Ernest (iFire) Lee.

Polygonal Laplacian — slice 2 of the DeGoes22 rewrite.

DeGoes22 §4.2 + Appendix A invokes the polygonal Laplacian of de Goes
et al. 2020 ("Discrete Differential Operators on Polygonal Meshes"). The
"correct" formulation uses a virtual element method / discrete gradient
construction; we use a simpler-but-equivalent-on-triangulated-faces
approximation here: fan-triangulate each polygon and sum the cotangent
Laplacians of the resulting triangles. This preserves the load-bearing
invariants the cut-mesh solver depends on:

  * symmetry              L = Lᵀ
  * row sums to zero      L · 1 = 0   (kernel contains constants)
  * positive semidefinite (not yet verified in Lean — instance-only)

The full VEM construction can be a follow-up slice once the rest of the
pipeline lands; for the cut-mesh side, the polygons we'll be Laplacianing
are typically small (3- to 6-sided sub-polygons after curvenet cuts), and
fan-triangulation Laplacian is what most published curvenet-style solvers
use in practice.
-/

import Curvenet.Vec3

namespace Curvenet
namespace PolygonLaplacian

/-- Cotangent of the triangle's interior angle at `apex`.
   `cot(θ) = (a·b) / |a × b|` where `a = other1 - apex`, `b = other2 - apex`. -/
def cotAtVertex (apex other1 other2 : Vec3) : Float :=
  let a : Vec3 := other1 - apex
  let b : Vec3 := other2 - apex
  let dot := a.x * b.x + a.y * b.y + a.z * b.z
  let cx := a.y * b.z - a.z * b.y
  let cy := a.z * b.x - a.x * b.z
  let cz := a.x * b.y - a.y * b.x
  let crossLen := (cx * cx + cy * cy + cz * cz).sqrt
  dot / crossLen

/-- Get a Float entry from a row-major n×n matrix stored as `Array Float`. -/
@[inline] def get (m : Array Float) (n i j : Nat) : Float :=
  m[i * n + j]!

/-- Set a Float entry into a row-major n×n matrix. -/
@[inline] def set (m : Array Float) (n i j : Nat) (v : Float) : Array Float :=
  m.set! (i * n + j) v

/-- Add to a Float entry in a row-major n×n matrix. -/
@[inline] def addAt (m : Array Float) (n i j : Nat) (v : Float) : Array Float :=
  m.set! (i * n + j) (m[i * n + j]! + v)

/-- Cotangent Laplacian for a single triangle, returned as a 3×3 row-major
   matrix. Edge weights `w_uv = cot(angle at apex)/2`. Row sum is 0. -/
def triangleCotLaplacian (a b c : Vec3) : Array Float := Id.run do
  let cotA := cotAtVertex a b c
  let cotB := cotAtVertex b a c
  let cotC := cotAtVertex c a b
  let w_bc := cotA * 0.5
  let w_ac := cotB * 0.5
  let w_ab := cotC * 0.5
  -- Order rows/cols as (a=0, b=1, c=2).
  let mut m : Array Float := Array.replicate 9 0.0
  m := set m 3 0 1 (-w_ab); m := set m 3 1 0 (-w_ab)
  m := set m 3 0 2 (-w_ac); m := set m 3 2 0 (-w_ac)
  m := set m 3 1 2 (-w_bc); m := set m 3 2 1 (-w_bc)
  m := set m 3 0 0 (w_ab + w_ac)
  m := set m 3 1 1 (w_ab + w_bc)
  m := set m 3 2 2 (w_ac + w_bc)
  return m

/-- Polygonal cotangent Laplacian by fan triangulation from vertex 0:
   triangles (0, i, i+1) for i = 1..n-2. Each triangle contributes its
   3×3 cotangent Laplacian to the corresponding (0, i, i+1) rows/cols
   of the n×n polygon matrix. -/
def polygonCotLaplacian (poly : Array Vec3) : Array Float := Id.run do
  let n := poly.size
  let mut m : Array Float := Array.replicate (n * n) 0.0
  if n < 3 then return m
  let v0 := poly[0]!
  for i in [1:n-1] do
    let vi := poly[i]!
    let vj := poly[i+1]!
    let tri := triangleCotLaplacian v0 vi vj
    -- Map local indices (0, 1, 2) -> global (0, i, i+1).
    let idx : Array Nat := #[0, i, i+1]
    for li in [0:3] do
      for lj in [0:3] do
        let gi := idx[li]!
        let gj := idx[lj]!
        m := addAt m n gi gj (get tri 3 li lj)
  return m

/-- |row i sum| < eps for all rows in n×n matrix `m`. -/
def rowSumsWithin (m : Array Float) (n : Nat) (eps : Float) : Bool := Id.run do
  for i in [0:n] do
    let mut s : Float := 0.0
    for j in [0:n] do
      s := s + get m n i j
    if s.abs ≥ eps then return false
  return true

/-- |m_ij - m_ji| < eps for all i, j in n×n matrix. -/
def isSymmetricWithin (m : Array Float) (n : Nat) (eps : Float) : Bool := Id.run do
  for i in [0:n] do
    for j in [i+1:n] do
      if (get m n i j - get m n j i).abs ≥ eps then return false
  return true

end PolygonLaplacian

/- ============================================================ -/
/- Concrete instance checks.                                   -/
/- ============================================================ -/

namespace LapExamples

open PolygonLaplacian

/-- Equilateral triangle in the XZ plane with edge length 1. -/
private def equilateralTri : Array Vec3 :=
  #[ ⟨0.0, 0.0, 0.0⟩
   , ⟨1.0, 0.0, 0.0⟩
   , ⟨0.5, 0.0, 0.8660254037844386⟩  -- (1, sqrt(3)/2)
   ]

/-- Unit square in the XZ plane (CCW from origin). -/
private def unitSquare : Array Vec3 :=
  #[ ⟨0.0, 0.0, 0.0⟩
   , ⟨1.0, 0.0, 0.0⟩
   , ⟨1.0, 0.0, 1.0⟩
   , ⟨0.0, 0.0, 1.0⟩
   ]

/-- Regular pentagon (radius 1, CCW). Tests the fan-triangulation path
   through n=5 sub-triangles. -/
private def regularPentagon : Array Vec3 :=
  let twoPi : Float := 6.283185307179586
  Array.ofFn (n := 5) (fun (k : Fin 5) =>
    let θ := twoPi * Float.ofNat k.val / 5.0
    ⟨θ.cos, 0.0, θ.sin⟩)

/-- Triangle Laplacian: row sums vanish. -/
example :
    let L := triangleCotLaplacian equilateralTri[0]! equilateralTri[1]! equilateralTri[2]!
    rowSumsWithin L 3 1e-12 = true := by native_decide

/-- Triangle Laplacian: symmetric. -/
example :
    let L := triangleCotLaplacian equilateralTri[0]! equilateralTri[1]! equilateralTri[2]!
    isSymmetricWithin L 3 1e-12 = true := by native_decide

/-- Equilateral triangle: each off-diagonal entry equals -cot(60°)/2 = -1/(2√3). -/
example :
    let L := triangleCotLaplacian equilateralTri[0]! equilateralTri[1]! equilateralTri[2]!
    let expected : Float := -1.0 / (2.0 * 3.0.sqrt)
    ((get L 3 0 1 - expected).abs < 1e-12) = true := by native_decide

/-- Quad Laplacian via fan-triangulation: row sums vanish. -/
example :
    let L := polygonCotLaplacian unitSquare
    rowSumsWithin L 4 1e-12 = true := by native_decide

/-- Quad Laplacian: symmetric. -/
example :
    let L := polygonCotLaplacian unitSquare
    isSymmetricWithin L 4 1e-12 = true := by native_decide

/-- Pentagon Laplacian: row sums vanish. -/
example :
    let L := polygonCotLaplacian regularPentagon
    rowSumsWithin L 5 1e-9 = true := by native_decide

/-- Pentagon Laplacian: symmetric. -/
example :
    let L := polygonCotLaplacian regularPentagon
    isSymmetricWithin L 5 1e-9 = true := by native_decide

end LapExamples

end Curvenet
