/-
MIT License — Copyright (c) 2026 K. S. Ernest (iFire) Lee.

Dense linear algebra primitives — slice 5 of the DeGoes22 rewrite.

The DeGoes22 solve (§4.3, Eq. 6) is two sparse linear systems sharing the
same LHS `VᵀLₕV`. For Lean instance theorems we run on tiny meshes, dense
arithmetic suffices: assemble `VᵀLₕV` densely, factorize via Gaussian
elimination with partial pivoting, back-substitute. The C++ runtime will
swap this for a sparse Cholesky factorization in a much later slice.

This file holds the matrix primitives and the linear solver only.
Slice 6 will assemble `VᵀLₕV` from `Curvenet.CutMesh` + `Curvenet.PolygonLaplacian`
and run an end-to-end solve.
-/

namespace Curvenet
namespace DenseLinAlg

/-- Row-major matrix of dimension `n × m`. Stored as a flat `Array Float` of
   size `n * m`. -/
abbrev Mat := Array Float

@[inline] def get (a : Mat) (m i j : Nat) : Float := a[i * m + j]!
@[inline] def set (a : Mat) (m i j : Nat) (v : Float) : Mat := a.set! (i * m + j) v

/-- Allocate an n×m zero matrix. -/
def zeros (n m : Nat) : Mat := Array.replicate (n * m) 0.0

/-- n×n identity. -/
def identity (n : Nat) : Mat := Id.run do
  let mut a := zeros n n
  for i in [0:n] do
    a := set a n i i 1.0
  return a

/-- (n×k) · (k×m) -> (n×m). -/
def matMul (n k m : Nat) (a b : Mat) : Mat := Id.run do
  let mut out := zeros n m
  for i in [0:n] do
    for j in [0:m] do
      let mut s : Float := 0.0
      for p in [0:k] do
        s := s + get a k i p * get b m p j
      out := set out m i j s
  return out

/-- Transpose of n×m -> m×n. -/
def transpose (n m : Nat) (a : Mat) : Mat := Id.run do
  let mut out := zeros m n
  for i in [0:n] do
    for j in [0:m] do
      out := set out n j i (get a m i j)
  return out

/-- Solve A · x = b for an n×n matrix A and an n-vector b via Gaussian
   elimination with partial pivoting. Returns `x` of length n. The matrix
   is assumed to be invertible; degenerate systems silently return whatever
   the elimination leaves behind. -/
def solve (n : Nat) (A : Mat) (b : Array Float) : Array Float := Id.run do
  -- Build an augmented n×(n+1) matrix.
  let mut aug : Mat := Array.replicate (n * (n + 1)) 0.0
  for i in [0:n] do
    for j in [0:n] do
      aug := set aug (n + 1) i j (get A n i j)
    aug := set aug (n + 1) i n b[i]!
  -- Forward elimination with partial pivoting.
  for k in [0:n] do
    -- Find pivot row with max |aug[r, k]| for r >= k.
    let mut piv := k
    let mut bestAbs : Float := (get aug (n + 1) k k).abs
    for r in [k+1:n] do
      let v := (get aug (n + 1) r k).abs
      if v > bestAbs then
        bestAbs := v
        piv := r
    if piv ≠ k then
      for j in [0:n+1] do
        let tmp := get aug (n + 1) k j
        aug := set aug (n + 1) k j (get aug (n + 1) piv j)
        aug := set aug (n + 1) piv j tmp
    let pivot := get aug (n + 1) k k
    if pivot == 0.0 then continue
    for r in [k+1:n] do
      let factor := get aug (n + 1) r k / pivot
      for j in [k:n+1] do
        let nv := get aug (n + 1) r j - factor * get aug (n + 1) k j
        aug := set aug (n + 1) r j nv
  -- Back substitution.
  let mut x : Array Float := Array.replicate n 0.0
  for i in [0:n] do
    let row := n - 1 - i
    let mut s : Float := get aug (n + 1) row n
    for j in [row+1:n] do
      s := s - get aug (n + 1) row j * x[j]!
    let pivot := get aug (n + 1) row row
    let v := if pivot == 0.0 then 0.0 else s / pivot
    x := x.set! row v
  return x

/-- Element-wise |a_i - b_i| < eps over two same-length vectors. -/
def vecWithinEps (a b : Array Float) (eps : Float) : Bool := Id.run do
  if a.size ≠ b.size then return false
  for i in [0:a.size] do
    if (a[i]! - b[i]!).abs ≥ eps then return false
  return true

/-- Element-wise tolerance check for matrices in row-major form. -/
def matWithinEps (a b : Mat) (n m : Nat) (eps : Float) : Bool := Id.run do
  for i in [0:n] do
    for j in [0:m] do
      if (get a m i j - get b m i j).abs ≥ eps then return false
  return true

end DenseLinAlg

/- ============================================================ -/
/- Concrete linear-solver checks.                              -/
/- ============================================================ -/

namespace LinAlgExamples

open DenseLinAlg

/-- 3×3 identity solves trivially: I · x = b ⇒ x = b. -/
example :
    let A := identity 3
    let b : Array Float := #[1.0, 2.0, 3.0]
    let x := solve 3 A b
    vecWithinEps x b 1e-12 = true := by native_decide

/-- Diagonal system: diag(2, 3, 4) · x = (4, 9, 16) ⇒ x = (2, 3, 4). -/
example :
    let A : Mat := #[2.0, 0.0, 0.0,
                      0.0, 3.0, 0.0,
                      0.0, 0.0, 4.0]
    let b : Array Float := #[4.0, 9.0, 16.0]
    let x := solve 3 A b
    vecWithinEps x #[2.0, 3.0, 4.0] 1e-12 = true := by native_decide

/-- Symmetric 2×2 SPD: [[2, 1], [1, 2]] · x = (3, 3) ⇒ x = (1, 1). -/
example :
    let A : Mat := #[2.0, 1.0, 1.0, 2.0]
    let b : Array Float := #[3.0, 3.0]
    let x := solve 2 A b
    vecWithinEps x #[1.0, 1.0] 1e-12 = true := by native_decide

/-- Pivoting needed: top row leading-zero-ish requires a swap.
   [[0, 1], [1, 0]] · x = (2, 3) ⇒ x = (3, 2). -/
example :
    let A : Mat := #[0.0, 1.0, 1.0, 0.0]
    let b : Array Float := #[2.0, 3.0]
    let x := solve 2 A b
    vecWithinEps x #[3.0, 2.0] 1e-12 = true := by native_decide

/-- Verify A · x = b reproduces the input on a small SPD system: assemble
   AᵀA explicitly, solve, then check residual. -/
example :
    let A : Mat := #[2.0, 0.0, 1.0,
                      0.0, 3.0, 0.0,
                      1.0, 0.0, 4.0]
    let b : Array Float := #[5.0, 6.0, 9.0]
    let x := solve 3 A b
    -- Recompute b' = A · x and compare to b.
    let bMat : Mat := matMul 3 3 1 A #[x[0]!, x[1]!, x[2]!]
    vecWithinEps bMat b 1e-10 = true := by native_decide

/-- matMul / transpose sanity: (Aᵀ A) is 3×3 symmetric for any 3×3 A. -/
example :
    let A : Mat := #[1.0, 2.0, 3.0,
                      4.0, 5.0, 6.0,
                      7.0, 8.0, 0.0]
    let At := transpose 3 3 A
    let AtA := matMul 3 3 3 At A
    let AtAt := transpose 3 3 AtA
    matWithinEps AtA AtAt 3 3 1e-12 = true := by native_decide

end LinAlgExamples

end Curvenet
