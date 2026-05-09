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

/-- (n×m) · m-vector -> n-vector. -/
def matVec (n m : Nat) (a : Mat) (v : Array Float) : Array Float := Id.run do
  let mut out : Array Float := Array.replicate n 0.0
  for i in [0:n] do
    let mut s : Float := 0.0
    for j in [0:m] do
      s := s + get a m i j * v[j]!
    out := out.set! i s
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

/-- Solve A · X = B for n×n A and n×k B, column by column. Returns n×k X.
   Naive — re-runs Gaussian elimination per column. Slice 5's `solve` is
   re-used as-is; the C++ runtime will swap in a once-and-reuse LU/Cholesky
   factorization. -/
def solveMulti (n k : Nat) (A : Mat) (B : Mat) : Mat := Id.run do
  let mut X : Mat := zeros n k
  for col in [0:k] do
    let mut bCol : Array Float := Array.replicate n 0.0
    for i in [0:n] do
      bCol := bCol.set! i (get B k i col)
    let xCol := solve n A bCol
    for i in [0:n] do
      X := set X k i col xCol[i]!
  return X

/-- Cached LU factorization with partial pivoting. Self-contained
   replacement for Eigen's `SimplicialLLT` in the cut-mesh runtime —
   factor once at bind time, reuse via `solveWithLU` for every per-frame
   RHS column. The C++ mirror is `curvenet::dense::LUFactor`. -/
structure LUFactor where
  n     : Nat
  a     : Mat                -- n×n; L below diagonal (unit, implicit), U on/above
  piv   : Array Nat          -- row permutation from partial pivoting
  valid : Bool
deriving Repr, Inhabited

/-- Factorize an n×n matrix into LU with partial pivoting. -/
def factorizeLU (n : Nat) (Ain : Mat) : LUFactor := Id.run do
  let mut a : Mat := Ain
  let mut piv : Array Nat := Array.ofFn (n := n) (fun (i : Fin n) => i.val)
  for k in [0:n] do
    -- Partial pivot.
    let mut pivotRow : Nat := k
    let mut best : Float := (a[k * n + k]!).abs
    for r in [k+1:n] do
      let v := (a[r * n + k]!).abs
      if v > best then
        best := v
        pivotRow := r
    if pivotRow ≠ k then
      for j in [0:n] do
        let tmp := a[k * n + j]!
        a := a.set! (k * n + j) a[pivotRow * n + j]!
        a := a.set! (pivotRow * n + j) tmp
      let tmpPiv := piv[k]!
      piv := piv.set! k piv[pivotRow]!
      piv := piv.set! pivotRow tmpPiv
    let pivot := a[k * n + k]!
    if pivot == 0.0 then
      continue
    for r in [k+1:n] do
      let factor := a[r * n + k]! / pivot
      a := a.set! (r * n + k) factor
      for j in [k+1:n] do
        let nv := a[r * n + j]! - factor * a[k * n + j]!
        a := a.set! (r * n + j) nv
  return { n := n, a := a, piv := piv, valid := true }

/-- Solve A·x = b using a precomputed LU factorization. O(n²). -/
def solveWithLU (f : LUFactor) (b : Array Float) : Array Float := Id.run do
  let n := f.n
  if !f.valid then return Array.replicate n 0.0
  if b.size ≠ n then return Array.replicate n 0.0
  -- Apply row permutation: y = P b
  let mut y : Array Float := Array.replicate n 0.0
  for i in [0:n] do
    y := y.set! i b[f.piv[i]!]!
  -- Forward sub: L y = (P b), L unit lower triangular (diagonal implicit).
  for i in [0:n] do
    let mut s : Float := y[i]!
    for j in [0:i] do
      s := s - f.a[i * n + j]! * y[j]!
    y := y.set! i s
  -- Back sub: U x = y.
  let mut x : Array Float := Array.replicate n 0.0
  for i in [0:n] do
    let row := n - 1 - i
    let mut s : Float := y[row]!
    for j in [row+1:n] do
      s := s - f.a[row * n + j]! * x[j]!
    let pivot := f.a[row * n + row]!
    let v := if pivot == 0.0 then 0.0 else s / pivot
    x := x.set! row v
  return x

/-- Multi-column solve sharing one LU factorization. -/
def solveMultiWithLU (f : LUFactor) (k : Nat) (B : Mat) : Mat := Id.run do
  let n := f.n
  let mut X : Mat := zeros n k
  for col in [0:k] do
    let mut bCol : Array Float := Array.replicate n 0.0
    for i in [0:n] do
      bCol := bCol.set! i (get B k i col)
    let xCol := solveWithLU f bCol
    for i in [0:n] do
      X := set X k i col xCol[i]!
  return X

/-- Element-wise tolerance check for matrices in row-major form. -/
def matWithinEps (a b : Mat) (n m : Nat) (eps : Float) : Bool := Id.run do
  for i in [0:n] do
    for j in [0:m] do
      if (get a m i j - get b m i j).abs ≥ eps then return false
  return true

/- ============================================================ -/
/- Cholesky factorization for SPD matrices                     -/
/- ============================================================ -/
/-
Cholesky factor `A = L · Lᵀ` for symmetric positive-definite A.
Stored in-place in the lower triangle of L (upper triangle ignored).

Per-meshlet local matrices in the deformer's Schur-complement path
are SPD (after symmetric Dirichlet pinning at boundary verts), and
they're small (~256 verts) so dense Cholesky's O(n³) factor is
~5.6M flops per meshlet — cheap. The per-frame back-substitution is
O(n²) ≈ 66k flops per meshlet, dominating compute time but still
small enough that 274 meshlets at 12 RHS columns fits in the budget.
-/

/-- Cholesky factorize an n×n SPD matrix. Returns L (lower-triangular,
   upper triangle zero). Stable on well-conditioned SPD; produces NaN
   on indefinite or near-singular input (the runtime should mollify
   the matrix first). -/
def choleskyFactor (n : Nat) (Ain : Mat) : Mat := Id.run do
  let mut L : Mat := zeros n n
  for j in [0:n] do
    -- L[j, j] = sqrt(A[j, j] - Σ_{k<j} L[j, k]²)
    let mut s : Float := 0.0
    for k in [0:j] do
      let v := get L n j k
      s := s + v * v
    let diag : Float := (get Ain n j j - s).sqrt
    L := set L n j j diag
    for i in [j+1:n] do
      -- L[i, j] = (A[i, j] - Σ_{k<j} L[i, k]·L[j, k]) / L[j, j]
      let mut t : Float := 0.0
      for k in [0:j] do
        t := t + get L n i k * get L n j k
      L := set L n i j ((get Ain n i j - t) / diag)
  return L

/-- Forward solve `L · y = b` for lower-triangular L. -/
def forwardSolve (n : Nat) (L : Mat) (b : Array Float) : Array Float := Id.run do
  let mut y : Array Float := Array.replicate n 0.0
  for i in [0:n] do
    let mut s : Float := 0.0
    for j in [0:i] do
      s := s + get L n i j * y[j]!
    y := y.set! i ((b[i]! - s) / get L n i i)
  return y

/-- Backward solve `Lᵀ · x = y` for upper-triangular Lᵀ. We pass L
   (the lower-triangular factor) directly and read its transpose. -/
def backwardSolve (n : Nat) (L : Mat) (y : Array Float) : Array Float := Id.run do
  let mut x : Array Float := Array.replicate n 0.0
  for ii in [0:n] do
    let i := n - 1 - ii
    let mut s : Float := 0.0
    for j in [i+1:n] do
      -- Reading Lᵀ[i, j] = L[j, i]
      s := s + get L n j i * x[j]!
    x := x.set! i ((y[i]! - s) / get L n i i)
  return x

/-- Solve `A · x = b` given Cholesky factor L of A. Composes the
   forward and backward triangular solves. -/
def solveWithCholesky (n : Nat) (L : Mat) (b : Array Float) : Array Float :=
  backwardSolve n L (forwardSolve n L b)

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

/- ============================================================ -/
/- LU factorization (cached, factor-once-solve-many) checks.    -/
/- The C++ runtime caches one of these on RestCache and reuses  -/
/- it for every per-frame RHS column — same role Eigen's        -/
/- `SimplicialLLT` would play, but self-contained.              -/
/- ============================================================ -/

/-- Identity round-trip: factorize I, solve I·x = b, recover b. -/
example :
    let A := identity 3
    let f := factorizeLU 3 A
    let b : Array Float := #[1.0, 2.0, 3.0]
    let x := solveWithLU f b
    vecWithinEps x b 1e-12 = true := by native_decide

/-- Diagonal solve via LU: diag(2, 3, 4)·x = (4, 9, 16) ⇒ x = (2, 3, 4). -/
example :
    let A : Mat := #[2.0, 0.0, 0.0,
                      0.0, 3.0, 0.0,
                      0.0, 0.0, 4.0]
    let f := factorizeLU 3 A
    let b : Array Float := #[4.0, 9.0, 16.0]
    let x := solveWithLU f b
    vecWithinEps x #[2.0, 3.0, 4.0] 1e-12 = true := by native_decide

/-- 2×2 SPD solve via LU matches the in-place Gaussian elimination from
   the original slice 5 routine. -/
example :
    let A : Mat := #[2.0, 1.0, 1.0, 2.0]
    let f := factorizeLU 2 A
    let b : Array Float := #[3.0, 3.0]
    let x := solveWithLU f b
    vecWithinEps x #[1.0, 1.0] 1e-12 = true := by native_decide

/-- Pivoting required: leading-zero pivot triggers a row swap recorded
   in `f.piv`; solving still recovers the right answer. -/
example :
    let A : Mat := #[0.0, 1.0, 1.0, 0.0]
    let f := factorizeLU 2 A
    let b : Array Float := #[2.0, 3.0]
    let x := solveWithLU f b
    vecWithinEps x #[3.0, 2.0] 1e-12 = true := by native_decide

/-- Round-trip: A·x reproduces b after factor + solve on a tridiagonal
   SPD example. -/
example :
    let A : Mat := #[2.0, 1.0, 0.0,
                      1.0, 2.0, 1.0,
                      0.0, 1.0, 2.0]
    let f := factorizeLU 3 A
    let b : Array Float := #[5.0, 6.0, 9.0]
    let x := solveWithLU f b
    let bAgain : Array Float :=
      #[ A[0]! * x[0]! + A[1]! * x[1]! + A[2]! * x[2]!
       , A[3]! * x[0]! + A[4]! * x[1]! + A[5]! * x[2]!
       , A[6]! * x[0]! + A[7]! * x[1]! + A[8]! * x[2]! ]
    vecWithinEps bAgain b 1e-10 = true := by native_decide

/-- Multi-column solve sharing one factorization: identity LHS
   reproduces the RHS columns unchanged. -/
example :
    let A := identity 3
    let f := factorizeLU 3 A
    let B : Mat := #[1.0, 2.0, 3.0,
                      4.0, 5.0, 6.0,
                      7.0, 8.0, 9.0]
    let X := solveMultiWithLU f 3 B
    matWithinEps X B 3 3 1e-12 = true := by native_decide

/-- `solveWithLU` agrees with the slice-5 single-shot `solve` on a
   non-trivial system. The factor-once-solve-many refactor is
   semantics-preserving (factorization adds no error beyond the
   underlying Gaussian elimination). -/
example :
    let A : Mat := #[4.0, 3.0, 0.0,
                      6.0, 3.0, 0.0,
                      0.0, 0.0, 1.0]
    let b : Array Float := #[7.0, 9.0, 1.0]
    let xDirect := solve 3 A b
    let f := factorizeLU 3 A
    let xLU := solveWithLU f b
    vecWithinEps xDirect xLU 1e-10 = true := by native_decide

/-- Cholesky on the 2×2 SPD `[[4, 2], [2, 3]]`. Analytical L =
   `[[2, 0], [1, √2]]`. -/
example :
    let A : Mat := #[4.0, 2.0,
                      2.0, 3.0]
    let L := choleskyFactor 2 A
    let expected : Mat := #[2.0, 0.0,
                             1.0, 2.0.sqrt]
    matWithinEps L expected 2 2 1e-12 = true := by native_decide

/-- Cholesky composes back to A: `L · Lᵀ ≈ A` on the 2×2. -/
example :
    let A : Mat := #[4.0, 2.0,
                      2.0, 3.0]
    let L := choleskyFactor 2 A
    let LT := transpose 2 2 L
    let LLT := matMul 2 2 2 L LT
    matWithinEps LLT A 2 2 1e-12 = true := by native_decide

/-- Forward+backward Cholesky solve recovers x on the 2×2 SPD. -/
example :
    let A : Mat := #[4.0, 2.0,
                      2.0, 3.0]
    let b : Array Float := #[10.0, 8.0]
    -- Analytical: x = A⁻¹ b. det(A) = 8; A⁻¹ = [[3/8, -1/4], [-1/4, 1/2]].
    -- x = (3/8 · 10 - 1/4 · 8, -1/4 · 10 + 1/2 · 8) = (1.75, 1.5)
    let L := choleskyFactor 2 A
    let x := solveWithCholesky 2 L b
    vecWithinEps x #[1.75, 1.5] 1e-12 = true := by native_decide

/-- Cholesky on a 4×4 1D Laplacian-like SPD matches LU's answer on
   the same problem (independent verification of the new path). -/
example :
    let A : Mat :=
      #[ 2.0, -1.0,  0.0,  0.0,
         -1.0,  2.0, -1.0,  0.0,
          0.0, -1.0,  2.0, -1.0,
          0.0,  0.0, -1.0,  2.0]
    let b : Array Float := #[1.0, 0.0, 0.0, 0.0]
    let xLU := solve 4 A b
    let L := choleskyFactor 4 A
    let xCh := solveWithCholesky 4 L b
    vecWithinEps xLU xCh 1e-10 = true := by native_decide

/-- Multi-RHS via reusing the same Cholesky factor: solve A · x_i = b_i
   for several b_i without refactoring. -/
example :
    let A : Mat := #[4.0, 2.0,
                      2.0, 3.0]
    let L := choleskyFactor 2 A
    let b1 : Array Float := #[10.0, 8.0]   -- expected (1.75, 1.5)
    let b2 : Array Float := #[4.0, 3.0]    -- expected (0.75, 0.5)
    let x1 := solveWithCholesky 2 L b1
    let x2 := solveWithCholesky 2 L b2
    (vecWithinEps x1 #[1.75, 1.5] 1e-12 ∧
     vecWithinEps x2 #[0.75, 0.5] 1e-12) = true := by native_decide

end LinAlgExamples

end Curvenet
