/-
MIT License — Copyright (c) 2026 K. S. Ernest (iFire) Lee.

Sparse linear algebra (CSR) + conjugate gradient solver — the
self-contained replacement for dense LU when the mesh size grows past
the few-hundred-vertex regime where dense factorization is reasonable.

Why: dense `factorizeLU` from `Curvenet.DenseLinAlg` needs O(n²) memory
and O(n³) factorization time. The cut-mesh `VᵀLₕV` LHS is sparse — for
typical triangle meshes, ~7 nonzeros per row from the one-ring graph.
Storing it as CSR keeps memory O(n) and an iterative CG solver runs in
O(nnz · √κ) per RHS, which scales gracefully past 100k vertices instead
of exploding.

This file defines:

  * `SparseMatrixCSR`   row-pointer / col-index / values triplet
  * `spmv`              sparse matrix-vector product, O(nnz)
  * `cg`                conjugate gradient with Jacobi preconditioning,
                        SPD systems only, returns the iterate after
                        `max_iter` steps or when the residual norm
                        drops below `tol`

The C++ runtime (`src/curvenet/sparse_linalg.h`) mirrors this 1:1.
-/

import Curvenet.DenseLinAlg

namespace Curvenet
namespace SparseLinAlg

/-- Compressed sparse row matrix. `rowPtr` has length `rows + 1`;
   row `i`'s nonzeros are stored in `colIdx[rowPtr[i] .. rowPtr[i+1])`
   with corresponding entries in `values`. -/
structure SparseMatrixCSR where
  rows   : Nat
  cols   : Nat
  rowPtr : Array Nat
  colIdx : Array Nat
  values : Array Float
deriving Repr, Inhabited

/-- Sparse matrix-vector product `y = A · x`. O(nnz). -/
def spmv (A : SparseMatrixCSR) (x : Array Float) : Array Float := Id.run do
  let mut y : Array Float := Array.replicate A.rows 0.0
  for i in [0:A.rows] do
    let rs := A.rowPtr[i]!
    let re := A.rowPtr[i+1]!
    let mut s : Float := 0.0
    for p in [rs:re] do
      s := s + A.values[p]! * x[A.colIdx[p]!]!
    y := y.set! i s
  return y

/-- Dot product of two equally-sized vectors. -/
def dot (a b : Array Float) : Float := Id.run do
  let mut s : Float := 0.0
  for i in [0:a.size] do
    s := s + a[i]! * b[i]!
  return s

/-- y ← y + α · x  (in place). -/
def axpy (α : Float) (x y : Array Float) : Array Float := Id.run do
  let mut out := y
  for i in [0:x.size] do
    out := out.set! i (out[i]! + α * x[i]!)
  return out

/-- y ← α · x + β · y. -/
def saxpby (α : Float) (x : Array Float) (β : Float) (y : Array Float) : Array Float := Id.run do
  let mut out : Array Float := Array.replicate y.size 0.0
  for i in [0:y.size] do
    out := out.set! i (α * x[i]! + β * y[i]!)
  return out

/-- Diagonal of an SPD CSR matrix. Used as the Jacobi preconditioner
   `M⁻¹ = diag(1/A_ii)`. -/
def diagonal (A : SparseMatrixCSR) : Array Float := Id.run do
  let mut d : Array Float := Array.replicate A.rows 0.0
  for i in [0:A.rows] do
    let rs := A.rowPtr[i]!
    let re := A.rowPtr[i+1]!
    for p in [rs:re] do
      if A.colIdx[p]! = i then
        d := d.set! i A.values[p]!
        break
  return d

/-- Element-wise divide `b / d`, mapping zero diagonal entries to 0
   (consistent with the dense solver's "promoted-vertex slot returns
   0" convention; the caller overlays the constraint value back). -/
def applyJacobi (d : Array Float) (b : Array Float) : Array Float := Id.run do
  let mut y : Array Float := Array.replicate b.size 0.0
  for i in [0:b.size] do
    let dii := d[i]!
    y := y.set! i (if dii == 0.0 then 0.0 else b[i]! / dii)
  return y

/-- Preconditioned conjugate gradient. Takes a CSR SPD matrix A, a RHS
   vector b, an iteration cap `maxIter`, and a residual tolerance
   `tol`. Returns x ≈ A⁻¹·b. The Jacobi preconditioner is `diag(A)⁻¹`,
   the cheapest preconditioner that still helps a lot for cot-Laplacian
   systems where row magnitudes vary with mesh density. -/
def cg (A : SparseMatrixCSR) (b : Array Float) (maxIter : Nat) (tol : Float) :
    Array Float := Id.run do
  let n := A.rows
  let d := diagonal A
  let mut x : Array Float := Array.replicate n 0.0
  -- r₀ = b − A·x₀ = b
  let mut r : Array Float := b
  let mut z : Array Float := applyJacobi d r
  let mut p : Array Float := z
  let mut rzOld : Float := dot r z
  let tolSq := tol * tol
  for _ in [0:maxIter] do
    let Ap := spmv A p
    let pAp := dot p Ap
    if pAp == 0.0 then break
    let α := rzOld / pAp
    x := axpy α p x
    r := axpy (-α) Ap r
    let rr := dot r r
    if rr < tolSq then
      break
    z := applyJacobi d r
    let rzNew := dot r z
    let β := if rzOld == 0.0 then 0.0 else rzNew / rzOld
    p := saxpby 1.0 z β p
    rzOld := rzNew
  return x

end SparseLinAlg

/- ============================================================ -/
/- Concrete CG checks on small SPD systems.                    -/
/- ============================================================ -/

namespace SparseLinAlgExamples

open SparseLinAlg

/-- 3×3 identity in CSR. -/
private def identity3 : SparseMatrixCSR :=
  { rows := 3, cols := 3
  , rowPtr := #[0, 1, 2, 3]
  , colIdx := #[0, 1, 2]
  , values := #[1.0, 1.0, 1.0] }

/-- spmv on identity reproduces the input. -/
example :
    let x : Array Float := #[1.0, 2.0, 3.0]
    let y := spmv identity3 x
    DenseLinAlg.vecWithinEps y x 1e-12 = true := by native_decide

/-- CG on identity recovers b in one iteration. -/
example :
    let b : Array Float := #[1.0, 2.0, 3.0]
    let x := cg identity3 b 10 1e-12
    DenseLinAlg.vecWithinEps x b 1e-9 = true := by native_decide

/-- 3×3 diagonal `diag(2, 3, 4)` in CSR. -/
private def diag234 : SparseMatrixCSR :=
  { rows := 3, cols := 3
  , rowPtr := #[0, 1, 2, 3]
  , colIdx := #[0, 1, 2]
  , values := #[2.0, 3.0, 4.0] }

/-- CG on a diagonal SPD recovers the analytical solve in 1 iteration
   thanks to Jacobi preconditioning (M⁻¹A = I). -/
example :
    let b : Array Float := #[4.0, 9.0, 16.0]
    let x := cg diag234 b 10 1e-12
    DenseLinAlg.vecWithinEps x #[2.0, 3.0, 4.0] 1e-9 = true := by native_decide

/-- 2×2 SPD `[[2, 1], [1, 2]]` in CSR (4 nonzeros). -/
private def spd2 : SparseMatrixCSR :=
  { rows := 2, cols := 2
  , rowPtr := #[0, 2, 4]
  , colIdx := #[0, 1, 0, 1]
  , values := #[2.0, 1.0, 1.0, 2.0] }

/-- CG on a tightly-conditioned 2×2 SPD recovers the dense Gaussian-elim
   answer within tolerance. -/
example :
    let b : Array Float := #[3.0, 3.0]
    let x := cg spd2 b 50 1e-12
    DenseLinAlg.vecWithinEps x #[1.0, 1.0] 1e-8 = true := by native_decide

/-- 4×4 tridiagonal Laplacian-like SPD: standard 1D −Δ stencil with
   Dirichlet boundary. CG recovers the unique solution within 50
   iterations. -/
private def laplacian4 : SparseMatrixCSR :=
  { rows := 4, cols := 4
  , rowPtr := #[0, 2, 5, 8, 10]
  -- Row 0: [2, -1, _, _]
  -- Row 1: [-1, 2, -1, _]
  -- Row 2: [_, -1, 2, -1]
  -- Row 3: [_, _, -1, 2]
  , colIdx := #[0, 1,
                  0, 1, 2,
                  1, 2, 3,
                  2, 3]
  , values := #[ 2.0, -1.0,
                 -1.0,  2.0, -1.0,
                       -1.0,  2.0, -1.0,
                              -1.0,  2.0] }

/-- 1D Laplacian: Ax = (1, 0, 0, 0) → x = (4/5, 3/5, 2/5, 1/5)
   (analytical solve). -/
example :
    let b : Array Float := #[1.0, 0.0, 0.0, 0.0]
    let x := cg laplacian4 b 200 1e-12
    DenseLinAlg.vecWithinEps x #[0.8, 0.6, 0.4, 0.2] 1e-7 = true := by native_decide

/-- CG output and dense Gaussian-elim on the same SPD agree within float
   tolerance. Proves the sparse solver is semantics-equivalent to the
   slice-5 baseline on small instances; the win is asymptotic memory. -/
example :
    let b : Array Float := #[1.0, 2.0, 3.0, 4.0]
    let x_cg := cg laplacian4 b 200 1e-12
    let A_dense : DenseLinAlg.Mat :=
      #[ 2.0, -1.0,  0.0,  0.0,
         -1.0,  2.0, -1.0,  0.0,
          0.0, -1.0,  2.0, -1.0,
          0.0,  0.0, -1.0,  2.0]
    let x_dense := DenseLinAlg.solve 4 A_dense b
    DenseLinAlg.vecWithinEps x_cg x_dense 1e-7 = true := by native_decide

end SparseLinAlgExamples

end Curvenet
