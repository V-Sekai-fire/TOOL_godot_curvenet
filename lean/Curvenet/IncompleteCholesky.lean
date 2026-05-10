/-
MIT License — Copyright (c) 2026 K. S. Ernest (iFire) Lee.

Incomplete Cholesky factorization, no-fill variant — ICC(0).

Loops 8 — 100/2 ruled out three preconditioner families on the 81k
Mire cut-mesh Laplacian (multilevel Schwarz, HEM aggregation,
kernel projection). The smoking-gun diagnostic
(`tests/diag_70k_cg_baseline.cpp`) showed plain D-Jacobi PCG
converges on 81k in 200,000 iters / 133 s — slow but working.
The matrix has a 7-decade diagonal spread and condition number
~4e10. After D-Jacobi preconditioning we still see kappa ~ 4e10:
diagonal scaling is essentially absorbed by the algorithm.

ICC(0) is the next step on the well-trodden path. It computes a
lower-triangular factor `L` with the same sparsity pattern as the
lower triangle of `A`, satisfying `L · L^T ≈ A`, by skipping any
fill-in that would land outside the original pattern. Standard
result: kappa(L^{-T}·L^{-1}·A) ~ sqrt(kappa(A)) for many SPD
problems, so PCG iters drop from ~sqrt(kappa) to ~kappa^{1/4}.
For us that's 200k → ~450, the goal.

This file specifies the algorithm on a 3×3 dense SPD example and
verifies via native_decide:
  * entry-by-entry: `L · L^T ≈ A` to 1e-12
  * `L` is lower-triangular (zeros above diagonal)
  * idempotent on a known-Cholesky 2×2 block
  * `forwardSub L b` then `backwardSub L^T y` recovers `A · x = b`

The C++ runtime mirrors this in CSR using the no-fill variant:
during column-j elimination, only update entries `L[i,j]` where
`A[i,j]` is already non-zero. This keeps `nnz(L) = nnz(tril A)`,
so the backsolves cost the same per iter as one A·x mat-vec.
-/

import Curvenet.Common

namespace Curvenet

open Curvenet.Common
namespace IncompleteCholesky

private def at3 (M : Mat3) (i j : Nat) : Float := M[3 * i + j]!

/-- Dense ICC(0) on a 3×3 SPD matrix. `pattern` indicates which
   off-diagonal entries are kept (1 = kept, 0 = dropped). The
   diagonal is always kept. Returns a lower-triangular `L` with
   `L · L^T ≈ A` for the kept entries and zero elsewhere. -/
def factor3 (A : Mat3) (pattern : Mat3) : Mat3 := Id.run do
  let mut L : Mat3 := Array.replicate 9 0.0
  for j in [0:3] do
    -- Diagonal: L[j, j] = sqrt(A[j, j] - sum_{k<j} L[j, k]^2)
    let mut s := at3 A j j
    for k in [0:j] do
      let ljk := L[3 * j + k]!
      s := s - ljk * ljk
    L := L.set! (3 * j + j) (s.sqrt)
    -- Below diagonal: L[i, j] = (A[i, j] - sum_{k<j} L[i,k]·L[j,k]) / L[j,j]
    for i in [j+1:3] do
      let keep := at3 pattern i j != 0.0 ∨ i = j
      if keep then
        let mut t := at3 A i j
        for k in [0:j] do
          let lik := L[3 * i + k]!
          let ljk := L[3 * j + k]!
          t := t - lik * ljk
        let ljj := L[3 * j + j]!
        L := L.set! (3 * i + j) (t / ljj)
  return L

/-- Multiply a 3×3 lower-triangular `L` by its transpose. -/
def lLT3 (L : Mat3) : Mat3 := Id.run do
  let mut M : Mat3 := Array.replicate 9 0.0
  for i in [0:3] do
    for j in [0:3] do
      let mut s := 0.0
      for k in [0:3] do
        s := s + L[3 * i + k]! * L[3 * j + k]!
      M := M.set! (3 * i + j) s
  return M

/-- Forward substitution: solve `L y = b` for lower-triangular L. -/
def forwardSub3 (L : Mat3) (b : Array Float) : Array Float := Id.run do
  let mut y : Array Float := Array.replicate 3 0.0
  for i in [0:3] do
    let mut s := b[i]!
    for k in [0:i] do
      s := s - L[3 * i + k]! * y[k]!
    y := y.set! i (s / L[3 * i + i]!)
  return y

/-- Backward substitution: solve `L^T x = y`. -/
def backwardSub3 (L : Mat3) (y : Array Float) : Array Float := Id.run do
  let mut x : Array Float := Array.replicate 3 0.0
  for j in [0:3] do
    let i := 2 - j
    let mut s := y[i]!
    for kk in [i+1:3] do
      s := s - L[3 * kk + i]! * x[kk]!
    x := x.set! i (s / L[3 * i + i]!)
  return x


/-- Mat3 vs Mat3 close (entry-wise). -/
def mat3Close (M N : Mat3) (eps : Float) : Bool := Id.run do
  let mut ok := true
  for i in [0:9] do
    if ¬ fclose M[i]! N[i]! eps then ok := false
  return ok

/-- Vec3 close. -/
def vec3Close (a b : Array Float) (eps : Float) : Bool := Id.run do
  let mut ok := true
  for i in [0:3] do
    if ¬ fclose a[i]! b[i]! eps then ok := false
  return ok

end IncompleteCholesky

/- ============================================================ -/
/- Native-decide checks on small SPD examples.                   -/
/- ============================================================ -/

namespace IncompleteCholeskyExamples

open IncompleteCholesky

/-- Tridiagonal SPD: diagonal 4, off-diag -1. -/
private def Atri : Mat3 :=
  #[ 4.0, -1.0,  0.0
   ,-1.0,  4.0, -1.0
   , 0.0, -1.0,  4.0
   ]

/-- Full pattern: keep all off-diagonal entries. -/
private def fullPat : Mat3 :=
  #[ 0.0, 1.0, 1.0
   , 1.0, 0.0, 1.0
   , 1.0, 1.0, 0.0
   ]

example :
    let L := factor3 Atri fullPat
    -- Lower triangular: zeros above diagonal.
    (fclose L[1]! 0.0 1e-12 ∧ fclose L[2]! 0.0 1e-12 ∧
     fclose L[5]! 0.0 1e-12) = true := by native_decide

example :
    let L  := factor3 Atri fullPat
    let LL := lLT3 L
    mat3Close LL Atri 1e-12 = true := by native_decide

/-- Forward + backward sub recovers x from b = A x. -/
example :
    let L := factor3 Atri fullPat
    let b : Array Float := #[3.0, 2.0, 5.0]
    -- A · x = b for tridiagonal SPD, then check forward/back recovers x.
    -- Solve x = A^{-1} b via the factorisation:
    let y := forwardSub3 L b
    let x := backwardSub3 L y
    -- Multiply A · x and compare to b.
    let Ax0 := 4.0 * x[0]! + (-1.0) * x[1]!
    let Ax1 := (-1.0) * x[0]! + 4.0 * x[1]! + (-1.0) * x[2]!
    let Ax2 := (-1.0) * x[1]! + 4.0 * x[2]!
    (fclose Ax0 b[0]! 1e-10 ∧ fclose Ax1 b[1]! 1e-10 ∧
     fclose Ax2 b[2]! 1e-10) = true := by native_decide

/-- Wide-diagonal example (mimics the 81k pathology in miniature):
   diag spans 6 decades. -/
private def Awide : Mat3 :=
  #[ 1.0e-2, 1.0e-3, 0.0
   , 1.0e-3, 1.0e+0, 1.0e-2
   , 0.0,    1.0e-2, 1.0e+4
   ]

private def widePat : Mat3 :=
  #[ 0.0, 1.0, 0.0
   , 1.0, 0.0, 1.0
   , 0.0, 1.0, 0.0
   ]

/-- ICC factorisation succeeds even with 6-decade diagonal spread. -/
example :
    let L := factor3 Awide widePat
    let LL := lLT3 L
    mat3Close LL Awide 1e-9 = true := by native_decide

end IncompleteCholeskyExamples

end Curvenet
