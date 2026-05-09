/-
MIT License — Copyright (c) 2026 K. S. Ernest (iFire) Lee.

Two-level additive Schwarz operators.

The 1-level Schwarz with 1-ring overlap stalls at the deformer's
PCVR-target mesh size (81k verts) because information has to walk
through hundreds of meshlets at one ring per outer iter. Standard
fix per Smith-Bjørstad-Gropp 1996 and Wu 2022: add a coarse
correction step. Each meshlet contributes one coarse "node"; the
coarse problem A_c · y_c = R · r is small enough to solve directly,
and prolonging y_c back to the fine grid corrects the global
low-frequency modes that 1-level Schwarz can't reach.

Operators:

  R  : restriction      (fine → coarse), R[c, i] = 1 iff vert i
       belongs to coarse node c, 0 otherwise. Each fine vert
       belongs to exactly one coarse node (the meshlet that owns
       it under our 1-ring partition).

  Rᵀ : prolongation     (coarse → fine), distributes coarse value
       to all fine verts in the corresponding meshlet.

  A_c = R · A · Rᵀ      Galerkin coarse-matrix construction
       (preserves SPD; standard choice for symmetric problems).

This file specifies the operators on small instances (4-vert input,
2-meshlet partition) and verifies via native_decide:
  * R · v sums fine values per coarse cluster
  * Rᵀ · w distributes coarse values to all members
  * R · (Rᵀ · w) = sizes(c) · w (each coarse node gets sum of its
    own constant value over its members = members · value)
  * (R · A · Rᵀ) is symmetric when A is
  * A_c row-sum is zero when A's row-sums are (constants in kernel)
-/

import Curvenet.Vec3

namespace Curvenet
namespace TwoLevelSchwarz

/-- `coarseOf` is a vector indexed by fine-grid vertex; each entry
   is the coarse node (meshlet id) the vertex belongs to. -/
abbrev CoarseMap := Array Nat

/-- Restrict a fine vector to a coarse vector by summing values
   over each cluster: `c[meshlet(i)] += f[i]`. -/
def restrict (cmap : CoarseMap) (numCoarse : Nat) (f : Array Float) : Array Float := Id.run do
  let mut c : Array Float := Array.replicate numCoarse 0.0
  for i in [0:f.size] do
    let cid := cmap[i]!
    c := c.set! cid (c[cid]! + f[i]!)
  return c

/-- Prolong a coarse vector to fine values by broadcasting each
   coarse value to its cluster members: `f[i] = c[meshlet(i)]`. -/
def prolong (cmap : CoarseMap) (numFine : Nat) (c : Array Float) : Array Float := Id.run do
  let mut f : Array Float := Array.replicate numFine 0.0
  for i in [0:numFine] do
    let cid := cmap[i]!
    f := f.set! i c[cid]!
  return f

/-- Cluster sizes (number of fine verts in each coarse node). -/
def coarseSizes (cmap : CoarseMap) (numCoarse : Nat) : Array Nat := Id.run do
  let mut s : Array Nat := Array.replicate numCoarse 0
  for i in [0:cmap.size] do
    s := s.set! cmap[i]! (s[cmap[i]!]! + 1)
  return s

/-- Galerkin coarse-matrix construction: `A_c = R · A · Rᵀ`.
   Equivalently `A_c[c1, c2] = Σ_{i,j with cmap[i]=c1, cmap[j]=c2} A[i, j]`.
   Matrix stored as flat row-major. -/
def galerkin (cmap : CoarseMap) (numCoarse : Nat)
              (n : Nat) (A : Array Float) : Array Float := Id.run do
  let mut Ac : Array Float := Array.replicate (numCoarse * numCoarse) 0.0
  for i in [0:n] do
    for j in [0:n] do
      let c1 := cmap[i]!
      let c2 := cmap[j]!
      let v  := A[i * n + j]!
      Ac := Ac.set! (c1 * numCoarse + c2) (Ac[c1 * numCoarse + c2]! + v)
  return Ac

/-- Component-wise approximate equality. -/
def vecClose (a b : Array Float) (eps : Float) : Bool := Id.run do
  if a.size ≠ b.size then return false
  for i in [0:a.size] do
    if (a[i]! - b[i]!).abs ≥ eps then return false
  return true

/-- Matrix is symmetric within `eps`. -/
def matIsSymmetric (n : Nat) (A : Array Float) (eps : Float) : Bool := Id.run do
  for i in [0:n] do
    for j in [i+1:n] do
      if (A[i * n + j]! - A[j * n + i]!).abs ≥ eps then return false
  return true

/-- Sum of each row of an n×n matrix. -/
def rowSum (n : Nat) (A : Array Float) : Array Float := Id.run do
  let mut s : Array Float := Array.replicate n 0.0
  for i in [0:n] do
    let mut t : Float := 0.0
    for j in [0:n] do
      t := t + A[i * n + j]!
    s := s.set! i t
  return s

end TwoLevelSchwarz

/- ============================================================ -/
/- Concrete `native_decide` checks on 4-vert, 2-meshlet inputs. -/
/- ============================================================ -/

namespace TwoLevelSchwarzExamples

open TwoLevelSchwarz

/-- 4 fine verts split into 2 meshlets: {0, 1} → coarse 0, {2, 3} → 1. -/
private def cmap2x2 : CoarseMap := #[0, 0, 1, 1]

/-- restrict sums fine values per cluster: f = (1, 2, 3, 4) → c = (3, 7). -/
example :
    let f := #[1.0, 2.0, 3.0, 4.0]
    let c := restrict cmap2x2 2 f
    vecClose c #[3.0, 7.0] 1e-12 = true := by native_decide

/-- prolong distributes coarse to all members: c = (5, 9) →
   f = (5, 5, 9, 9). -/
example :
    let c := #[5.0, 9.0]
    let f := prolong cmap2x2 4 c
    vecClose f #[5.0, 5.0, 9.0, 9.0] 1e-12 = true := by native_decide

/-- restrict ∘ prolong on a coarse vector multiplies each entry by
   its cluster size: with sizes (2, 2), c = (5, 9) goes through
   prolong → (5,5,9,9), then restrict → (10, 18). -/
example :
    let c := #[5.0, 9.0]
    let rt := restrict cmap2x2 2 (prolong cmap2x2 4 c)
    vecClose rt #[10.0, 18.0] 1e-12 = true := by native_decide

/-- coarseSizes returns the cluster member counts. -/
example :
    let s := coarseSizes cmap2x2 2
    s = #[2, 2] := by native_decide

/-- Galerkin on a 4×4 SPD-like matrix produces a 2×2 coarse matrix
   that's symmetric. Test matrix is the 4-vertex 1D Laplacian
   (`tridiag(-1, 2, -1)` with Dirichlet boundary), block-aggregated:
   coarse 0 = {0, 1}, coarse 1 = {2, 3}. -/
example :
    let A : Array Float :=
      #[ 2.0, -1.0,  0.0,  0.0,
         -1.0,  2.0, -1.0,  0.0,
          0.0, -1.0,  2.0, -1.0,
          0.0,  0.0, -1.0,  2.0]
    let Ac := galerkin cmap2x2 2 4 A
    matIsSymmetric 2 Ac 1e-12 = true := by native_decide

/-- Galerkin on a graph Laplacian preserves the row-sum-zero property
   at the coarse level (constants stay in the kernel after coarsening).
   Test: 4-vertex path graph Laplacian, 0-1-2-3. Rows sum to zero;
   aggregated by `cmap2x2` the coarse matrix should also have
   row-sum zero. (The earlier draft used a Dirichlet 1D Laplacian
   whose boundary rows sum to +1, not 0, and rightly failed.) -/
example :
    let A : Array Float :=
      #[ 1.0, -1.0,  0.0,  0.0,
         -1.0,  2.0, -1.0,  0.0,
          0.0, -1.0,  2.0, -1.0,
          0.0,  0.0, -1.0,  1.0]
    let Ac := galerkin cmap2x2 2 4 A
    let s  := rowSum 2 Ac
    vecClose s #[0.0, 0.0] 1e-12 = true := by native_decide

end TwoLevelSchwarzExamples

end Curvenet
