/-
MIT License — Copyright (c) 2026 K. S. Ernest (iFire) Lee.

Greedy graph coloring — load-bearing primitive for parallel-
within-color symmetric Gauss-Seidel smoother used by HSC's V-cycle.

Background: SGS is fundamentally sequential because the row-i
update reads x[j] for already-processed j < i. To parallelise
on GPU (Vulkan compute on Quest 3 / Apple Metal), we need a
mathematical reformulation: multi-color SGS. Pre-color the
graph so no two adjacent vertices share a color. Within a color,
all row updates are independent and can dispatch in parallel.
Process colors sequentially.

Greedy coloring (Welsh-Powell variant): walk vertices in some
order, assign each the smallest color not used by any already-
colored neighbor. On triangle meshes this typically produces
4-5 colors (since the 4-color theorem bounds it for planar
graphs — but we color on the cot-Laplacian's adjacency, which
for our cut-mesh is planar-equivalent at the level-0 stage).

This file specifies the algorithm on small instances:
  * 3-cycle (triangle): 3 colors needed
  * 4-cycle: 2 colors (bipartite)
  * Path 0-1-2-3-4: 2 colors
  * Star K_{1,4}: 2 colors (center vs leaves)

Native-decide checks verify each vertex's color differs from
all its neighbors.
-/

namespace Curvenet
namespace GraphColoring

/-- Adjacency: per-vertex list of neighbor indices. -/
abbrev Adjacency := Array (Array Nat)

/-- Greedy coloring: walk vertices in index order, assign each
   the smallest color id not used by any earlier neighbor. -/
def greedyColor (adj : Adjacency) : Array Nat := Id.run do
  let n := adj.size
  let mut color : Array Nat := Array.replicate n 0
  for i in [0:n] do
    let nbrs := adj[i]!
    -- Mark which colors are forbidden by already-colored neighbors.
    let mut used : Array Bool := Array.replicate n false
    for j in nbrs do
      if j < i then   -- only earlier-coloured neighbors are decided
        used := used.set! color[j]! true
    -- Find smallest available color.
    let mut c : Nat := 0
    while c < n ∧ used[c]! do
      c := c + 1
    color := color.set! i c
  return color

/-- Number of distinct colors used. -/
def numColors (color : Array Nat) : Nat := Id.run do
  let mut m : Nat := 0
  for c in color do
    if c + 1 > m then m := c + 1
  return m

/-- Check that no two adjacent vertices share a color. -/
def isValidColoring (adj : Adjacency) (color : Array Nat) : Bool := Id.run do
  let mut ok := true
  for i in [0:adj.size] do
    for j in adj[i]! do
      if color[i]! = color[j]! then ok := false
  return ok

end GraphColoring

/- ============================================================ -/
/- Native-decide checks on small graphs.                         -/
/- ============================================================ -/

namespace GraphColoringExamples

open GraphColoring

/-- Triangle 0-1-2: chromatic number 3. -/
private def triangle : Adjacency :=
  #[ #[1, 2], #[0, 2], #[0, 1] ]

example : isValidColoring triangle (greedyColor triangle) = true := by
  native_decide

example : numColors (greedyColor triangle) = 3 := by native_decide

/-- 4-cycle 0-1-2-3-0: bipartite, chromatic number 2. -/
private def cycle4 : Adjacency :=
  #[ #[1, 3], #[0, 2], #[1, 3], #[0, 2] ]

example : isValidColoring cycle4 (greedyColor cycle4) = true := by
  native_decide

example : numColors (greedyColor cycle4) = 2 := by native_decide

/-- Path 0-1-2-3-4: bipartite. -/
private def path5 : Adjacency :=
  #[ #[1], #[0, 2], #[1, 3], #[2, 4], #[3] ]

example : isValidColoring path5 (greedyColor path5) = true := by
  native_decide

example : numColors (greedyColor path5) = 2 := by native_decide

/-- Star K_{1,4}: 0 center connected to 1,2,3,4. Bipartite. -/
private def star4 : Adjacency :=
  #[ #[1, 2, 3, 4], #[0], #[0], #[0], #[0] ]

example : isValidColoring star4 (greedyColor star4) = true := by
  native_decide

example : numColors (greedyColor star4) = 2 := by native_decide

end GraphColoringExamples

end Curvenet
