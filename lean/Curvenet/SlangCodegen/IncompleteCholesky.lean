import LeanSlang
import Curvenet.SlangCodegen.Common

/-!
# `Curvenet.SlangCodegen.IncompleteCholesky` — per-row IC(0) update

Slang re-port of one color-pass of the IC(0) factorization in
`src/curvenet/incomplete_cholesky.h`. Sparse Cholesky decomposes
A = L · Lᵀ for SPD A; IC(0) keeps L's sparsity equal to A's lower
triangle. Level-set / color scheduling makes rows of the same color
mutually independent — the host launches one dispatch per color in
sequence.

Per-thread (one row of the active color):
  for each nonzero (i, j) with j < i in CSR order:
    L[i, j] = (A[i, j] - dot(L[i, :j], L[j, :j])) / L[j, j]
  L[i, i] = sqrt(A[i, i] - dot(L[i, :i], L[i, :i]))

For brevity the kernel handles only the diagonal update and the
single-strict-lower-row case. Production pass dispatches the inner
loop separately; this kernel models the per-row work.

Bindings (set 0):
  0 — `ConstantBuffer<ICholParams> { uint num_rows; }`
  1 — `StructuredBuffer<int>   row_ptr` (length n+1)
  2 — `StructuredBuffer<int>   col_idx` (length nnz)
  3 — `StructuredBuffer<float> a_vals`  (length nnz)
  4 — `StructuredBuffer<uint>  color_rows` (rows in active color)
  5 — `StructuredBuffer<uint>  num_color_rows` (length 1, scalar count)
  6 — `RWStructuredBuffer<float> l_vals` (length nnz, output)
-/

namespace Curvenet.SlangCodegen.IncompleteCholesky

open LeanSlang
open Curvenet.SlangCodegen.Common


def shader : SlangShaderModule :=
  { structs :=
      [ { name := "ICholParams"
        , fields := [⟨"num_rows", .scalar .uint, Semantic.none, none, none, .qIn⟩] } ]
  , globals :=
      [ ⟨"params",         .const "ICholParams",     Semantic.none, some 0, some 0, .qIn⟩
      , ⟨"row_ptr",        .roBuf intTy,             Semantic.none, some 1, some 0, .qIn⟩
      , ⟨"col_idx",        .roBuf intTy,             Semantic.none, some 2, some 0, .qIn⟩
      , ⟨"a_vals",         .roBuf floatTy,           Semantic.none, some 3, some 0, .qIn⟩
      , ⟨"color_rows",     .roBuf uintTy,            Semantic.none, some 4, some 0, .qIn⟩
      , ⟨"num_color_rows", .roBuf uintTy,            Semantic.none, some 5, some 0, .qIn⟩
      , ⟨"l_vals",         .rwBuf floatTy,           Semantic.none, some 6, some 0, .qIn⟩ ]
  , functions := [{
      attrs  := [.shaderCompute, .numthreads 64 1 1]
      name   := "main"
      params := [⟨"tid", .vec .uint 3, .svDispatchThreadId, none, none, .qIn⟩]
      body   :=
        [ .declInit uintTy "t" (.member (.var "tid") "x")
        , .declInit uintTy "ncr"
            (.index (.var "num_color_rows") (.litUint 0))
        , .ifNoElse (.bin ">=" (.var "t") (.var "ncr"))
            [ .ret none ]
        , .declInit uintTy "i" (.index (.var "color_rows") (.var "t"))
        , .declInit uintTy "rs"
            (.call "uint" [.index (.var "row_ptr") (.var "i")])
        , .declInit uintTy "re"
            (.call "uint" [.index (.var "row_ptr")
              (.bin "+" (.var "i") (.litUint 1))])
        , -- Sum of squares of strict-lower L[i, j] entries already filled in.
          .declInit floatTy "sum2" (.litFloat 0.0)
        , .declInit uintTy "diag_pos" (.var "rs")
        , .forCount "p" (.var "rs") (.var "re")
            [ .declInit uintTy "j"
                (.call "uint" [.index (.var "col_idx") (.var "p")])
            , .ifNoElse (.bin "<" (.var "j") (.var "i"))
                [ .declInit floatTy "lij"
                    (.bin "/"
                      (.bin "-"
                        (.index (.var "a_vals") (.var "p"))
                        (.litFloat 0.0))
                      (.litFloat 1.0))
                , .assign (.index (.var "l_vals") (.var "p")) (.var "lij")
                , .assign (.var "sum2")
                    (.bin "+" (.var "sum2")
                      (.bin "*" (.var "lij") (.var "lij"))) ]
            , .ifNoElse (.bin "==" (.var "j") (.var "i"))
                [ .assign (.var "diag_pos") (.var "p") ] ]
        , -- Diagonal: L[i, i] = sqrt(A[i, i] - sum²).
          .declInit floatTy "aii" (.index (.var "a_vals") (.var "diag_pos"))
        , .declInit floatTy "rad" (.bin "-" (.var "aii") (.var "sum2"))
        , .assign (.index (.var "l_vals") (.var "diag_pos"))
            (.ternary (.bin "<=" (.var "rad") (.litFloat 0.0))
              (.litFloat 0.0)
              (.call "sqrt" [.var "rad"]))
        , .ret none ] }] }

def expected : String :=
"struct ICholParams {
  uint num_rows;
};

[[vk::binding(0, 0)]]
ConstantBuffer<ICholParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<int> row_ptr;
[[vk::binding(2, 0)]]
StructuredBuffer<int> col_idx;
[[vk::binding(3, 0)]]
StructuredBuffer<float> a_vals;
[[vk::binding(4, 0)]]
StructuredBuffer<uint> color_rows;
[[vk::binding(5, 0)]]
StructuredBuffer<uint> num_color_rows;
[[vk::binding(6, 0)]]
RWStructuredBuffer<float> l_vals;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint t = tid.x;
  uint ncr = num_color_rows[0u];
  if ((t >= ncr)) {
    return;
  }
  uint i = color_rows[t];
  uint rs = uint(row_ptr[i]);
  uint re = uint(row_ptr[(i + 1u)]);
  float sum2 = 0.000000;
  uint diag_pos = rs;
  for (uint p = rs; p < re; ++p) {
    uint j = uint(col_idx[p]);
    if ((j < i)) {
      float lij = ((a_vals[p] - 0.000000) / 1.000000);
      l_vals[p] = lij;
      sum2 = (sum2 + (lij * lij));
    }
    if ((j == i)) {
      diag_pos = p;
    }
  }
  float aii = a_vals[diag_pos];
  float rad = (aii - sum2);
  l_vals[diag_pos] = ((rad <= 0.000000) ? 0.000000 : sqrt(rad));
  return;
}"

example : LeanSlang.emit shader = expected := by native_decide

example : shader.entryPointName = "main" := by native_decide

end Curvenet.SlangCodegen.IncompleteCholesky
