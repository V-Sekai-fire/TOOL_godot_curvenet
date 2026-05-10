import LeanSlang
import Curvenet.DirectDeltaMush

/-!
# `Curvenet.SlangCodegen` — DDM matvec compute kernel

Lifts the runtime kernel specified in `Curvenet.DirectDeltaMush` (the
LBS-flavored matvec `pos[v] = Σᵢ W[v,i] · F_i · rest[v]`) into a
Slang `[shader("compute")]` source via `LeanSlang`.

The CPU-side `Influences` (an `Array (Nat × Float)` per vertex) is
flattened on the host into CSR layout for the GPU:

- `restPos    : StructuredBuffer<float3>`     — input rest positions
- `outPos     : RWStructuredBuffer<float3>`   — deformed output
- `transforms : StructuredBuffer<float4x4>`   — per-handle affines
- `inflStart  : StructuredBuffer<uint>`       — CSR row pointer (length V+1)
- `inflIdx    : StructuredBuffer<uint>`       — flat handle indices, total nnz
- `inflW      : StructuredBuffer<float>`      — flat weights, total nnz

One thread per vertex; bounds-check is the host's job (dispatch
exactly `ceilDiv V 64` groups of 64).

The pinned reference text below is asserted by `native_decide`. Any
drift in `LeanSlang.Emit` that affects this output trips here.
-/

namespace Curvenet.SlangCodegen

open LeanSlang

/-- The DDM matvec compute kernel as a Slang shader module. -/
def directDeltaMushKernel : SlangShaderModule :=
  let f3 : SlangType := .vec .float 3
  let f4 : SlangType := .vec .float 4
  let m4 : SlangType := .mat .float 4 4
  let u  : SlangType := .scalar .uint
  let f  : SlangType := .scalar .float
  let bnd (n : Nat) (name : String) (t : SlangType) : SlangBinding :=
    { name := name, type := t, semantic := Semantic.none
    , binding := some n, space := some 0 }
  let body : List SlangStmt :=
    [ .declInit u  "v"   (.member (.var "tid") "x")
    , .declInit f3 "r"   (.index  (.var "restPos") (.var "v"))
    , .declInit f3 "acc" (.call "float3" [.litFloat 0.0, .litFloat 0.0, .litFloat 0.0])
    , .declInit u  "s"   (.index  (.var "inflStart") (.var "v"))
    , .declInit u  "e"   (.index  (.var "inflStart")
                                  (.bin "+" (.var "v") (.litUint 1)))
    , .forCount "k" (.var "s") (.var "e")
        [ .declInit u  "i"  (.index (.var "inflIdx") (.var "k"))
        , .declInit f  "w"  (.index (.var "inflW")   (.var "k"))
        , .declInit m4 "T"  (.index (.var "transforms") (.var "i"))
        , .declInit f4 "rh"
            (.call "float4"
              [ .member (.var "r") "x"
              , .member (.var "r") "y"
              , .member (.var "r") "z"
              , .litFloat 1.0 ])
        , .declInit f4 "t" (.call "mul" [.var "T", .var "rh"])
        , .assign (.member (.var "acc") "x")
            (.bin "+" (.member (.var "acc") "x")
                      (.bin "*" (.var "w") (.member (.var "t") "x")))
        , .assign (.member (.var "acc") "y")
            (.bin "+" (.member (.var "acc") "y")
                      (.bin "*" (.var "w") (.member (.var "t") "y")))
        , .assign (.member (.var "acc") "z")
            (.bin "+" (.member (.var "acc") "z")
                      (.bin "*" (.var "w") (.member (.var "t") "z")))
        ]
    , .assign (.index (.var "outPos") (.var "v")) (.var "acc")
    ]
  { globals :=
      [ bnd 0 "restPos"    (.roBuf f3)
      , bnd 1 "outPos"     (.rwBuf f3)
      , bnd 2 "transforms" (.roBuf m4)
      , bnd 3 "inflStart"  (.roBuf u)
      , bnd 4 "inflIdx"    (.roBuf u)
      , bnd 5 "inflW"      (.roBuf f)
      ]
  , functions := [{
      attrs  := [.shaderCompute, .numthreads 64 1 1]
      name   := "main"
      params := [{ name := "tid", type := .vec .uint 3
                 , semantic := Semantic.svDispatchThreadId
                 , binding := none, space := none }]
      body   := body
    }] }

/-- Pinned reference emission. Drift in `LeanSlang.Emit` trips
    `native_decide` below. Update both this string and the kernel
    in lockstep. -/
def directDeltaMushKernelExpected : String :=
"[[vk::binding(0, 0)]]
StructuredBuffer<float3> restPos;
[[vk::binding(1, 0)]]
RWStructuredBuffer<float3> outPos;
[[vk::binding(2, 0)]]
StructuredBuffer<float4x4> transforms;
[[vk::binding(3, 0)]]
StructuredBuffer<uint> inflStart;
[[vk::binding(4, 0)]]
StructuredBuffer<uint> inflIdx;
[[vk::binding(5, 0)]]
StructuredBuffer<float> inflW;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint v = tid.x;
  float3 r = restPos[v];
  float3 acc = float3(0.000000, 0.000000, 0.000000);
  uint s = inflStart[v];
  uint e = inflStart[(v + 1u)];
  for (uint k = s; k < e; ++k) {
    uint i = inflIdx[k];
    float w = inflW[k];
    float4x4 T = transforms[i];
    float4 rh = float4(r.x, r.y, r.z, 1.000000);
    float4 t = mul(T, rh);
    acc.x = (acc.x + (w * t.x));
    acc.y = (acc.y + (w * t.y));
    acc.z = (acc.z + (w * t.z));
  }
  outPos[v] = acc;
}"

/-- The pretty-printer matches the pinned reference. -/
example :
    LeanSlang.emit directDeltaMushKernel = directDeltaMushKernelExpected := by
  native_decide

/-- The kernel's entry-point name is `main`. -/
example : directDeltaMushKernel.entryPointName = "main" := by native_decide

end Curvenet.SlangCodegen
