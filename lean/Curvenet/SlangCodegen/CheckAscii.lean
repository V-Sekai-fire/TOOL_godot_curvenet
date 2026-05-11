import LeanSlang

/-!
# `Curvenet.SlangCodegen.CheckAscii` — pure-ASCII predicate over a uint32-packed byte buffer

Slang port of cuJSON's `checkAscii` kernel from `parse_standard_json.cu`
(AutomataLab/cuJSON, MIT). Reduces a uint32 view of an arbitrary byte
buffer to a single bit: `1` iff any byte has its top bit set (i.e. the
input is *not* pure ASCII), `0` otherwise.

The cuJSON CUDA version uses `__shared__` + `atomicOr` to aggregate
across a thread block. Slang's cpp target doesn't support
`InterlockedOr` (atomics are GPU-only), so the kernel is restructured:
each thread accumulates a local OR over its assigned uint32 words and
writes one flag (`0` / `1`) to its own output slot. The host (or a
follow-on reduction kernel) does the final any-bit-set fold over the
flag array. The algorithm is identical — the parallelism just stays
embarrassingly parallel through the kernel boundary instead of being
merged inside it.

Bindings (set 0):
  binding 0 — `ConstantBuffer<CheckAsciiParams> { uint nWords; uint wordsPerThread; }`
  binding 1 — `StructuredBuffer<uint> data`     (input, uint32-packed bytes)
  binding 2 — `RWStructuredBuffer<uint> flags`  (length = nThreads, 0 or 1 per slot)

Reference (cuJSON, `parse_standard_json.cu`):

```cuda
__global__
void checkAscii(uint32_t* blockCompressed_GPU, uint64_t size,
                int total_padded_32, bool* hastUTF8, int WORDS) {
  int threadId = threadIdx.x;  __shared__ uint32_t shared_flag;
  if (threadId == 0) shared_flag = 0;
  __syncthreads();
  int index  = blockIdx.x * blockDim.x + threadId;
  int stride = blockDim.x * gridDim.x;
  for (long i = index; i < total_padded_32; i += stride) {
    int start = i * WORDS;
    for (int j = start; j < size && j < start + WORDS; j++)
      if ((blockCompressed_GPU[j] & 0x80808080) != 0) atomicOr(&shared_flag, 1);
    __syncthreads();
  }
  if (threadId == 0 && shared_flag) *hastUTF8 = true;
}
```
-/

namespace Curvenet.SlangCodegen.CheckAscii

open LeanSlang

def shader : SlangShaderModule :=
  { structs :=
      [ { name := "CheckAsciiParams"
        , fields :=
            [ ⟨"nWords",         .scalar .uint, Semantic.none, none, none, .qIn⟩
            , ⟨"wordsPerThread", .scalar .uint, Semantic.none, none, none, .qIn⟩ ] } ]
  , globals :=
      [ ⟨"params", .const "CheckAsciiParams", Semantic.none, some 0, some 0, .qIn⟩
      , ⟨"data",   .roBuf (.scalar .uint),    Semantic.none, some 1, some 0, .qIn⟩
      , ⟨"flags",  .rwBuf (.scalar .uint),    Semantic.none, some 2, some 0, .qIn⟩ ]
  , functions := [{
      attrs  := [.shaderCompute, .numthreads 256 1 1]
      name   := "main"
      params := [⟨"tid", .vec .uint 3, .svDispatchThreadId, none, none, .qIn⟩]
      body   :=
        [ .declInit (.scalar .uint) "start"
            (.bin "*" (.member (.var "tid") "x") (.member (.var "params") "wordsPerThread"))
        , .declInit (.scalar .uint) "endIdx"
            (.call "min"
              [ .bin "+" (.var "start") (.member (.var "params") "wordsPerThread")
              , .member (.var "params") "nWords" ])
        , .declInit (.scalar .uint) "localAcc" (.litUint 0)
        , .forCount "j" (.var "start") (.var "endIdx")
            [ .assign (.var "localAcc")
                (.bin "|" (.var "localAcc") (.index (.var "data") (.var "j"))) ]
        , .assign (.index (.var "flags") (.member (.var "tid") "x"))
            (.ternary
              (.bin "!=" (.bin "&" (.var "localAcc") (.litUint 0x80808080)) (.litUint 0))
              (.litUint 1) (.litUint 0))
        ] }] }

def expected : String :=
"struct CheckAsciiParams {
  uint nWords;
  uint wordsPerThread;
};

[[vk::binding(0, 0)]]
ConstantBuffer<CheckAsciiParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<uint> data;
[[vk::binding(2, 0)]]
RWStructuredBuffer<uint> flags;

[shader(\"compute\")] [numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint start = (tid.x * params.wordsPerThread);
  uint endIdx = min((start + params.wordsPerThread), params.nWords);
  uint localAcc = 0u;
  for (uint j = start; j < endIdx; ++j) {
    localAcc = (localAcc | data[j]);
  }
  flags[tid.x] = (((localAcc & 2155905152u) != 0u) ? 1u : 0u);
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide

end Curvenet.SlangCodegen.CheckAscii
