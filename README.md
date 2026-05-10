# TOOL_godot_curvenet

> Lean 4 spec + Slang shader codegen for Pixar's **Profile Curves**
> character articulation algorithm (de Goes, Sheffler, Fleischer,
> *SIGGRAPH 2022*).
>
> Lean is the source of truth — every algorithm is a `def` with
> `native_decide` proofs of the load-bearing invariants. From those
> specs the project emits **Slang** compute shaders that round-trip
> through `slangc` to SPIR-V and Apple Metal `.metallib`, with
> spec ≡ kernel equivalence proven via a slangc-cpp + `@[extern]`
> bridge into Lean.

[**Engineering reference: `docs/DEVELOPING.md`**](docs/DEVELOPING.md)
— architecture, build, validation pipeline, CI, layer mapping.

[**References: `references.bib`**](references.bib) — DeGoes22,
Nguyen23, Le-Lewis 2019 (DDM runtime), Krishnan-Fattal-Szeliski
2013 (HSC inner-CG preconditioner), and 31 more.
