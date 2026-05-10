# Trust Topology audit — TOOL_godot_curvenet

Audit of this project's verification pipeline using Michael Rothrock's
[Trust Topology framework](https://michael.roth.rocks/research/trust-topology).
The framework treats reliability as a property of how verification
gates are arranged, not of any single model or check inside them, and
provides four diagnostic properties: **overlap ratio**, **verification
amplification**, the **deterministic ceiling**, and the **liveness
constraint**.

The audit was sequenced via taskweft —
[`trust_topology_audit` domain + `trust_topology_audit_curvenet`
problem](https://github.com/V-Sekai-fire/taskweft-plans/pull/2). The
planner returned the same 7-step decomposition that drives the
sections below (inventory → 4 properties → synthesis → publish).

## 1 — Gate inventory

Every shipped change passes through five verification gates, in this
order:

| # | Gate                          | Kind          | Source of truth | Coverage |
|---|-------------------------------|---------------|------------------|----------|
| G1 | Lean `native_decide` proofs  | deterministic | `lean/Curvenet/`   | 23 modules · 258 proofs |
| G2 | RapidCheck property tests    | stochastic    | `tests/test_*.cpp` | 13 binaries · 92 properties |
| G3 | `lake build`                 | deterministic | Lean kernel        | 26 jobs |
| G4 | `scons -j8` / `make -C tests test` | deterministic | clang + linker | 0 warnings |
| G5 | Manual editor verification   | stochastic    | human eye          | Mire rig in Godot 4.x |

There is no *automated* equivalent of G5 — checking that a deformed
Mire mesh "looks right" requires a human. Per the framework, that
sits explicitly above the deterministic ceiling and is fine — but it
must be acknowledged, not glossed over.

## 2 — Overlap ratio

> Two gates with 80% overlap are one gate that runs twice.

**G1 (Lean) vs G2 (RC tests).** Both check algorithmic correctness.
On the surface they look redundant, but their *error-class coverage*
differs:

- G1 catches **algorithmic specification bugs**: small instances
  hand-curated to exercise each branch. If the cot-Laplacian formula
  is wrong, a 3-vertex equilateral-triangle proof catches it
  immediately.
- G2 catches **implementation drift**: random instances test that
  the C++ code matches the Lean spec under inputs the Lean fixtures
  don't reach. If a refactor introduces an off-by-one in CSR
  scattering, RC eventually finds it.

Estimated overlap: ~30%. Both detect catastrophic algorithmic bugs
(symmetric matrix asymmetric, etc.). The 70% complement is real —
during the ~100-loop preconditioner trajectory, RC props caught
several drift bugs that Lean's small-instance proofs couldn't reach,
and Lean caught at least one specification mistake that RC's random
inputs never triggered.

**G3 (lake) vs G1 (Lean proofs).** No overlap. lake just confirms
the proofs *compile*; the proofs themselves are G1.

**G4 (scons / RC) vs G2.** Some overlap — RC binaries are built by
the test target, so a build break is caught at both. ~20%.

**G5 (manual) vs G1-G4.** Near-zero overlap. Manual verification
covers what is structurally below the ceiling: artist-felt deformation
quality, gizmo responsiveness, intuitive frame placement. None of the
automated gates can see this.

**Verdict**: gate composition is healthy. Each gate contributes a
distinct error class. The biggest *latent* overlap risk is between
Lean and RC if both drift toward "verify the same fixture" — keep RC
inputs randomized rather than copy-pasting Lean fixtures.

## 3 — Verification amplification

> A weak upstream gate is the most expensive place to have a gap.

The pipeline runs **upstream → downstream** as:

```
Lean spec  →  C++ mirror  →  RC tests  →  scons build  →  Manual editor
   G1            (impl)         G2            G4              G5
```

G1 (Lean) is the upstream gate. It is by design the most *constrained*
input space (small fixtures, decidable equality with eps), so its
rejection rate is high during early development — analogous to
Rothrock's plan gate.

Once G1 passes, every downstream gate operates on an input that is
already known to be specification-consistent. RC (G2) doesn't need
to ask "is the algorithm right?" — only "does the C++ implementation
match what Lean already proved?" That's a strictly narrower question.

**Failure mode this prevents**: a buggy specification that passes
all tests because the tests were derived from the buggy spec. With
Lean upstream, tests are forced to verify against an *independently
proved* reference.

**Failure mode it can't prevent**: a Lean spec that doesn't capture
all the relevant properties. If the spec under-specifies, tests can
trivially conform.

**Verdict**: amplification is well-utilized for the algorithmic
layer. The gap is at the layer above: there is no gate between
"matches the Lean spec" and "produces a deformation an artist
accepts on Mire". Currently G5 (manual) bridges that, but it's the
only gate doing so.

## 4 — Deterministic ceiling

> Structural correctness is not semantic correctness.

The ceiling — the line above which only stochastic / unobservable
checks remain — sits **above G4 and below G5** in this project:

- **Below the ceiling (provable)**: Lean proofs, lake build, scons
  build, RC props that test exact numerical equality within eps.
  These either pass or they don't; the failure mode is structural.
- **Sharp boundary**: when an RC property fails, the failure is
  reproducible — same seed, same counter-example. When a manual
  verification fails, the artist says "it looks wrong" and the agent
  has to decompose what that means.
- **Above the ceiling (estimated)**: deformation quality on real
  rigs. The deformer can compile, pass all tests, and *still*
  produce visually wrong output if the §3 frame composition is too
  far from artist intent.
- **Above the ceiling (unobservable)**: artist intent itself. No
  pipeline stage has direct access to what the artist *meant* by
  drawing this profile curve here. Each artifact (curve, weights,
  deformed mesh) is a lower-fidelity projection of intent.

**The pipeline guarantees fidelity to specification, not fidelity
to intent.** This is exactly Rothrock's framing of the source-of-
truth boundary, and it's why oracle routing (escalation to a human)
is not optional.

**Verdict**: the ceiling is correctly drawn. The risk is treating
G5 as if it were below the ceiling — it isn't. Specifically:
- Don't write tests that try to *automate* "deformation looks right".
  That's a category error.
- Do write tests that automate observable invariants (no NaN, partition
  of unity, identity Fc preserves rest pose). Those are below the
  ceiling.

## 5 — Liveness constraint

> Each gate narrows the space of acceptable output. Add too many,
> the system thrashes in retries.

Empirical liveness for this session: most commits pass G1–G4 on the
first try. Failures so far (across this session's work):

- Lean compile errors during `DirectDeltaMushBind.lean` (Float
  decidability) — caught at G1, took one retry to fix.
- Lean compile error during `IntersectionFrames.lean` (mysterious
  "unterminated comment") — caught at G1; required dropping the
  block-comment section header for a `--` line comment, two retries.
- C++ link error on `test_direct_delta_mush` (called wrong make
  target) — caught at G4, one retry.

That's roughly 3 retries across ~10 substantive commits, each retry
< 1 minute. The system is well inside its liveness budget.

**Could we add another gate?** Per Rothrock, every additional gate
is a throughput cost. Candidates that have come up but haven't been
added:

- Pre-commit hooks that run `lake build` + `make test`. Would catch
  G1/G2 failures earlier (before commit). Cost: ~30s on every
  commit. **Verdict**: worth doing.
- A real-rig deformation-quality gate (an automated render of Mire
  with a fixed curve set, image-diffed against a baseline). Cost:
  ~minutes per run; substantial setup. Closes the gap between G4
  and G5. **Verdict**: worth doing eventually but liveness cost is
  real; defer until G5 misses become observable.

**Verdict**: liveness is healthy. Adding a CI runner for G1+G2 is
the cheapest available improvement.

## 6 — Synthesis & recommendations

Summary of findings:

1. **Gate composition is healthy.** G1–G5 each catch a different
   error class with low overlap. The 5-gate topology is in
   equilibrium.

2. **The biggest gap is between G4 and G5.** Today there is no
   automated gate testing whether the deformer produces *correct*
   output on the Mire rig — only whether the code that runs the
   deformer compiles and matches its Lean spec. G5 (manual) fills
   this but doesn't scale and isn't run on every commit.

3. **G1 → G2 amplification is correctly utilized.** Lean upstream
   forces RC tests to verify against a proved reference rather than
   self-consistency.

4. **The deterministic ceiling is correctly drawn.** Don't try to
   automate G5. Do add gates *below* the ceiling that catch more
   invariants automatically.

5. **No oracle routing exists yet.** When Mire deforms wrong, there
   is no structured escalation: it's a human spotting it in the
   editor and filing an issue. Per the framework, this should be
   explicit.

### Recommendations (ranked by cost/benefit)

| Priority | Action | Cost | Benefit |
|---|---|---|---|
| High | CI workflow that runs `lake build` + `make -C tests test` + `scons -j8` on every push | ~30 min setup, ~3 min per push | Closes the "I forgot to run the gates locally" failure mode |
| High | Image-diff regression test: Mire rig + fixed curves → render → compare | ~1 day setup, ~30 s per run | First automated gate for deformation quality (between G4 and G5) |
| Medium | Multi-platform CI (Linux + Windows builds) | ~1 day setup, ~5 min per push | Catches `unknown_unknowns/05_multi_platform_builds` issues automatically |
| Medium | Property test: random Curve3D + random mesh → no NaN in output | ~2 hours setup, ~5 s per run | Below-the-ceiling invariant currently uncovered |
| Low | Make G5 (manual editor check) a checklist in PR template | ~30 min setup, ~5 min per PR | Forces consideration of artist-felt quality before merge |
| Low | Document the oracle-routing path: when G5 fails, what's the protocol? | 1 hour | Makes the existing implicit escalation explicit |

The high-priority items each add a gate that closes a structurally
identifiable gap; the medium and low items are quality-of-life.

## See also

- [`docs/features/`](features/) — what we ship + known gaps
- [Trust Topology framework (Rothrock)](https://michael.roth.rocks/research/trust-topology)
- [taskweft-plans PR #2](https://github.com/V-Sekai-fire/taskweft-plans/pull/2)
  — the audit pipeline as a reusable HTN domain
- [Archive repo](https://github.com/V-Sekai-fire/TOOL_godot_curvenet_archive)
  — retired-trajectory record (the algorithms that *did* fail their
  gates)
