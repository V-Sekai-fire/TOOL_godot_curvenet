# Vendor notes — Slang

This is an in-tree vendor of [shader-slang/slang](https://github.com/shader-slang/slang)
following the project's existing pattern (`thirdparty/godot-cpp/`,
`thirdparty/meshoptimizer/`, `thirdparty/rapidcheck/`). MIT-licensed
(see `LICENSE`).

## Why

The xr-grid port replaces the Godot GDExtension integration with a
Lean-driven Slang shader-codegen pipeline:

- Lean modules under `lean/Curvenet/` emit `.slang` source for the
  DDM matvec runtime kernel and the bind-time harvest.
- `slangc` lowers each `.slang` to SPIR-V at bind time; the cached
  SPIR-V is mmap'd by the xr-grid Vulkan compute dispatcher.
- No Godot dependency — the runtime library is self-contained.

## Vendored revision

`63207ece82420eeb0f9a606b0739b7d8534a6662` (cloned shallow on 2026-05-10).

## What was trimmed from the upstream tree

- `tests/` (~21 MB, internal test fixtures)
- `docs/`, `examples/`, `extras/`, `docker/`, `typings/` —
  documentation and bindings we don't consume.
- `flake.lock`, `flake.nix`, `REVIEW.md`, `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md`, `REUSE.toml`, `CLAUDE.md` — upstream-only
  metadata.
- `.git/`, `.github/` — clone provenance.

What stays: `source/`, `include/`, `prelude/`, `tools/` (slangc CLI),
`external/`, `cmake/`, `CMakeLists.txt`, `CMakePresets.json`, the
license files, and `slang-tag-version.h.in`.

Total on-disk: ~30 MB.

## Updating

Re-run the trim:
```
cd /tmp && rm -rf slang-probe
git clone --depth 1 https://github.com/shader-slang/slang.git slang-probe
cd slang-probe
rm -rf tests docs examples extras docker typings \
       flake.lock flake.nix REVIEW.md CONTRIBUTING.md \
       CODE_OF_CONDUCT.md REUSE.toml CLAUDE.md .git .github
cd ..
rm -rf <repo>/thirdparty/slang
mv slang-probe <repo>/thirdparty/slang
```

Update the vendored revision line above with the new commit SHA.
