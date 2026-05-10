# Vendor notes — Slang

In-tree vendor of [shader-slang/slang](https://github.com/shader-slang/slang)
following the project's existing pattern (`thirdparty/godot-cpp/`,
`thirdparty/meshoptimizer/`, `thirdparty/rapidcheck/`). MIT-licensed
(see `LICENSE`).

## Why

Brings Slang as the shader compiler for the Godot deformer's runtime
DDM matvec path. The build pipeline mirrors
[DevPrice/godot-slang](https://github.com/DevPrice/godot-slang)
(`build_slang.py` invokes CMake on this directory, produces a
shared lib, scons links the GDExtension against it).

## Vendored revision

`63207ece82420eeb0f9a606b0739b7d8534a6662` (cloned shallow with
`--recurse-submodules --shallow-submodules`).

## What was trimmed from the upstream tree

- `tests/` (~21 MB, internal test fixtures)
- `docs/`, `extras/`, `docker/`, `typings/` — documentation we don't consume.
- `flake.lock`, `flake.nix`, `REVIEW.md`, `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md`, `REUSE.toml`, `CLAUDE.md` — upstream-only metadata.
- `.git/`, `.github/` (top-level + each submodule).

What stays: `source/`, `include/`, `prelude/`, `tools/` (slangc CLI),
`examples/` (CMake unconditionally `add_subdirectory()`s it; cheap to keep),
`external/` with all submodule contents intact (glslang, spirv-tools,
spirv-headers, vulkan-headers, slang-rhi, etc.), `cmake/`,
`CMakeLists.txt`, `CMakePresets.json`, the license files, and
`slang-tag-version.h.in`.

Total on-disk: ~199 MB.

## Updating

```sh
cd /tmp && rm -rf slang-fresh
git clone --depth 1 --recurse-submodules --shallow-submodules \
  https://github.com/shader-slang/slang.git slang-fresh
cd slang-fresh
rm -rf tests docs extras docker typings \
       flake.lock flake.nix REVIEW.md CONTRIBUTING.md \
       CODE_OF_CONDUCT.md REUSE.toml CLAUDE.md \
       .git .github
find external -name '.git' -exec rm -rf {} + 2>/dev/null
find external -name '.github' -type d -exec rm -rf {} + 2>/dev/null
cd ..
rm -rf <repo>/thirdparty/slang
mv slang-fresh <repo>/thirdparty/slang
```

Update the vendored revision line above with the new commit SHA.
