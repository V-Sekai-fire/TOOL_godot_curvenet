# 05 — Multi-platform builds

## What we don't know

The project is developed and tested on **macOS only** (Apple
Silicon / M2 Pro). Windows, Linux, Android (Quest 3), and iOS
behavior are entirely untested. The GDExtension framework targets
all of them via `scons`, but only the macOS path has been exercised.

## What might break

- C++14 + `-stdlib=libc++` is macOS-specific. Linux defaults differ
  on libstdc++ vs libc++ ABI; Windows MSVC has its own ABI. We may
  hit STL behavior differences (`std::vector` allocation patterns,
  `std::sort` stability, etc.).
- Float endianness: little-endian assumed throughout. ARM big-endian
  is exotic but possible in some embedded contexts.
- Multi-threaded code in `hierarchical_sparsify.h` uses `std::thread`
  and shared-nothing parallel HSC. Thread-pool semantics and
  scheduler behavior differ across platforms.
- Vulkan compute path (planned for Quest 3 — see
  [known unknowns 03](../known_unknowns/03_quest3_gpu_dispatch.md))
  has zero exposure to Adreno drivers. Quest 3's GPU compute support
  is robust but quirky on edge cases.
- `EditorUndoRedoManager` API surface may differ across Godot
  versions on different platforms (godot-cpp pinned but not
  cross-validated).
- File I/O paths: the bind cache (when implemented in
  [known unknowns 08](../known_unknowns/08_bind_cache_persistence.md))
  needs a writable sidecar location. Sandbox rules differ across
  desktops, and Quest 3 / mobile have entirely different storage
  models.

## How we'd find out

- CI matrix: macOS / Linux / Windows builds via GitHub Actions running
  the same `scons -j8` and `make -C tests test`.
- Manual test pass on Linux desktop + Windows desktop + Quest 3
  device.
- Run `tests/test_*` binaries on each platform; failures reveal
  ABI / STL / numerics issues.
- Stress test on Quest 3 specifically: long-running scene with
  drags, monitor for crashes / memory growth.

## Mitigation if it breaks

- Per-platform `#ifdef` blocks where unavoidable (favor avoiding;
  prefer C++17/20 `if constexpr` patterns when introduced).
- Pin compiler versions in CI so issues are reproducible.
- Treat each new platform as a discovery — file findings as new
  known unknowns ([known unknowns/](../known_unknowns/)) entries when
  surfaced.
