"""Verify Python ↔ Lean safetensors interop.

For each vector in `tests/safetensors/vectors/`:

  1. Read original by parsing bytes directly (covers BF16, BOOL —
     dtypes that the numpy framework can't materialize).
  2. Run `lake exe safetensors_roundtrip` to make Lean re-emit it.
  3. Read the round-tripped file the same way.
  4. Assert: same tensor key set, same dtype, same shape, byte-equal
     payloads, same metadata key/value pairs.
  5. As a separate smoke check, ask the reference `safetensors`
     library to open the round-tripped file (skipping BF16-only
     vectors that numpy can't materialize). Catches header
     well-formedness regressions that the byte path might miss.

Run from anywhere; no args:

    python3 tests/safetensors/run_tests.py
"""

import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

from safetensors import safe_open


REPO = Path(__file__).resolve().parents[2]
VECTORS = REPO / "tests" / "safetensors" / "vectors"
RT_DIR = REPO / "tests" / "safetensors" / "rt"
LAKE = "lake"


def _read_raw(path: Path):
    """Parse a .safetensors file via byte ops, returning
    (tensors: dict[name -> (dtype_str, shape_tuple, raw_bytes)], metadata: dict)."""
    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValueError(f"{path.name}: too short")
    hlen = struct.unpack_from("<Q", raw, 0)[0]
    if 8 + hlen > len(raw):
        raise ValueError(f"{path.name}: header length out of bounds")
    header = json.loads(raw[8 : 8 + hlen])
    body = raw[8 + hlen :]
    metadata = header.pop("__metadata__", {}) or {}
    tensors = {}
    for name, info in header.items():
        lo, hi = info["data_offsets"]
        tensors[name] = (info["dtype"], tuple(info["shape"]), bytes(body[lo:hi]))
    return tensors, metadata


def _diff(a, b):
    if set(a) != set(b):
        return f"key set differs: {sorted(set(a) ^ set(b))}"
    for k in a:
        ad, ash, abys = a[k]
        bd, bsh, bbys = b[k]
        if ad != bd:
            return f"{k}: dtype {ad} vs {bd}"
        if ash != bsh:
            return f"{k}: shape {ash} vs {bsh}"
        if abys != bbys:
            return f"{k}: bytes differ ({len(abys)} vs {len(bbys)})"
    return None


def _has_bf16(tensors):
    return any(d == "BF16" for d, _, _ in tensors.values())


def _safetensors_lib_can_open(path: Path) -> str | None:
    """Open with the reference library and return None on success, or
    a description of the failure. Skips numpy-incompatible dtypes."""
    try:
        with safe_open(str(path), framework="numpy") as s:
            for k in s.keys():
                _ = s.get_tensor(k)
        return None
    except Exception as e:  # noqa: BLE001
        return repr(e)


def main() -> int:
    if not VECTORS.exists():
        print(f"no vectors at {VECTORS} — run gen_vectors.py first")
        return 1
    if RT_DIR.exists():
        shutil.rmtree(RT_DIR)
    RT_DIR.mkdir(parents=True)

    vectors = sorted(VECTORS.glob("*.safetensors"))
    if not vectors:
        print("no vectors found")
        return 1

    fails = []
    for vf in vectors:
        rt = RT_DIR / vf.name
        cmd = [LAKE, "exe", "safetensors_roundtrip", str(vf), str(rt)]
        proc = subprocess.run(
            cmd, cwd=REPO / "lean", capture_output=True, text=True
        )
        if proc.returncode != 0:
            fails.append((vf.name, f"lake exe failed: {proc.stderr.strip()}"))
            continue

        a_t, a_m = _read_raw(vf)
        b_t, b_m = _read_raw(rt)
        d = _diff(a_t, b_t)
        if d is not None:
            fails.append((vf.name, f"tensor diff: {d}"))
            continue
        if a_m != b_m:
            fails.append((vf.name, f"metadata diff: {a_m} vs {b_m}"))
            continue

        # Separate smoke: ref lib accepts the round-tripped file.
        if _has_bf16(a_t):
            lib_msg = "skip-bf16"
        else:
            err = _safetensors_lib_can_open(rt)
            if err is not None:
                fails.append((vf.name, f"ref lib rejected rt file: {err}"))
                continue
            lib_msg = "lib-ok"

        print(
            f"PASS  {vf.name:30s}  "
            f"{len(a_t):2d} tensors  {len(a_m):2d} meta  {lib_msg}"
        )

    if fails:
        print()
        for name, msg in fails:
            print(f"FAIL  {name:30s}  {msg}")
        return 1
    print(f"\nall {len(vectors)} vectors round-trip cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
