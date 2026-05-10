"""Verify Python ↔ Lean safetensors interop.

For each vector in `tests/safetensors/vectors/`:

  1. Read original by parsing bytes directly (covers BF16, BOOL —
     dtypes that the numpy framework can't materialize).
  2. Run `lake exe safetensors_roundtrip` to make Lean re-emit it.
  3. Read the round-tripped file the same way.
  4. Assert: same tensor key set, same dtype, same shape, byte-equal
     payloads, same metadata key/value pairs.
  5. As a separate smoke check, ask the reference `safetensors`
     library to open the round-tripped file. `safe_open` + `keys()` +
     `metadata()` validate the header structure and offsets for *every*
     tensor including BF16; `get_tensor()` is skipped only on BF16 keys
     because numpy lacks a bfloat16 dtype (the byte-level round-trip in
     `_diff` already covers BF16 payloads).

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


def _safetensors_lib_can_open(path: Path, raw_dtypes: dict[str, str]) -> str | None:
    """Verify the reference `safetensors` library accepts the file:
      - `safe_open` succeeds (header well-formedness)
      - every tensor key is enumerable
      - metadata is readable
      - every NON-BF16 tensor materializes via numpy (validates byte
        layout / offset alignment / dtype tag end-to-end)
    BF16 tensors are skipped from the numpy materialization step only —
    numpy lacks a bfloat16 dtype, so `get_tensor` raises `TypeError:
    data type 'bfloat16' not understood`. Their structural correctness
    is covered by `safe_open` + `keys()` and the byte-level diff in
    `_diff`. Returns None on success or a description of the failure."""
    try:
        with safe_open(str(path), framework="numpy") as s:
            keys = list(s.keys())
            _md = dict(s.metadata() or {})
            for k in keys:
                if raw_dtypes.get(k) == "BF16":
                    continue
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
        # Pass dtypes from the byte-level parse so we can skip BF16
        # tensors from the numpy materialization step (only that step
        # — `safe_open`, `keys()`, `metadata()` cover BF16 too).
        raw_dtypes = {k: v[0] for k, v in b_t.items()}
        err = _safetensors_lib_can_open(rt, raw_dtypes)
        if err is not None:
            fails.append((vf.name, f"ref lib rejected rt file: {err}"))
            continue
        bf16_keys = sum(1 for d in raw_dtypes.values() if d == "BF16")
        lib_msg = (
            f"lib-ok ({bf16_keys} BF16 structural)" if bf16_keys
            else "lib-ok"
        )

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
