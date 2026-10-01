"""Install the tested token mean, retaining an explicit original control.

MOSAIC_OPENDDE_AGGREGATION=original retains the previous scatter-add path.
The default is stable: stable integer sort, fixed-order segmented scan with
float32 accumulation for half precision, and one write per output segment.
Choose the mode before compilation; use separate processes for controls.
This changes reduction order and is not bitwise native parity.
"""

import importlib.util
from pathlib import Path


MARKER = "# MOSAIC STABLE TOKEN MEAN v1"
ORIGINAL = """    atom_to_token_idx = atom_to_token_idx.astype(jnp.int32)
    d = x_atom.shape[-1]
    out_shape = x_atom.shape[:-2] + (n_token, d)
"""
REPLACEMENT = """    atom_to_token_idx = atom_to_token_idx.astype(jnp.int32)
    # MOSAIC STABLE TOKEN MEAN v1
    from mosaic.opendde_numerics import aggregation_mode, stable_segment_mean

    if aggregation_mode() == "stable":
        return stable_segment_mean(x_atom, atom_to_token_idx, n_token)
    d = x_atom.shape[-1]
    out_shape = x_atom.shape[:-2] + (n_token, d)
"""


def patch_file(target):
    target = Path(target)
    source = target.read_text()
    if MARKER in source:
        if source.count(REPLACEMENT) != 1:
            raise ValueError("aggregation patch marker exists but its body differs")
        changed = False
    else:
        if source.count(ORIGINAL) != 1:
            raise ValueError("expected exactly one supported token aggregation body")
        patched = source.replace(ORIGINAL, REPLACEMENT)
        compile(patched, str(target), "exec")
        target.write_text(patched)
        changed = True
    Path(importlib.util.cache_from_source(str(target))).unlink(missing_ok=True)
    return changed


def main():
    spec = importlib.util.find_spec("jopendde")
    if spec is None or spec.origin is None:
        raise RuntimeError("jopendde is not installed")
    target = Path(spec.origin).parent / "transformer.py"
    changed = patch_file(target)
    print(f"{target}: {'patched' if changed else 'already patched'} token aggregation")


if __name__ == "__main__":
    main()
