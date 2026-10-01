"""Independent, model-free backbone and raw-to-atom37 reporting checks.

Distance limits are deliberately broad sanity bounds, not structure validation.
Only consecutive residues within a chain are connected. Missing atoms, ambiguous
names and nonfinite coordinates fail explicitly rather than being omitted.
"""

import numpy as np

BACKBONE_NAMES = ("N", "CA", "C", "O")
BACKBONE_SLOTS = (0, 1, 2, 4)
DISTANCE_LIMITS = {
    "N_CA": (1.0, 2.0),
    "CA_C": (1.0, 2.0),
    "C_O": (0.9, 1.7),
    "peptide_C_N": (1.0, 1.8),
    "adjacent_CA": (2.5, 4.5),
}


def decode_atom_names(encoded):
    encoded = np.asarray(encoded)
    if encoded.ndim != 3 or encoded.shape[1:] != (4, 64):
        raise ValueError("expected atom-name encoding [atoms, 4, 64]")
    return np.array(
        ["".join(chr(int(c) + 32) for c in row).strip() for row in encoded.argmax(-1)]
    )


def backbone_geometry(backbone, mask, asym_id, residue_index):
    """Audit N/CA/C/O coordinates; retain every per-bond failure and distance."""
    backbone = np.asarray(backbone, dtype=float)
    mask = np.asarray(mask) > 0.5
    asym, index = np.asarray(asym_id), np.asarray(residue_index)
    n = len(asym)
    if backbone.shape != (n, 4, 3) or mask.shape != (n, 4) or index.shape != (n,):
        raise ValueError("inconsistent backbone/metadata shapes")
    finite = np.isfinite(backbone).all(-1)
    available = mask & finite
    chains = []
    for chain in dict.fromkeys(asym.tolist()):
        ids = np.flatnonzero(asym == chain)
        pairs = [
            (int(a), int(b))
            for a, b in zip(ids[:-1], ids[1:])
            if index[b] == index[a] + 1
        ]
        bonds = {
            "N_CA": [(i, 0, i, 1) for i in ids],
            "CA_C": [(i, 1, i, 2) for i in ids],
            "C_O": [(i, 2, i, 3) for i in ids],
            "peptide_C_N": [(a, 2, b, 0) for a, b in pairs],
            "adjacent_CA": [(a, 1, b, 1) for a, b in pairs],
        }
        metrics = {}
        for name, entries in bonds.items():
            low, high = DISTANCE_LIMITS[name]
            distances, failures = [], []
            for a, sa, b, sb in entries:
                d = (
                    float(np.linalg.norm(backbone[a, sa] - backbone[b, sb]))
                    if available[a, sa] and available[b, sb]
                    else None
                )
                distances.append(d)
                if d is None or not low <= d <= high:
                    failures.append([int(a), int(b)])
            valid = [d for d in distances if d is not None]
            metrics[name] = dict(
                count=len(entries),
                median_A=float(np.median(valid)) if valid else None,
                distances_A=distances,
                violations=len(failures),
                token_pairs=failures,
            )
        ordered_unique = bool(np.all(np.diff(index[ids]) > 0))
        chains.append(
            dict(
                asym_id=int(chain),
                residue_count=len(ids),
                ordered_unique_residues=ordered_unique,
                skipped_residue_gaps=len(ids) - 1 - len(pairs),
                passed=bool(available[ids].all())
                and ordered_unique
                and all(m["violations"] == 0 for m in metrics.values()),
                distances=metrics,
            )
        )
    return dict(
        passed=bool(n) and all(c["passed"] for c in chains),
        missing_atom_count=int((~mask).sum()),
        nonfinite_atom_count=int((mask & ~finite).sum()),
        thresholds_A=DISTANCE_LIMITS,
        chains=chains,
        interpretation="Broad backbone sanity check only; passing does not establish a correct fold or pose.",
    )


def audit_raw_mapping(
    coords,
    atom_to_token,
    atom_names,
    atom_mask,
    asym_id,
    residue_index,
    mapped_coords=None,
    mapped_mask=None,
):
    """Extract backbone by explicit atom names, independently of dense atom slots."""
    coords = np.asarray(coords, dtype=float)
    tokens = np.asarray(atom_to_token)
    names = np.asarray(atom_names)
    active = np.asarray(atom_mask) > 0.5
    n = len(asym_id)
    if (
        coords.shape != (len(tokens), 3)
        or names.shape != tokens.shape
        or active.shape != tokens.shape
    ):
        raise ValueError("inconsistent raw atom metadata")
    bb = np.zeros((n, 4, 3), dtype=float)
    mask = np.zeros((n, 4), dtype=bool)
    issues, seen = [], set()
    if not np.isfinite(coords[active]).all():
        issues.append("nonfinite active raw coordinates")
    slots = {name: i for i, name in enumerate(BACKBONE_NAMES)}
    for i in np.flatnonzero(active):
        token = int(tokens[i])
        if token != tokens[i] or not 0 <= token < n:
            issues.append(f"invalid token index at atom {i}")
            continue
        identity = (token, str(names[i]))
        if identity in seen:
            issues.append(f"duplicate atom {identity}")
        seen.add(identity)
        if names[i] in slots:
            slot = slots[names[i]]
            bb[token, slot], mask[token, slot] = coords[i], True
    raw = backbone_geometry(bb, mask, asym_id, residue_index)
    result = dict(
        raw_geometry=raw,
        metadata_issues=issues,
        raw_passed=raw["passed"] and not issues,
    )
    if (mapped_coords is None) != (mapped_mask is None):
        raise ValueError("both mapped coordinates and mask are required")
    if mapped_coords is not None:
        mapped = np.asarray(mapped_coords)[:, BACKBONE_SLOTS]
        mmask = np.asarray(mapped_mask)[:, BACKBONE_SLOTS] > 0.5
        shared = mask & mmask
        delta = np.linalg.norm(bb[shared] - mapped[shared], axis=-1)
        error = float(delta.max()) if len(delta) and np.isfinite(delta).all() else None
        agrees = (
            not issues
            and bool(np.array_equal(mask, mmask))
            and error is not None
            and error <= 1e-4
        )
        result.update(
            mapped_geometry=backbone_geometry(mapped, mmask, asym_id, residue_index),
            mapping_agrees=agrees,
            mapping_max_error_A=error,
            mapping_tolerance_A=1e-4,
        )
    result["passed"] = (
        result["raw_passed"]
        and result.get("mapping_agrees", True)
        and result.get("mapped_geometry", raw)["passed"]
    )
    return result
