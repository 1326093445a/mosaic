"""Offline audit of saved predictions: is the binder on site, and how is it oriented?

Answers, from structures already on disk and with no GPU:

1. Does the predicted binder contact the intended site at all?
2. Does it do so with the **paratope** or with the framework? Germinal's gate
   turns on exactly this, and a proximity-only reading of it is misleading --
   see the warning below.
3. How does the pose error decompose into **rotation** and **translation**?
   This is what separates a binder displaced from the reference from one
   flipped at the reference site, which no site criterion can distinguish.

Produced docs/P17_JN1.md section 25. Reads `structures/candidate_NNNNN/seed_N.pdb`
and the matching `tables/predictions.csv` written by every run since section 15.

    python examples/p17_site_gate_audit.py results/<run> [more runs ...]

WARNING, and the reason this script exists in its current form: an earlier
version implemented only two of Germinal's three site criteria -- proximity at
5.3 A and >=3 epitope contacts at 6.0 A -- and 23 of 24 predictions passed,
which reads as "a retention gate would be a no-op". Adding
`percent_interface_cdr` changed that to 13 of 24 on one archive. Do not drop
the CDR criteria: proximity alone is satisfied by a binder touching the right
site with the wrong face.
"""

import argparse
import csv
from pathlib import Path
import sys

import gemmi
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Germinal's published thresholds (filter_utils.py::compute_hotspot_proximity)
PROXIMITY_A = 5.3
CONTACT_A = 6.0
MIN_CONTACTS = 3
MIN_INTERFACE_CDR = 0.5
INTERFACE_A = 4.0  # BindCraft's interface enumeration cutoff
FLIP_DEGREES = 120.0  # above this, call it a flip rather than a displacement


def heavy(residue):
    return [a for a in residue if a.element != gemmi.Element("H")]


def ca_coords(chain):
    out = []
    for residue in chain:
        for atom in residue:
            if atom.name == "CA":
                out.append([atom.pos.x, atom.pos.y, atom.pos.z])
                break
    return np.asarray(out)


def kabsch(P, Q):
    """Rigid transform taking P onto Q."""
    cp, cq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - cp).T @ (Q - cq))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cq - cp @ R.T


def atoms_by_residue(chain):
    pos, idx = [], []
    for i, residue in enumerate(chain):
        for atom in heavy(residue):
            pos.append([atom.pos.x, atom.pos.y, atom.pos.z])
            idx.append(i)
    return np.asarray(pos), np.asarray(idx)


def assess(path, epitope, cdr, reference_binder_ca, reference_target_ca):
    """Site, face and orientation for one predicted complex.

    `rotation_deg` and `centroid_shift_A` are the residual rigid motion of the
    binder *after* the predicted target has been fitted to the reference, so
    they decompose the same quantity target-aligned pose RMSD reports.
    """
    structure = gemmi.read_structure(str(path))
    structure.setup_entities()
    model = structure[0]
    if len(model) < 2:
        return None
    binder, target = model[0], model[1]
    if len(binder) != len(reference_binder_ca):
        return None
    if len(target) != len(reference_target_ca):
        return None

    bpos, bres = atoms_by_residue(binder)
    tpos, tres = atoms_by_residue(target)
    if not len(bpos) or not len(tpos):
        return None

    # Per-residue nearest-neighbour distances, both directions.
    d_t, _ = cKDTree(bpos).query(tpos, k=1)
    per_target = np.full(len(target), np.inf)
    np.minimum.at(per_target, tres, d_t)
    d_b, _ = cKDTree(tpos).query(bpos, k=1)
    per_binder = np.full(len(binder), np.inf)
    np.minimum.at(per_binder, bres, d_b)

    # CDR atoms only, which is what distinguishes paratope from framework.
    cdr_atoms = np.isin(bres, cdr)
    if cdr_atoms.any():
        d_c, _ = cKDTree(bpos[cdr_atoms]).query(tpos, k=1)
        per_target_cdr = np.full(len(target), np.inf)
        np.minimum.at(per_target_cdr, tres, d_c)
    else:
        per_target_cdr = np.full(len(target), np.inf)

    interface_binder = np.flatnonzero(per_binder <= INTERFACE_A)
    interface_target = set(np.flatnonzero(per_target <= INTERFACE_A).tolist())
    pct_cdr = (
        float(np.isin(interface_binder, cdr).sum() / len(interface_binder))
        if len(interface_binder)
        else 0.0
    )

    R, t = kabsch(ca_coords(target), reference_target_ca)
    aligned = ca_coords(binder) @ R.T + t
    R2, _ = kabsch(aligned, reference_binder_ca)
    rotation = float(
        np.degrees(np.arccos(np.clip((np.trace(R2) - 1.0) / 2.0, -1.0, 1.0)))
    )

    epi = set(int(i) for i in epitope)
    overlap = interface_target & epi
    near = bool(per_target[epitope].min() <= PROXIMITY_A)
    contacts = int((per_target[epitope] <= CONTACT_A).sum())
    cdr_contacts = int((per_target_cdr[epitope] <= CONTACT_A).sum())
    return dict(
        closest_site_A=float(per_target[epitope].min()),
        site_contacts=contacts,
        cdr_site_contacts=cdr_contacts,
        interface_target_residues=len(interface_target),
        percent_interface_cdr=pct_cdr,
        site_recall=len(overlap) / len(epi),
        site_precision=(len(overlap) / len(interface_target)) if interface_target else 0.0,
        rotation_deg=rotation,
        centroid_shift_A=float(np.linalg.norm(aligned.mean(0) - reference_binder_ca.mean(0))),
        orientation="flipped" if rotation > FLIP_DEGREES else "displaced",
        # All four criteria. Dropping the last two makes this near-vacuous.
        gate_pass=bool(
            near
            and contacts >= MIN_CONTACTS
            and cdr_contacts > 0
            and pct_cdr > MIN_INTERFACE_CDR
        ),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--reference", type=Path, default=None,
                        help="default: P17_JN1.pdb for a 184-residue target, "
                             "P17_Alpha.pdb for 195")
    parser.add_argument("--out", type=Path, default=Path("site_gate_audit.csv"))
    args = parser.parse_args(argv)

    from p17_alpha_reference import (ALPHA_BINDER_CHAIN, ALPHA_COMPLEX,
                                     ALPHA_TARGET_CHAIN, contact_epitope,
                                     load_complex)
    from p17_hallucination_search import CDR_RESIDUE_INDICES_1IDX

    cdr = np.array(sorted(i - 1 for i in CDR_RESIDUE_INDICES_1IDX))
    repo = Path(__file__).resolve().parent.parent
    alpha = load_complex(ALPHA_COMPLEX, ALPHA_BINDER_CHAIN, ALPHA_TARGET_CHAIN)
    jn1 = load_complex(repo / "P17_JN1.pdb", "B", "T")
    systems = {
        195: ("Alpha", contact_epitope(alpha["binder_ca"], alpha["target_ca"], 8.0),
              alpha["binder_ca"], alpha["target_ca"]),
        # JN.1's five literature hotspots: a coarse denominator for recall.
        184: ("JN.1", np.array([114, 116, 145, 147, 149]),
              jn1["binder_ca"], jn1["target_ca"]),
    }

    rows = []
    for run in args.runs:
        for table in sorted(run.rglob("tables/predictions.csv")):
            base = table.parent.parent
            for pred in csv.DictReader(table.open()):
                pdb = (base / "structures"
                       / f"candidate_{int(pred['candidate_id']):05d}"
                       / f"seed_{pred['selection_seed']}.pdb")
                if not pdb.is_file():
                    continue
                structure = gemmi.read_structure(str(pdb))
                structure.setup_entities()
                n_target = len(structure[0][1])
                if n_target not in systems:
                    continue
                name, epitope, rb, rt = systems[n_target]
                result = assess(pdb, epitope, cdr, rb, rt)
                if result is None:
                    continue
                rows.append(dict(
                    run=str(base), system=name,
                    candidate=pred["candidate_id"], seed=pred["selection_seed"],
                    is_wt=pred["is_wt"],
                    pose_rmsd_A=float(pred["binder_pose_rmsd_A"]),
                    ipsae_min=float(pred["ipsae_min"]), **result))

    if not rows:
        print("No predictions matched; check the run paths.", file=sys.stderr)
        return 2
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for name in sorted({r["system"] for r in rows}):
        sub = [r for r in rows if r["system"] == name]
        nz = [r for r in sub if r["is_wt"] != "True"]
        wt = [r for r in sub if r["is_wt"] == "True"]
        flipped = [r for r in nz if r["orientation"] == "flipped"]
        print(f"\n=== {name}: {len(sub)} predictions ({len(nz)} non-WT) ===")
        print(f"  full gate (all four criteria): {sum(r['gate_pass'] for r in nz)}"
              f"/{len(nz)} non-WT, {sum(r['gate_pass'] for r in wt)}/{len(wt)} WT")
        print(f"  proximity-only (misleading):   "
              f"{sum(r['closest_site_A'] <= PROXIMITY_A and r['site_contacts'] >= MIN_CONTACTS for r in nz)}"
              f"/{len(nz)}")
        print(f"  closest approach to site: median "
              f"{np.median([r['closest_site_A'] for r in nz]):.1f} A")
        print(f"  flipped (> {FLIP_DEGREES:.0f} deg): {len(flipped)}/{len(nz)}"
              + (f", rotation {np.mean([r['rotation_deg'] for r in flipped]):.0f} deg, "
                 f"centroid {np.mean([r['centroid_shift_A'] for r in flipped]):.1f} A, "
                 f"%interface CDR {np.mean([r['percent_interface_cdr'] for r in flipped]):.2f}"
                 if flipped else ""))
        disp = [r for r in nz if r["orientation"] == "displaced"]
        if disp:
            print(f"  displaced: {len(disp)}/{len(nz)}, rotation "
                  f"{np.mean([r['rotation_deg'] for r in disp]):.0f} deg, centroid "
                  f"{np.mean([r['centroid_shift_A'] for r in disp]):.1f} A")
        if len(nz) > 2:
            po = np.array([r["pose_rmsd_A"] for r in nz])
            ip = np.array([r["ipsae_min"] for r in nz])
            pc = np.array([r["percent_interface_cdr"] for r in nz])
            print(f"  corr(pose, site_recall) = "
                  f"{np.corrcoef(po, [r['site_recall'] for r in nz])[0, 1]:+.3f}"
                  f" | corr(%interface CDR, ipSAE) = {np.corrcoef(pc, ip)[0, 1]:+.3f}")
            print(f"  best pose {po.min():.1f} A at ipSAE {ip[po.argmin()]:.3f}"
                  f"  |  best ipSAE {ip.max():.3f} at pose {po[ip.argmax()]:.1f} A"
                  f"   <- what retention picks")
    print(f"\nwrote {args.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
