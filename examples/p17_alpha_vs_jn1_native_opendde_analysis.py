"""Real RMSD + ipSAE(pae_cutoff=12, dist_cutoff=12) analysis of the native
(torch) OpenDDE P17-vs-Alpha (binding) vs. P17-vs-JN.1 (non-binding)
predictions produced from examples/opendde_inputs/p17_alpha.json and
examples/opendde_inputs/p17_jn1.json (run with --need_atom_confidence true
so the per-run `*_full_data_sample_0.json` has the full token_pair_pae
matrix needed for ipSAE; the default `*_summary_confidence_*.json` only
has scalar iptm/ptm/gpde aggregates).

Those two JSONs are the canonical inputs behind every number here -- the
binder chain is byte-identical between them, so the ONLY thing that differs
is the target (Alpha, 195 aa vs. JN.1, 184 aa). `opendde pred -i` accepts
any path, so point it straight at these rather than keeping a second copy
inside the OpenDDE clone (which is a separate git repo, untracked here).

Two things this answers, directly requested by the user after seeing the
scalar ipTM/gPDE comparison:
  1. RMSD: is the predicted BINDER POSE itself wrong for the non-binding
     complex (JN.1), not just "less confident"? Computed the same way as
     mosaic's BinderPoseRMSD: Kabsch-align the predicted TARGET chain CAs
     onto the real deposited target CAs, apply that same rigid transform
     to the predicted binder CAs, then measure binder RMSD against the
     real deposited binder CAs. This isolates binder pose error from
     target pose error -- a large value means the model placed the binder
     in genuinely the wrong location/orientation relative to the target,
     not just an internally-consistent-but-unconfident placement.
  2. ipSAE(12, 12): the real, standard interface metric (Dunbrack et al.,
     https://github.com/DunbrackLab/IPSAE, `python ipsae.py <pae> <cif> 12
     12`). The headline "ipSAE" column that script reports is the
     "d0res, asym-max" variant: for each interface residue i in chain1,
     average ptm_func(pae[i, valid_js], d0) over js in chain2 with
     pae[i,j] < pae_cutoff (d0 computed per-residue from the count of such
     valid js, Yang & Skolnick 2004 / Dunbrack eqn 15), then take the max
     over i -- this exactly matches mosaic's own
     src/mosaic/losses/structure_prediction.py BinderTargetIPSAE /
     TargetBinderIPSAE (confirmed by reading the official script's source
     directly), so this is a faithful reimplementation for our case, not
     an approximation. dist_cutoff=12 does NOT feed the ipSAE score itself
     in the official script (verified: dist_cutoff only gates a separate
     diagnostic "how many interface residues are within 12A" residue
     count) -- reported here too, alongside the score, for completeness.

Generate the predictions this reads (from the OpenDDE clone's venv, which
is a separate install from mosaic's -- see patches/ and the OpenDDE repo's
own pyproject; OPENDDE_ROOT_DIR reuses the checkpoints mosaic already
cached, so nothing is re-downloaded):

    cd OpenDDE && source .venv/bin/activate
    export OPENDDE_ROOT_DIR=/home/yfeng17/.cache/mosaic/opendde
    export LAYERNORM_TYPE=torch
    for name in alpha jn1; do
      opendde pred \\
        -i ../examples/opendde_inputs/p17_${name}.json \\
        -o ./test_outputs/p17_${name}_full \\
        -s 0,1,2 -c 3 -p 64 -e 1 -d bf16 -n opendde_v1 \\
        --load_checkpoint_path "$OPENDDE_ROOT_DIR/checkpoint/opendde_abag.pt" \\
        --use_msa false --use_template false \\
        --trimul_kernel torch --triatt_kernel torch \\
        --need_atom_confidence true
    done

`-d bf16` is not optional in practice: it selects torch.autocast, and the
same run in fp32 OOMs on a 24GB card.

Then:
    .venv/bin/python examples/p17_alpha_vs_jn1_native_opendde_analysis.py \\
        --output results/p17_alpha_vs_jn1_native_opendde/rmsd_ipsae.csv
"""
import argparse
import csv
import glob
import json
from pathlib import Path

import gemmi
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
OPENDDE_REPO = REPO_ROOT / "OpenDDE"

COMPLEXES = [
    {
        "name": "P17_vs_Alpha",
        "label": "binding",
        "out_dir": OPENDDE_REPO / "test_outputs" / "p17_alpha_full",
        "reference_pdb": REPO_ROOT / "P17_Alpha.pdb",
        "binder_chain": "B",
        "target_chain": "A",
    },
    {
        "name": "P17_vs_JN1",
        "label": "non-binding",
        "out_dir": OPENDDE_REPO / "test_outputs" / "p17_jn1_full",
        "reference_pdb": REPO_ROOT / "P17_JN1.pdb",
        "binder_chain": "B",
        "target_chain": "T",
    },
]

PAE_CUTOFF = 12.0
DIST_CUTOFF = 12.0


def amino_acid_residues(chain):
    return [r for r in chain
            if (info := gemmi.find_tabulated_residue(r.name)) and info.is_amino_acid()]


def ca_coords(chain):
    coords = []
    for res in amino_acid_residues(chain):
        for a in res:
            if a.name == "CA":
                coords.append([a.pos.x, a.pos.y, a.pos.z])
                break
    return np.array(coords, dtype=np.float64)


def kabsch(mobile: np.ndarray, target: np.ndarray):
    """Returns (R, t) such that mobile @ R + t best fits target (least squares)."""
    mobile_c = mobile - mobile.mean(axis=0)
    target_c = target - target.mean(axis=0)
    H = mobile_c.T @ target_c
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    t = target.mean(axis=0) - mobile.mean(axis=0) @ R.T
    return R, t


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=-1))))


def calc_d0(n_residues: np.ndarray) -> np.ndarray:
    n = np.maximum(26.0, n_residues.astype(np.float64))
    return np.maximum(1.0, 1.24 * (n - 15.0) ** (1.0 / 3.0) - 1.8)


def ptm_func(x: np.ndarray, d0) -> np.ndarray:
    return 1.0 / (1.0 + (x / d0) ** 2.0)


def ipsae_d0res_asym_max(pae_matrix: np.ndarray, distances: np.ndarray,
                          asym_id: np.ndarray, chain1_val: int, chain2_val: int,
                          pae_cutoff: float, dist_cutoff: float):
    """Faithful reimplementation of the official ipsae.py's headline
    'ipSAE' column (the d0res, asym-max variant) for the chain1->chain2
    direction, plus the dist_cutoff-based interface-residue-count
    diagnostic (which, per the official script, does not feed the score
    itself)."""
    is1 = asym_id == chain1_val
    is2 = asym_id == chain2_val
    valid_pairs_matrix = np.outer(is1, is2) & (pae_matrix < pae_cutoff)
    n0res_byres = valid_pairs_matrix.sum(axis=1)
    d0res_byres = calc_d0(n0res_byres)

    byres = np.zeros(pae_matrix.shape[0])
    idx1 = np.where(is1)[0]
    for i in idx1:
        valid = valid_pairs_matrix[i]
        if valid.any():
            byres[i] = ptm_func(pae_matrix[i, valid], d0res_byres[i]).mean()
    max_idx = idx1[np.argmax(byres[idx1])]
    ipsae = float(byres[max_idx])

    dist_valid = np.outer(is1, is2) & (pae_matrix < pae_cutoff) & (distances < dist_cutoff)
    n_interface_res1 = int(np.any(dist_valid, axis=1)[is1].sum())
    n_interface_res2 = int(np.any(dist_valid, axis=0)[is2].sum())
    return ipsae, n_interface_res1, n_interface_res2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for cx in COMPLEXES:
        print(f"\n=== {cx['name']} ({cx['label']}) ===", flush=True)
        ref_st = gemmi.read_structure(str(cx["reference_pdb"]))
        ref_st.setup_entities()
        ref_model = ref_st[0]
        ref_binder_ca = ca_coords(ref_model[cx["binder_chain"]])
        ref_target_ca = ca_coords(ref_model[cx["target_chain"]])
        n_binder = len(ref_binder_ca)
        n_target = len(ref_target_ca)
        print(f"  reference: binder {n_binder} CA, target {n_target} CA", flush=True)

        full_data_files = sorted(
            glob.glob(str(cx["out_dir"] / cx["name"] / "seed_*" / "predictions" /
                           f"{cx['name']}_full_data_sample_0.json"))
        )
        for fd_path in full_data_files:
            seed = Path(fd_path).parent.parent.name.replace("seed_", "")
            cif_path = Path(fd_path).parent / f"{cx['name']}_sample_0.cif"

            fd = json.load(open(fd_path))
            pae_matrix = np.array(fd["token_pair_pae"], dtype=np.float64)
            asym_id = np.array(fd["token_asym_id"], dtype=np.int64)
            chain1_val, chain2_val = asym_id[0], asym_id[-1]  # binder first, target second

            pred_st = gemmi.read_structure(str(cif_path))
            pred_st.setup_entities()
            pred_model = pred_st[0]
            pred_binder_ca = ca_coords(pred_model[cx["binder_chain"]])
            pred_target_ca = ca_coords(pred_model[cx["target_chain"]])
            all_ca = np.concatenate([pred_binder_ca, pred_target_ca], axis=0)
            distances = np.sqrt(
                ((all_ca[:, None, :] - all_ca[None, :, :]) ** 2).sum(axis=-1)
            )

            assert pred_binder_ca.shape[0] == n_binder, "binder length mismatch"
            assert pred_target_ca.shape[0] == n_target, "target length mismatch"
            assert pae_matrix.shape[0] == n_binder + n_target, "token count mismatch"

            # target-aligned binder pose RMSD (mosaic's BinderPoseRMSD convention)
            R, t = kabsch(pred_target_ca, ref_target_ca)
            aligned_target_ca = pred_target_ca @ R.T + t
            aligned_binder_ca = pred_binder_ca @ R.T + t
            target_rmsd_after_align = rmsd(aligned_target_ca, ref_target_ca)
            binder_pose_rmsd = rmsd(aligned_binder_ca, ref_binder_ca)

            # unaligned (raw superposition-free) binder-vs-reference RMSD, for context:
            # how far apart are the two binder CA clouds without any fitting at all
            raw_binder_rmsd = rmsd(pred_binder_ca, ref_binder_ca)

            bt_ipsae, bt_n_int_binder, bt_n_int_target = ipsae_d0res_asym_max(
                pae_matrix, distances, asym_id, chain1_val, chain2_val,
                PAE_CUTOFF, DIST_CUTOFF,
            )
            tb_ipsae, tb_n_int_target, tb_n_int_binder = ipsae_d0res_asym_max(
                pae_matrix, distances, asym_id, chain2_val, chain1_val,
                PAE_CUTOFF, DIST_CUTOFF,
            )
            ipsae_min = min(bt_ipsae, tb_ipsae)

            row = {
                "complex": cx["name"], "label": cx["label"], "seed": seed,
                "binder_pose_rmsd": binder_pose_rmsd,
                "target_rmsd_after_align": target_rmsd_after_align,
                "raw_binder_rmsd_no_alignment": raw_binder_rmsd,
                f"bt_ipsae_pae{int(PAE_CUTOFF)}": bt_ipsae,
                f"tb_ipsae_pae{int(PAE_CUTOFF)}": tb_ipsae,
                f"ipsae_min_pae{int(PAE_CUTOFF)}": ipsae_min,
                f"n_interface_res_binder_dist{int(DIST_CUTOFF)}": bt_n_int_binder,
                f"n_interface_res_target_dist{int(DIST_CUTOFF)}": bt_n_int_target,
            }
            print(f"  seed={seed}  binder_pose_rmsd={binder_pose_rmsd:6.2f}A "
                  f"(target realigns to {target_rmsd_after_align:.2f}A, confirming the "
                  f"target itself isn't what's off)  "
                  f"bt_ipsae={bt_ipsae:.4f}  tb_ipsae={tb_ipsae:.4f}  "
                  f"ipsae_min={ipsae_min:.4f}  "
                  f"n_interface_res(binder/target, dist<{int(DIST_CUTOFF)}A)="
                  f"{bt_n_int_binder}/{bt_n_int_target}", flush=True)
            rows.append(row)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.output}", flush=True)

    print("\n=== per-complex mean across seeds ===", flush=True)
    for cx in COMPLEXES:
        sub = [r for r in rows if r["complex"] == cx["name"]]
        if not sub:
            continue
        keys = [k for k in sub[0] if k not in ("complex", "label", "seed")]
        means = {k: np.mean([r[k] for r in sub]) for k in keys}
        print(f"  {cx['name']} ({cx['label']}): " +
              "  ".join(f"{k}={v:.3f}" for k, v in means.items()), flush=True)


if __name__ == "__main__":
    main()
