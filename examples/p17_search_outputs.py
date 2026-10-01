"""Persist the exact forward predictions used by the P17 search runner."""

import csv
from pathlib import Path
import shutil

import numpy as np


def compact_prediction(output):
    """Keep structure/confidence arrays without transferring large logits to CPU."""
    import equinox as eqx

    return eqx.tree_at(
        lambda p: (
            p.distogram_logits,
            p.distogram_bins,
            p.pae_logits,
            p.pae_bins,
            p.structure_coordinates,
        ),
        output,
        replace=(None,) * 5,
    )


def pose_metrics(ca, reference_binder_ca, reference_target_ca):
    """CA RMSDs in angstroms; align target first to measure binder pose drift."""
    ca = np.asarray(ca, dtype=np.float64)
    binder = np.asarray(reference_binder_ca, dtype=np.float64)
    target = np.asarray(reference_target_ca, dtype=np.float64)
    if (
        binder.ndim != 2
        or target.ndim != 2
        or binder.shape[1:] != (3,)
        or target.shape[1:] != (3,)
        or min(len(binder), len(target)) < 1
        or ca.shape != (len(binder) + len(target), 3)
        or not all(np.all(np.isfinite(x)) for x in (ca, binder, target))
    ):
        raise ValueError("invalid reference/predicted CA coordinates")

    def fit(mobile, reference):
        cm, cr = mobile.mean(0), reference.mean(0)
        u, _, vt = np.linalg.svd((mobile - cm).T @ (reference - cr))
        correction = np.eye(3)
        correction[-1, -1] = np.linalg.det(u @ vt)
        rotation = u @ correction @ vt
        return rotation, cr - cm @ rotation

    def rmsd(a, b):
        return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=-1))))

    split = len(binder)
    rotation, translation = fit(ca[split:], target)
    aligned = ca @ rotation + translation
    binder_rotation, binder_translation = fit(ca[:split], binder)
    return dict(
        binder_pose_rmsd_A=rmsd(aligned[:split], binder),
        target_aligned_rmsd_A=rmsd(aligned[split:], target),
        binder_internal_rmsd_A=rmsd(
            ca[:split] @ binder_rotation + binder_translation, binder
        ),
    )


class SearchOutputs:
    """One fresh run directory; candidate IDs follow first evaluation order."""

    def __init__(self, root, reference_binder_ca=None, reference_target_ca=None):
        self.root = Path(root)
        self.ids = {}
        if (reference_binder_ca is None) != (reference_target_ca is None):
            raise ValueError("both reference chains are required")
        self.reference_binder_ca = reference_binder_ca
        self.reference_target_ca = reference_target_ca
        for name in ("tables", "logs", "structures", "confidence", "best"):
            (self.root / name).mkdir()
        (self.root / "README.md").write_text(
            "# P17 search results\n\n"
            "- `config.json`: settings, model versions and input sequences.\n"
            "- `summary.json`: completion status, best sequence and compute counts "
            "(written after successful completion).\n"
            "- `tables/candidates.csv`: all scored candidates, final membership, "
            "ranking and structure folders (written at completion).\n"
            "- `tables/predictions.csv`: one row per saved candidate/structural seed, "
            "including sequence, confidence metrics and relative file paths.\n"
            "- `structures/candidate_00000/seed_0.pdb`: scored complex structures; "
            "candidate 0 is WT. Matching `.cif` files are also saved. Includes rejected candidates.\n"
            "- `confidence/candidate_00000/seed_0.npz`: matching PAE (angstroms), "
            "pLDDT (0–1), CA coordinates, atom37 coordinates/mask, chain IDs, "
            "residue numbers and sequence. Load with `numpy.load`.\n"
            "Pose columns in predictions.csv are in angstroms: binder_pose_rmsd_A "
            "aligns the target first; binder_internal_rmsd_A independently aligns "
            "the binder; target_aligned_rmsd_A measures target fit. They are "
            "diagnostics; see config.json for optional pose-retention settings.\n\n"
            "- `best/`: copies of the winning candidate's PDBs for every selection "
            "seed, available after successful completion.\n"
            "- `logs/events.jsonl`, `logs/memory.jsonl`: search and memory records.\n\n"
            "PDBs contain predicted protein heavy atoms, with pLDDT × 100 in the "
            "B-factor column. Chain IDs and residue numbers come from the model; "
            "the prediction CSV identifies binder and target chains.\n"
            "Structures are the exact scoring predictions, not extra refolds or "
            "experimental structures. No gradient-path structures are exported.\n"
            "The score is the mean directional-min ipSAE across selection seeds; "
            "individual seed scores are in predictions.csv.\n"
            "Prediction records are saved incrementally, so completed predictions "
            "remain available if a later evaluation fails.\n"
        )

    def candidate_id(self, sequence):
        key = tuple(int(i) for i in sequence)
        if key not in self.ids:
            self.ids[key] = len(self.ids)
        return self.ids[key]

    def save_prediction(
        self, candidate_id, seed, binder_sequence, target_sequence, output, metrics
    ):
        """Save synchronized host arrays and index them only after files exist."""
        from mosaic.common import TOKENS
        from mosaic.alphafold.common.protein import PDB_CHAIN_IDS

        sequence = "".join(
            TOKENS[i] for i in np.asarray(output.full_sequence).argmax(-1)
        )
        if sequence != binder_sequence + target_sequence:
            raise ValueError(
                "predicted structure sequence does not match scored complex"
            )
        coords, mask = np.asarray(output.atom37_coords), np.asarray(output.atom37_mask)
        ca = np.asarray(output.backbone_coordinates)[:, 1]
        arrays = (
            output.pae,
            output.plddt,
            ca,
            mask,
            output.asym_id,
            output.residue_idx,
        )
        if (
            not all(np.all(np.isfinite(a)) for a in arrays)
            or not np.all(np.isfinite(coords[mask >= 0.5]))
            or not all(np.isfinite(v) for v in metrics.values())
        ):
            raise ValueError("cannot export nonfinite prediction")
        if not np.all(mask[:, 1] >= 0.5) or not np.allclose(
            coords[:, 1], ca, atol=1e-4
        ):
            raise ValueError("exported CA atoms do not match scoring coordinates")
        asym = np.asarray(output.asym_id).astype(int)
        split = len(binder_sequence)
        if (
            len(set(asym[:split])) != 1
            or len(set(asym[split:])) != 1
            or asym[0] == asym[split]
        ):
            raise ValueError("expected separate binder and target chains")
        name = f"candidate_{candidate_id:05d}"
        pdb_path = Path("structures") / name / f"seed_{seed}.pdb"
        cif_path = pdb_path.with_suffix(".cif")
        if self.reference_binder_ca is not None:
            metrics = dict(
                metrics,
                **pose_metrics(ca, self.reference_binder_ca, self.reference_target_ca),
            )
        data_path = Path("confidence") / name / f"seed_{seed}.npz"
        for path in (pdb_path, cif_path, data_path):
            (self.root / path).parent.mkdir(exist_ok=True)
            if (self.root / path).exists():
                raise FileExistsError(self.root / path)
        structure = output.to_structure()
        structure.write_pdb(str(self.root / pdb_path))
        structure.make_mmcif_document().write_file(str(self.root / cif_path))
        np.savez_compressed(
            self.root / data_path,
            pae=np.asarray(output.pae, dtype=np.float32),
            plddt=np.asarray(output.plddt, dtype=np.float32),
            ca_coordinates=np.asarray(ca, dtype=np.float32),
            atom37_coords=np.asarray(coords, dtype=np.float32),
            atom37_mask=np.asarray(mask, dtype=np.float32),
            asym_id=asym,
            residue_idx=output.residue_idx,
            sequence=np.asarray(sequence),
        )
        row = dict(
            candidate_id=candidate_id,
            selection_seed=seed,
            is_wt=candidate_id == 0,
            binder_sequence=binder_sequence,
            target_sequence=target_sequence,
            binder_chain=PDB_CHAIN_IDS[asym[0]],
            target_chain=PDB_CHAIN_IDS[asym[split]],
            **metrics,
            structure_file=pdb_path.as_posix(),
            cif_file=cif_path.as_posix(),
            confidence_file=data_path.as_posix(),
        )
        index_path = self.root / "tables/predictions.csv"
        new_file = not index_path.exists()
        with index_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        return row

    def copy_best(self, candidate_id, seeds):
        paths = []
        for seed in seeds:
            for suffix in ("pdb", "cif"):
                source = (
                    self.root
                    / "structures"
                    / f"candidate_{candidate_id:05d}"
                    / f"seed_{seed}.{suffix}"
                )
                destination = self.root / "best" / source.name
                shutil.copy2(source, destination)
                paths.append(destination.relative_to(self.root).as_posix())
        (self.root / "best/README.md").write_text(
            f"# Best candidate: {candidate_id}\n\n"
            "These are copies of the exact scoring predictions, one per selection seed.\n"
            f"Confidence arrays: `../confidence/candidate_{candidate_id:05d}/`.\n"
            "See `../summary.json` for sequence and aggregate score, and "
            "`../tables/predictions.csv` for per-seed metrics.\n"
        )
        return paths
