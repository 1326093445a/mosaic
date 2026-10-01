"""Pose correspondence, geometry and effective proposal-influence checks.

No checkpoints are loaded here. Model callbacks come from the search runner so
settings, losses, random-key schedules and exported predictions stay identical.
"""

import csv
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from mosaic.search import _moves, _proposal_distribution
from p17_search_outputs import pose_metrics


def write_report(root, report):
    Path(root, "diagnostic.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


def require_nondegenerate(coords):
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3 or len(coords) < 3:
        raise ValueError("alignment requires at least three CA coordinates")
    if not np.isfinite(coords).all():
        raise ValueError("nonfinite coordinates")
    singular = np.linalg.svd(coords - coords.mean(0), compute_uv=False)
    if singular[1] <= max(1e-6, singular[0] * 1e-6):
        raise ValueError("degenerate CA alignment: coincident or collinear atoms")


def audit_reference(model, chain_ids, sequences, references):
    import gemmi

    report = dict(units="angstrom", mask="all residues, no omitted CA atoms", chains=[])
    if len(set(chain_ids)) != 2:
        raise ValueError("binder and target must be distinct chains")
    for chain_id, sequence, expected in zip(chain_ids, sequences, references):
        chains = [c for c in model if c.name == chain_id]
        if len(chains) != 1:
            raise ValueError(f"ambiguous chain {chain_id}")
        residues = list(chains[0])
        observed = gemmi.one_letter_code([r.name for r in residues]).upper()
        if observed != sequence or len(expected) != len(sequence):
            raise ValueError(f"sequence/CA correspondence mismatch in chain {chain_id}")
        coords, mapping, seen = [], [], set()
        for index, residue in enumerate(residues):
            residue_id = (residue.seqid.num, residue.seqid.icode.strip())
            atoms = [a for a in residue if a.name == "CA"]
            if residue_id in seen or len(atoms) != 1:
                raise ValueError(
                    f"ambiguous residue or missing/alternate CA: {chain_id}:{residue.seqid}"
                )
            seen.add(residue_id)
            atom = atoms[0]
            coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
            mapping.append(
                dict(
                    index_0idx=index,
                    residue_number=residue_id[0],
                    insertion_code=residue_id[1],
                    residue_name=residue.name,
                    amino_acid=sequence[index],
                )
            )
        coords = np.asarray(coords)
        if not np.allclose(coords, expected, atol=1e-4, rtol=0):
            raise ValueError("reference extraction changed residue order")
        require_nondegenerate(coords)
        distances = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        # Conservative sanity check, not proof of physical validity or units.
        if np.any((distances < 2.5) | (distances > 4.5)):
            raise ValueError(
                f"CA spacing outside 2.5–4.5 A in {chain_id}; audit units/gaps"
            )
        report["chains"].append(
            dict(
                chain=chain_id,
                sequence=sequence,
                mapping=mapping,
                adjacent_CA_distance_A=distances.tolist(),
            )
        )
    return report


def crosscheck_pose(ca, binder, target):
    """Compare independent NumPy reporting and actual JAX loss on the same atoms."""
    import jax.numpy as jnp
    from mosaic.losses.structure_prediction import BinderPoseRMSD

    ca = np.asarray(ca)
    require_nondegenerate(ca[len(binder) :])
    values = pose_metrics(ca, binder, target)
    output = SimpleNamespace(
        backbone_coordinates=jnp.asarray(ca[:, None, :].repeat(4, axis=1))
    )
    _, aux = BinderPoseRMSD(binder, target, rmsd_tolerance=0.0)(
        jnp.zeros((len(binder), 20)),
        output,
        key=None,
    )
    error = max(
        abs(float(aux["binder_pose_rmsd"]) - values["binder_pose_rmsd_A"]),
        abs(float(aux["pose_target_fit_rmsd"]) - values["target_aligned_rmsd_A"]),
    )
    if not np.isfinite(error) or error > 1e-3:
        raise ValueError(f"NumPy/JAX pose mismatch: {error} A")
    return values, error


def geometry_checks(binder, target):
    require_nondegenerate(binder)
    require_nondegenerate(target)
    reference = np.concatenate([binder, target])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    shifted = reference.copy()
    shifted[: len(binder)] += np.array([4.0, 0.0, 0.0])
    cases = dict(
        identity=reference,
        global_rigid=reference @ rotation + 7.0,
        binder_shift=shifted,
    )
    results = {}
    for name, coords in cases.items():
        values, error = crosscheck_pose(coords, binder, target)
        expected = 4.0 if name == "binder_shift" else 0.0
        if abs(values["binder_pose_rmsd_A"] - expected) > 1e-3:
            raise ValueError(f"failed {name} pose control")
        if (
            max(values["target_aligned_rmsd_A"], values["binder_internal_rmsd_A"])
            > 1e-3
        ):
            raise ValueError(f"failed {name} internal-shape control")
        results[name] = dict(values, numpy_jax_max_error_A=error)
    return results


def proposal_comparison(wt, mask, config, gradients):
    distributions, deltas, moves = [], [], None
    for gradient in gradients:
        gradient = np.asarray(gradient)
        if (
            gradient.shape != (len(wt), config.alphabet_size)
            or not np.isfinite(gradient).all()
        ):
            raise ValueError("invalid diagnostic gradient")
        current_moves, current_deltas = _moves(
            wt, wt, mask, config.edit_budget, gradient
        )
        if not current_moves:
            raise ValueError("no feasible diagnostic mutations")
        if moves is not None and moves != current_moves:
            raise ValueError("paired feasible move sets differ")
        moves = current_moves
        probs, _, _ = _proposal_distribution(current_deltas, config.target_entropy)
        distributions.append(probs)
        deltas.append(current_deltas)
    on, repeat, off = distributions
    a, b, c = [np.asarray(g)[mask].ravel() for g in gradients]
    denom = np.linalg.norm(a) * np.linalg.norm(c)
    return (
        dict(
            proposal_tv=float(np.abs(on - off).sum() / 2),
            repeat_proposal_tv=float(np.abs(on - repeat).sum() / 2),
            cdr_gradient_norms=[float(np.linalg.norm(g)) for g in (a, b, c)],
            cdr_gradient_difference_norm=float(np.linalg.norm(a - c)),
            cdr_repeat_difference_norm=float(np.linalg.norm(a - b)),
            on_off_cosine=float(a @ c / denom) if denom > 0 else None,
        ),
        moves,
        deltas,
        distributions,
    )


def run_diagnostic(
    *,
    root,
    wt,
    mask,
    config,
    gradient_on,
    gradient_off,
    predict,
    repeat_predict,
    references,
    seeds,
    pae_cutoff,
    distance_cutoff,
    max_target_rmsd,
    min_proposal_tv,
    repeat_factor,
):
    from p17_confidence_search import confidence_metrics

    started = perf_counter()
    evaluations = [gradient_on(wt), gradient_on(wt), gradient_off(wt)]
    for value, gradient, metrics in evaluations:
        if not np.isfinite(value) or not all(np.isfinite(v) for v in metrics.values()):
            raise ValueError("nonfinite diagnostic loss/metrics")
        for name in ("binder_pose_rmsd", "pose_target_fit_rmsd"):
            if name not in metrics:
                raise ValueError(f"missing proposal metric {name}")
    effect, moves, deltas, probabilities = proposal_comparison(
        wt,
        mask,
        config,
        [e[1] for e in evaluations],
    )
    rows = []
    for i, move in enumerate(moves):
        rows.append(
            dict(
                position_0idx=move[0],
                new_token=move[1],
                reverted_0idx=move[2],
                delta_on=deltas[0][i],
                delta_repeat=deltas[1][i],
                delta_off=deltas[2][i],
                probability_on=probabilities[0][i],
                probability_repeat=probabilities[1][i],
                probability_off=probabilities[2][i],
            )
        )
    with Path(root, "tables", "diagnostic_proposals.csv").open(
        "x", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        Path(root, "confidence", "diagnostic_gradients.npz"),
        on=evaluations[0][1],
        repeat=evaluations[1][1],
        off=evaluations[2][1],
        wt=wt,
        designable_mask=mask,
    )

    # Prespecified rule: WT and the top-probability move under each objective,
    # deduplicated. These are probes, not chosen by their observed outcome.
    probes = [("WT", wt.copy(), None)]
    seen = {tuple(wt)}
    for label, index in (
        ("top_on", int(np.argmax(probabilities[0]))),
        ("top_off", int(np.argmax(probabilities[2]))),
    ):
        pos, token, reverted = moves[index]
        seq = wt.copy()
        seq[pos] = token
        if reverted >= 0:
            seq[reverted] = wt[reverted]
        if tuple(seq) not in seen:
            probes.append((label, seq, index))
            seen.add(tuple(seq))
    observations, baseline = [], {}
    crosscheck_errors, target_fits = [], []

    def observe(label, sequence, index, seed, callback):
        pae, ca, iptm = callback(sequence, seed)
        pose, error = crosscheck_pose(ca, *references)
        crosscheck_errors.append(error)
        target_fits.append(pose["target_aligned_rmsd_A"])
        confidence = confidence_metrics(pae, ca, len(wt), pae_cutoff, distance_cutoff)
        row = dict(
            label=label,
            seed=int(seed),
            sequence=sequence.tolist(),
            iptm=float(iptm),
            **pose,
            **confidence,
        )
        if label == "WT":
            baseline[seed] = row
        else:
            for name in ("binder_pose_rmsd_A", "target_aligned_rmsd_A", "ipsae_min"):
                row["change_" + name] = row[name] - baseline[seed][name]
        if index is not None:
            row.update(
                predicted_composite_delta_on=float(deltas[0][index]),
                predicted_composite_delta_off=float(deltas[2][index]),
            )
        observations.append(row)

    for label, seq, index in probes:
        for seed in seeds:
            observe(label, seq, index, seed, predict)
    for seed in seeds:
        observe("WT_repeat", wt, None, seed, repeat_predict)
    target_fits.extend(e[2]["pose_target_fit_rmsd"] for e in evaluations)
    paired_pose_difference = abs(
        evaluations[0][2]["binder_pose_rmsd"] - evaluations[2][2]["binder_pose_rmsd"]
    )
    repeat_pose_difference = abs(
        evaluations[0][2]["binder_pose_rmsd"] - evaluations[1][2]["binder_pose_rmsd"]
    )
    required_tv = max(min_proposal_tv, repeat_factor * effect["repeat_proposal_tv"])
    checks = dict(
        completed=True,
        same_coordinate_reporting=max(crosscheck_errors) <= 1e-3,
        interpretable_target_fit=max(target_fits) <= max_target_rmsd,
        proposal_influence=effect["proposal_tv"] > required_tv,
        paired_pose_consistent=paired_pose_difference
        <= max(1e-3, repeat_factor * repeat_pose_difference),
    )
    return dict(
        schema_version=1,
        passed=all(checks.values()),
        checks=checks,
        failed_checks=[name for name, value in checks.items() if not value],
        thresholds=dict(
            max_target_rmsd_A=max_target_rmsd,
            min_proposal_tv=min_proposal_tv,
            repeat_factor=repeat_factor,
            required_proposal_tv=required_tv,
            status="provisional; preregistered in config.json",
        ),
        influence=effect,
        max_target_fit_rmsd_A=max(target_fits),
        gradient_evaluations=[dict(loss=e[0], metrics=e[2]) for e in evaluations],
        paired_pose_difference_A=paired_pose_difference,
        repeat_pose_difference_A=repeat_pose_difference,
        forward_observations=observations,
        full_gradient_calls=3,
        full_prediction_calls=len(observations),
        elapsed_seconds=perf_counter() - started,
        interpretation="Proposal influence is not proof of improved pose or affinity. "
        "Gradient differences are composite clipped-gradient effects. "
        "Saved structures are separate forward predictions.",
    )
