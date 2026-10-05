"""Reference loading and damage generation for the P17+Alpha recovery control.

Why this exists: the P17 -> JN.1 direction is not known to have a solution, so
a null result there cannot distinguish a bad gradient from an empty feasible
set (docs/P17_JN1.md section 19.3B). P17+Alpha does have a known answer -- it
is the sequence you start from -- so damaging it and measuring recovery makes
both outcomes informative.

`P17_Alpha.pdb` is experimental (PDB 8GZ5, X-ray, 1.70 A), unlike the modeled
`P17_JN1.pdb`, so pose RMSD here is measured against a real arrangement. It
needs handling the JN.1 file does not: waters and EDO interleaved with the
protein, a glycan-only chain, target/binder chains named A/B rather than T/B,
and 11 residues modeled in alternate conformations.

Loads structures and builds sequences. No model, no GPU, no prediction.
"""

import hashlib
from pathlib import Path

import gemmi
import numpy as np

REPO = Path(__file__).resolve().parent.parent

# Verified against the files themselves, not assumed from the PDB entry:
# Alpha chain A is the 195-residue Alpha RBD and chain B the 123-residue P17
# binder, whose sequence is byte-identical to P17_JN1.pdb chain B -- so the
# CDR mask in `p17_hallucination_search` transfers with no renumbering. Chain
# C holds only NAG/FUC and is dropped.
ALPHA_COMPLEX = REPO / "P17_Alpha.pdb"
ALPHA_BINDER_CHAIN = "B"
ALPHA_TARGET_CHAIN = "A"

# Adjacent-CA spacing accepted by the reference audit, matching the bounds the
# WT validator already applies. A unit/gap sanity check, not proof of validity.
CA_SPACING_RANGE_A = (2.5, 4.5)


def _amino_acids(chain):
    """Protein residues only, dropping waters, EDO, NAG/FUC and other hetero.

    Residue *order* is preserved. Filtering by `gemmi`'s own residue table
    rather than a hardcoded name list, so a nonstandard-but-real amino acid is
    not silently discarded as solvent.
    """
    residues = []
    for residue in chain:
        info = gemmi.find_tabulated_residue(residue.name)
        if info is not None and info.is_amino_acid():
            residues.append(residue)
    return residues


def _select_ca(residue, label):
    """The CA atom to use, resolving alternate conformations by occupancy.

    `P17_Alpha.pdb` is a 1.70 A crystal structure and models 11 residues in
    two conformations (altloc A/B). `P17_JN1.pdb`, being a relaxed model, has
    none, so the JN.1 loader never had to decide.

    Occupancy decides, not the altloc letter: at chain A residue 375 the 'A'
    conformer carries occupancy 0.34 against 'B''s 0.66, so taking altloc 'A'
    by convention would pick the minor conformer. Ties break on the altloc
    code so the choice stays deterministic.
    """
    atoms = [a for a in residue if a.name == "CA"]
    if not atoms:
        raise ValueError(
            f"{label} residue {residue.seqid.num}{residue.seqid.icode.strip()} "
            "has no CA atom"
        )
    if len(atoms) == 1:
        return atoms[0], None
    if not all(a.altloc for a in atoms):
        raise ValueError(
            f"{label} residue {residue.seqid.num} has {len(atoms)} CA atoms "
            "that are not all alternate conformations"
        )
    chosen = max(atoms, key=lambda a: (a.occ, a.altloc))
    return chosen, dict(
        residue=int(residue.seqid.num),
        chosen_altloc=chosen.altloc,
        chosen_occupancy=round(float(chosen.occ), 4),
        alternatives={a.altloc: round(float(a.occ), 4) for a in atoms},
    )


def _ca_coordinates(residues, label):
    """CA coordinates plus the record of every alternate conformation resolved."""
    coords, altlocs = [], []
    for residue in residues:
        atom, record = _select_ca(residue, label)
        if record is not None:
            altlocs.append(record)
        pos = atom.pos
        coords.append([pos.x, pos.y, pos.z])
    return np.asarray(coords, dtype=np.float32), altlocs


def _audit_numbering(residues, label):
    """Refuse an ambiguous residue correspondence rather than papering over it.

    Section 17.3 step 1 requires failing explicitly on insertion codes, gaps
    and missing CA atoms: the sequence fed to the model and the coordinates
    compared against it index the same array, so a silently dropped residue
    shifts every downstream position. A crystal structure can have all three;
    `P17_Alpha.pdb` happens to have none, and this is what establishes that
    rather than assuming it.
    """
    icodes = sorted({r.seqid.icode.strip() for r in residues if r.seqid.icode.strip()})
    if icodes:
        raise ValueError(f"{label} has insertion codes {icodes}; correspondence is ambiguous")
    numbers = [r.seqid.num for r in residues]
    gaps = [(a, b) for a, b in zip(numbers, numbers[1:]) if b != a + 1]
    if gaps:
        raise ValueError(
            f"{label} residue numbering is not contiguous: breaks at {gaps}. "
            "A gap means the extracted sequence omits residues present in the "
            "construct, which would misalign the model input against the "
            "reference coordinates."
        )
    return dict(first=numbers[0], last=numbers[-1], count=len(numbers))


def load_complex(path, binder_chain, target_chain):
    """Sequences, CA coordinates and a numbering audit for one reference.

    Returns a dict rather than a tuple: callers need the audit and the digest
    recorded alongside the arrays, and a positional API invites silently
    swapping binder for target.
    """
    path = Path(path)
    structure = gemmi.read_structure(str(path))
    structure.setup_entities()
    model = structure[0]
    present = {chain.name for chain in model}
    missing = {binder_chain, target_chain} - present
    if missing:
        raise ValueError(
            f"{path.name} has chains {sorted(present)}; requested {sorted(missing)} absent"
        )

    out = dict(
        source=str(path),
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        binder_chain=binder_chain,
        target_chain=target_chain,
    )
    for role, name in (("binder", binder_chain), ("target", target_chain)):
        residues = _amino_acids(model[name])
        if not residues:
            raise ValueError(f"{path.name} chain {name} holds no amino acids")
        label = f"{path.name} chain {name} ({role})"
        out[f"{role}_seq"] = gemmi.one_letter_code([r.name for r in residues]).upper()
        ca, altlocs = _ca_coordinates(residues, label)
        out[f"{role}_ca"] = ca
        out[f"{role}_altlocs_resolved"] = altlocs
        out[f"{role}_numbering"] = _audit_numbering(residues, label)
        out[f"{role}_dropped_residues"] = len(model[name]) - len(residues)
    if "X" in out["binder_seq"] or "X" in out["target_seq"]:
        raise ValueError(f"{path.name} contains residues gemmi could not code as standard")
    return out


def ca_spacing_report(ca, label):
    """Adjacent-CA spacing, as the reference geometry sanity check."""
    deltas = np.linalg.norm(np.diff(ca, axis=0), axis=-1)
    low, high = CA_SPACING_RANGE_A
    outliers = np.flatnonzero((deltas < low) | (deltas > high))
    return dict(
        label=label,
        median_A=float(np.median(deltas)),
        min_A=float(deltas.min()),
        max_A=float(deltas.max()),
        outlier_positions=[int(i) for i in outliers],
        passed=bool(outliers.size == 0),
    )


def contact_epitope(binder_ca, target_ca, contact_distance):
    """Target positions within `contact_distance` of any binder CA, 0-indexed.

    The JN.1 runs anchor the contact loss on five literature hotspots. Those
    indices are JN.1's own numbering and do not transfer: the Alpha target is
    195 residues against JN.1's 184, with its own 334-528 numbering. Deriving
    the epitope from the reference complex instead keeps it a property of the
    structure being recovered rather than a transferred constant, and the
    control's question is geometric recovery, not hotspot biology.

    CA-CA at 8 A is a coarse proxy for an interface: it neither implies a
    side-chain contact nor identifies energetically important residues.
    """
    distances = np.linalg.norm(
        np.asarray(binder_ca)[:, None, :] - np.asarray(target_ca)[None, :, :], axis=-1
    )
    return np.flatnonzero((distances <= contact_distance).any(axis=0)).astype(np.int32)


def ca_distance_matrix(binder_ca, target_ca):
    """Binder x target CA distances -- the pose-drift loss reference."""
    return np.linalg.norm(
        np.asarray(binder_ca)[:, None, :] - np.asarray(target_ca)[None, :, :], axis=-1
    )


# Damage generation ----------------------------------------------------------
#
# The control's logic: destroy the binder's CDRs until OpenDDE's confidence
# falls into the non-binding regime, then measure whether the gradient climbs
# back. A solution is guaranteed to exist in the feasible set, because the
# undamaged sequence is in it.

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def damage_sequence(sequence, designable_idx, n_edits, seed, *, forbid_wt=False):
    """Substitute `n_edits` designable positions, deterministically from `seed`.

    Random substitution, not chemically motivated disruption. Choosing
    "maximally disruptive" residues by hand would import exactly the
    structural biology this control is built to avoid, and would make the
    damage a second hypothesis rather than a fixed starting point.

    `forbid_wt` records the harder variant discussed in section 19.3B: with it
    set, the recovery search is still permitted to restore the original
    residue -- this flag only constrains how damage is *generated*, so an
    identity substitution cannot be drawn and the requested edit count is the
    realized one. Forbidding recovery to WT is a search-side constraint and is
    deliberately not implemented here.

    Returns the damaged sequence plus the exact substitutions, so the damaged
    starting point is recorded before anything is scored (section 19.3B: fixing
    it after seeing recovery results would make this a selection-biased search
    for an easy case).
    """
    designable_idx = np.asarray(designable_idx, dtype=int)
    if n_edits < 1:
        raise ValueError("n_edits must be at least 1")
    if n_edits > designable_idx.size:
        raise ValueError(
            f"requested {n_edits} edits across {designable_idx.size} designable positions"
        )
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(designable_idx, size=n_edits, replace=False))
    residues = list(sequence)
    substitutions = []
    for position in positions:
        wt_residue = residues[position]
        choices = [a for a in AMINO_ACIDS if a != wt_residue]
        if forbid_wt:
            # Nothing further to exclude: `choices` already omits WT. Kept
            # explicit so the flag's meaning is visible at the call site.
            pass
        new_residue = choices[int(rng.integers(len(choices)))]
        residues[position] = new_residue
        substitutions.append(
            dict(position_0idx=int(position), wt=wt_residue, damaged=new_residue)
        )
    damaged = "".join(residues)
    hamming = sum(a != b for a, b in zip(sequence, damaged))
    if hamming != n_edits:
        raise ValueError(f"requested {n_edits} edits, realized {hamming}")
    return damaged, substitutions


def damage_ladder(sequence, designable_idx, edit_counts, seed, *, forbid_wt=False):
    """One damaged sequence per edit count, each from a derived sub-seed.

    Section 19.3B asks for graded damage severity rather than one binary
    attempt: recovery from two destroyed positions versus eight is a curve,
    and a single level cannot distinguish "the gradient works" from "this one
    case was easy". Sub-seeds are derived from `seed` and the edit count so
    each level is reproducible on its own, and so adding a level does not
    change the sequences already generated for the others.
    """
    rungs = []
    for count in sorted(set(int(c) for c in edit_counts)):
        derived = int(np.random.SeedSequence([seed, count]).generate_state(1)[0])
        damaged, substitutions = damage_sequence(
            sequence, designable_idx, count, derived, forbid_wt=forbid_wt
        )
        rungs.append(
            dict(
                n_edits=count,
                seed=derived,
                sequence=damaged,
                substitutions=substitutions,
                hamming_from_wt=count,
            )
        )
    return rungs


def main(argv=None):
    """Emit the damage ladder as JSON, for a launcher to consume.

    Separated from scoring so the damaged starting points exist on disk,
    recorded and hashed, before any prediction runs. Loads no model.
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Damage ladder for the recovery control")
    parser.add_argument("--complex", type=Path, default=ALPHA_COMPLEX)
    parser.add_argument("--binder-chain", default=ALPHA_BINDER_CHAIN)
    parser.add_argument("--target-chain", default=ALPHA_TARGET_CHAIN)
    parser.add_argument("--edits", type=int, nargs="+", default=[2, 5, 8, 12])
    parser.add_argument("--damage-seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from p17_hallucination_search import CDR_RESIDUE_INDICES_1IDX

    reference = load_complex(args.complex, args.binder_chain, args.target_chain)
    binder_seq = reference["binder_seq"]
    designable_idx = np.array(
        [i for i in range(len(binder_seq)) if i + 1 in CDR_RESIDUE_INDICES_1IDX]
    )
    spacing = [
        ca_spacing_report(reference["binder_ca"], "binder"),
        ca_spacing_report(reference["target_ca"], "target"),
    ]
    failed = [r for r in spacing if not r["passed"]]
    if failed:
        raise ValueError(f"reference backbone spacing failed: {failed}")

    payload = dict(
        reference=dict(
            source=reference["source"],
            source_sha256=reference["source_sha256"],
            binder_chain=reference["binder_chain"],
            target_chain=reference["target_chain"],
            binder_seq=binder_seq,
            target_seq=reference["target_seq"],
            binder_numbering=reference["binder_numbering"],
            target_numbering=reference["target_numbering"],
            binder_altlocs_resolved=reference["binder_altlocs_resolved"],
            target_altlocs_resolved=reference["target_altlocs_resolved"],
            dropped_residues=dict(
                binder=reference["binder_dropped_residues"],
                target=reference["target_dropped_residues"],
            ),
            spacing=spacing,
        ),
        designable_positions_0idx=[int(i) for i in designable_idx],
        damage_seed=args.damage_seed,
        interpretation=(
            "Damaged starting points for the gradient recovery control, fixed "
            "before scoring. Recovery toward the reference's own confidence "
            "and pose is the proof-of-concept readout; it establishes nothing "
            "about binding."
        ),
        rungs=damage_ladder(binder_seq, designable_idx, args.edits, args.damage_seed),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {len(payload['rungs'])} damage rungs to {args.out}")
    for rung in payload["rungs"]:
        positions = sorted(s["position_0idx"] + 1 for s in rung["substitutions"])
        print(f"  {rung['n_edits']:2d} edits at 1-indexed positions {positions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Decoy targets --------------------------------------------------------------
#
# The negative control the design lacked (docs/P17_JN1.md section 20.11 item 1):
# every control so far asks whether a solution exists, none asks whether the
# pipeline would report success against a target it should fail on.


def epitope_scrambled_target(sequence, epitope_idx, seed):
    """A decoy that keeps the target's fold and destroys only its interface.

    The two cheaper decoys were measured on 2026-10-05 and both fail the fold
    confound in this predictor:

        decoy                      mean target pLDDT
        shuffled JN.1 (184 aa)                 0.405
        IL7RA, trimmed to 184 aa               0.425
        -- real targets, same path --
        JN.1 RBD                               0.890
        Alpha RBD                              0.967

    The shuffle has no native fold, which was expected. IL7RA failing was not:
    all three of its disulfides survive the trim, so truncation does not
    explain it. This predictor folds the RBD targets it was trained around and
    does not fold an unrelated receptor from sequence alone, which makes a
    generic out-of-distribution decoy unusable -- its low interface confidence
    would follow from the target not folding. An in-distribution decoy (another
    sarbecovirus RBD) has the opposite problem: P17 is an anti-RBD binder, so
    partial cross-reactivity is exactly what one would expect there.

    Scrambling only the epitope avoids both horns. The chain stays the real
    target everywhere outside the interface, so the predictor can still fold it
    -- which the caller's pLDDT gate verifies rather than assumes -- while the
    surface the binder needs is gone. Length and composition are preserved
    exactly, so the arm still differs from the real one in one input.

    Residues are permuted among the epitope positions, so a position can retain
    its own residue by chance; the returned identity records how many did.
    """
    epitope_idx = np.asarray(epitope_idx, dtype=int)
    if epitope_idx.size < 2:
        raise ValueError("need at least two epitope positions to permute")
    if epitope_idx.min() < 0 or epitope_idx.max() >= len(sequence):
        raise ValueError("epitope indices fall outside the target sequence")
    if len(set(epitope_idx.tolist())) != epitope_idx.size:
        raise ValueError("duplicate epitope indices")
    rng = np.random.default_rng(seed)
    residues = list(sequence)
    patch = [residues[i] for i in epitope_idx]
    rng.shuffle(patch)
    for position, residue in zip(epitope_idx, patch):
        residues[position] = residue
    scrambled = "".join(residues)
    if sorted(scrambled) != sorted(sequence):
        raise ValueError("scramble changed the composition")
    kept = sum(scrambled[i] == sequence[i] for i in epitope_idx)
    return scrambled, dict(
        kind="epitope_scrambled",
        seed=seed,
        length=len(scrambled),
        epitope_positions=epitope_idx.tolist(),
        epitope_size=int(epitope_idx.size),
        epitope_residues_unchanged_by_chance=int(kept),
        identity_to_real_target=round(
            sum(a == b for a, b in zip(scrambled, sequence)) / len(sequence), 4
        ),
        composition_matched=True,
        fold_preserved_outside_the_epitope=True,
    )


def real_decoy_target(path, chain, length):
    """A real unrelated protein, trimmed to `length`, as a decoy target.

    The shuffled decoy of `shuffled_target` is composition-matched but has no
    native fold -- measured at 0.405 mean target pLDDT on 2026-10-05, against
    the binder's 0.843 in the same prediction -- so its low interface
    confidence is explained by the predictor failing to fold it. A real protein
    removes that confound.

    The length has to match the reference target exactly, because the reference
    coordinates and every array shape are reused unchanged. Natural chains
    rarely have the required length, so the chain is trimmed symmetrically at
    the termini, which are the residues most often disordered. Whether the
    trimmed chain still folds is not assumed: the caller's fold check measures
    its pLDDT, and the kept range is recorded here so the trim is auditable.
    """
    structure = gemmi.read_structure(str(path))
    structure.setup_entities()
    structure.remove_ligands_and_waters()
    try:
        residues = [r for r in structure[0][chain]]
    except (KeyError, RuntimeError, ValueError) as error:
        chains = [c.name for c in structure[0]]
        raise ValueError(f"chain {chain!r} not in {path} (has {chains})") from error
    sequence = gemmi.one_letter_code([r.name for r in residues]).upper()
    if set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"):
        raise ValueError(
            f"chain {chain} of {path} has non-standard residues, which the "
            "search alphabet cannot represent"
        )
    if len(sequence) < length:
        raise ValueError(
            f"chain {chain} of {path} is {len(sequence)} aa, shorter than the "
            f"{length} aa the reference target requires; trimming cannot "
            "lengthen it"
        )
    drop = len(sequence) - length
    start = drop // 2
    trimmed = sequence[start : start + length]
    assert len(trimmed) == length
    return trimmed, dict(
        kind="real_trimmed",
        source=str(path),
        chain=chain,
        source_length=len(sequence),
        length=length,
        trimmed_from_n_terminus=start,
        trimmed_from_c_terminus=drop - start,
        kept_author_residues=[
            int(residues[start].seqid.num),
            int(residues[start + length - 1].seqid.num),
        ],
        composition_matched=False,
    )


def shuffled_target(sequence, seed):
    """A length- and composition-matched decoy target, deterministically.

    Matching the length matters mechanically: the reference coordinates belong
    to the real target, and reusing them keeps the pose loss, contact epitope
    and every array shape identical, so the decoy arm differs from the real arm
    in exactly one input. Matching composition removes amino-acid frequency as
    an explanation for any confidence difference.

    ⚠️ This decoy has a confound that must be checked before its result is
    used: a shuffled sequence has no native fold, so if the predictor cannot
    fold it the interface confidence will be low for reasons that have nothing
    to do with binding specificity, and the control passes vacuously. The
    check is the target-fit RMSD and pLDDT of the decoy's own WT prediction --
    if the target does not place consistently, the control establishes nothing.
    A real unrelated protein avoids this confound and costs a sequence source;
    `--decoy-sequence` accepts one.
    """
    rng = np.random.default_rng(seed)
    residues = list(sequence)
    rng.shuffle(residues)
    shuffled = "".join(residues)
    if sorted(shuffled) != sorted(sequence):
        raise ValueError("shuffle changed the composition")
    identity = sum(a == b for a, b in zip(shuffled, sequence)) / len(sequence)
    return shuffled, dict(
        kind="shuffled",
        seed=seed,
        length=len(shuffled),
        identity_to_real_target=round(identity, 4),
        composition_matched=True,
    )
