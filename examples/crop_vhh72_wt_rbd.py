"""Crop the WT RBD target chain in vhh72_wt_wt_rbd.cif to the
CDR-interface-proximal region, to reduce the atom count fed through
OpenDDE's differentiable forward+backward pass
(examples/vhh72_gradient_path_comparison.py OOMs on this GPU at full size --
see docs/guidance_alphaseq_testing_notes.md section 9 point 1 for context).

Interface criterion: Calpha-Calpha distance <= CUTOFF_ANGSTROM (8.0A),
between any CDR-residue Calpha and any target-residue Calpha -- not
any-heavy-atom minimum distance (an earlier version of this script used
10A over all heavy atoms; Calpha-only is a tighter, backbone-level
definition of "interface residue," standard for defining a contact
interface, and was re-derived on request, not left as the old default).
5A was tried first and gave only 3 target residues -- too small a shell to
carry any real local backbone context -- so the cutoff was widened to 8A.

The VHH72 binder chain (A, 125 residues) is kept entirely unchanged -- only
the target chain (A2, WT SARS-CoV-2 RBD) is cropped, selected by distance to
the binder's CDR atoms specifically (positions 26-33/51-58/97-114, the same
IMGT-derived indices as CDR_RESIDUE_INDICES in
vhh72_gradient_path_comparison.py) -- not the whole binder chain. Guidance
only ever touches these 34 designable positions; a target residue near
framework only (never redesigned) can't affect anything this comparison
measures, so restricting to CDR-only proximity is both a tighter crop and a
more accurate match for what's actually being tested. This is safe with
respect to everything already built against this structure:
cfg.cdr_residue_indices refers only to binder-chain positions, untouched by
cropping the target.

Two selections are computed:
  - the full interface shell (target residues within CUTOFF_ANGSTROM of any
    CDR atom) -- this is what actually gets written to the cropped CIF.
  - a smaller "hotspot" subset -- the TOP_N_HOTSPOT target residues with the
    single smallest minimum distance to any CDR atom, i.e. the closest,
    most-likely-critical contacts within the shell. Reported for reference
    (e.g. to sanity-check the crop looks like a real, focused epitope), not
    written as a separate structure -- this pipeline has no target-hotspot
    config field (checked directly against build_complex_yaml and the
    upstream boltzgen/joltzgen packages: no such concept exists anywhere in
    this stack). The crop itself is what encodes "where to bind" for
    BoltzGen, since the model only ever sees target residues that are
    actually included in the complex.

No sequence-window padding around selected shell residues (kept simple; can
add a +/-N residue buffer later if the model needs more local backbone
context than a bare interface shell provides). Original residue numbering
is preserved (not renumbered), so this cropped file can be compared
directly against the original.

Usage:
    .venv/bin/python examples/crop_vhh72_wt_rbd.py
"""
from pathlib import Path

import gemmi
import numpy as np

CUTOFF_ANGSTROM = 8.0
TOP_N_HOTSPOT = 8

# 1-indexed, matching res.seqid.num on chain A -- same as CDR_RESIDUE_INDICES
# in vhh72_gradient_path_comparison.py. Derived via a real ANARCI run (IMGT
# numbering scheme) on the real WT VHH72 sequence, verified fresh (not
# trusted from memory) via examples/anarci_vhh72_cdr_boundaries.py --
# confirmed exact match against these same 0-indexed boundaries (CDR1
# 25-32, CDR2 50-57, CDR3 96-113) before this script was written.
CDR_RESIDUE_INDICES = set(range(26, 34)) | set(range(51, 59)) | set(range(97, 115))

REPO_ROOT = Path(__file__).resolve().parent.parent
IN_CIF = REPO_ROOT / "vhh72_wt_wt_rbd.cif"
OUT_CIF = REPO_ROOT / "vhh72_wt_wt_rbd_cropped.cif"

print(f"=== Cropping {IN_CIF.name} target chain to CDR interface (cutoff={CUTOFF_ANGSTROM}A) ===", flush=True)

st = gemmi.read_structure(str(IN_CIF))
st.setup_entities()
model = st[0]

binder_chain = model["A"]
target_chain = model["A2"]
print(f"before: binder(A)={len(binder_chain)} residues, target(A2)={len(target_chain)} residues", flush=True)

cdr_residues = [res for res in binder_chain if res.seqid.num in CDR_RESIDUE_INDICES]
assert len(cdr_residues) == len(CDR_RESIDUE_INDICES), (
    f"expected {len(CDR_RESIDUE_INDICES)} CDR residues by seqid.num, found {len(cdr_residues)} "
    f"-- binder chain numbering may not match CDR_RESIDUE_INDICES' assumption"
)
def _ca_coord(res):
    for a in res:
        if a.name == "CA":
            return np.array([a.pos.x, a.pos.y, a.pos.z])
    return None

cdr_ca_coords = np.array([c for c in (_ca_coord(res) for res in cdr_residues) if c is not None])
assert len(cdr_ca_coords) == len(cdr_residues), (
    f"expected a Calpha on every one of the {len(cdr_residues)} CDR residues, "
    f"found {len(cdr_ca_coords)} -- check for missing backbone atoms"
)
print(f"CDR Calpha atoms (34 residues): {len(cdr_ca_coords)}", flush=True)
atoms_before = sum(len(r) for r in binder_chain) + sum(len(r) for r in target_chain)

kept_target_residues = []
skipped_no_ca = 0
for res in target_chain:
    ca = _ca_coord(res)
    if ca is None:
        skipped_no_ca += 1
        continue
    dists = np.linalg.norm(cdr_ca_coords - ca[None, :], axis=-1)
    min_dist = float(dists.min())
    if min_dist <= CUTOFF_ANGSTROM:
        kept_target_residues.append((res, min_dist))

if skipped_no_ca:
    print(f"(skipped {skipped_no_ca} target residues with no Calpha atom)", flush=True)
print(f"target residues with Calpha within {CUTOFF_ANGSTROM}A of a CDR Calpha: "
      f"{len(kept_target_residues)} (of {len(target_chain)})", flush=True)

hotspot = sorted(kept_target_residues, key=lambda rd: rd[1])[:TOP_N_HOTSPOT]
hotspot_str = ", ".join(
    f"{r.name}{r.seqid.num}({d:.2f}A)" for r, d in sorted(hotspot, key=lambda rd: rd[0].seqid.num)
)
print(f"\ntop {TOP_N_HOTSPOT} hotspot target residues (closest Calpha to any CDR Calpha): {hotspot_str}", flush=True)

kept_seqids = {res.seqid.num for res, _ in kept_target_residues}
kept_names = "".join(gemmi.find_tabulated_residue(r.name).one_letter_code.upper() for r, _ in kept_target_residues)
print(f"kept target seq_ids: {sorted(kept_seqids)}", flush=True)
print(f"kept target residues (by original numbering, not necessarily contiguous): {kept_names}", flush=True)

# Remove target residues NOT in the kept set. Iterate a static list of
# indices in reverse so removal doesn't shift subsequent indices.
to_remove = [i for i, res in enumerate(target_chain) if res.seqid.num not in kept_seqids]
for i in reversed(to_remove):
    del target_chain[i]

print(f"after: binder(A)={len(model['A'])} residues, target(A2)={len(model['A2'])} residues", flush=True)

# NOT a bug, reverted: st.setup_entities() deliberately does NOT rebuild
# entity.full_sequence from the post-deletion residue content --
# _entity_poly.pdbx_seq_one_letter_code stays the FULL, uncropped 209-residue
# target sequence even though _atom_site correctly has only the cropped 56
# residues. This is exactly what BoltzGen's own mmCIF parser
# (load_features_and_structure_writer, used by
# vhh72_gradient_path_comparison.py) depends on: it positionally aligns
# full_sequence against seqid numbering and treats the gaps as "unresolved"
# residues (see its own log line "Removing leading and/or trailing
# unresolved residues..."), which is exactly the semantics wanted here --
# NOT a length mismatch to fix. An earlier version of this script
# force-truncated full_sequence to match the kept residues, which fixed
# OpenDDE's `opendde json` CLI (a naive reader that takes full_sequence at
# face value) but broke BoltzGen's parser outright (AssertionError,
# polymer[i].name != res_name) since it destroyed the positional
# correspondence the "unresolved" gap-handling relies on. If OpenDDE's CLI
# needs a genuinely truncated sequence again, build its JSON input directly
# rather than truncating full_sequence here.

binder_atoms_after = sum(len(r) for r in model["A"])
target_atoms_after = sum(len(r) for r in model["A2"])
atoms_after = binder_atoms_after + target_atoms_after
print(f"after: binder(A)={binder_atoms_after} atoms, target(A2)={target_atoms_after} atoms, "
      f"total={atoms_after} atoms (was {atoms_before} before crop, "
      f"{atoms_after / atoms_before:.1%} of original)", flush=True)

st.setup_entities()
OUT_CIF.parent.mkdir(parents=True, exist_ok=True)
doc = st.make_mmcif_document()
doc.write_file(str(OUT_CIF))
print(f"\nwrote {OUT_CIF}", flush=True)

reparsed = gemmi.read_structure(str(OUT_CIF))
reparsed.setup_entities()
r_binder = reparsed[0]["A"]
r_target = reparsed[0]["A2"]
assert len(r_binder) == 125, f"binder chain changed length: {len(r_binder)} (expected 125, untouched)"
assert len(r_target) == len(kept_target_residues), (
    f"round-trip changed target residue count: {len(r_target)} vs {len(kept_target_residues)}"
)
print(f"round-trip OK: binder={len(r_binder)} residues (unchanged), "
      f"target={len(r_target)} residues (cropped)", flush=True)

reparsed_target_entity = next(e for e in reparsed.entities if "A2" in e.subchains)
print(
    f"entity.full_sequence deliberately NOT cropped ({len(reparsed_target_entity.full_sequence)} "
    f"residues, matching the original target) -- BoltzGen's parser needs this; "
    f"see the comment above. Naive readers (OpenDDE's `opendde json` CLI) will "
    f"see the full sequence, not the crop -- feed those a hand-built JSON with "
    f"the {len(kept_target_residues)}-residue sequence directly instead.",
    flush=True,
)
