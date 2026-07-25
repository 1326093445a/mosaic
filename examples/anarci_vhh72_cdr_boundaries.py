"""Derive VHH72's IMGT CDR boundaries from a real ANARCI run, and verify
them against the hardcoded values used elsewhere in this project
(examples/alphaseq_vhh72_cdr_contrast_pairs.py's CDR1/CDR2/CDR3,
examples/crop_vhh72_wt_rbd.py's CDR_RESIDUE_INDICES).

This does NOT call ANARCI itself -- ANARCI lives in a separate `anarci`
conda env, not a mosaic dependency (matches the existing precedent for the
raw-torch OpenDDE CLI, docs/guidance_alphaseq_testing_notes.md section 11).
Run ANARCI first, then parse its output with this script (pure Python, no
special deps beyond csv):

    conda run -n anarci ANARCI -i vhh72.fasta -s imgt --csv -o vhh72_anarci

where vhh72.fasta contains the real WT VHH72 sequence extracted directly
from vhh72_wt_wt_rbd.cif chain A (125 residues):

    QVQLQESGGGLVQAGGSLRLSCAASGRTFSEYAMGWFRQAPGKEREFVATISWSGGSTYYTDSVKGRFTI
    SRDNAKNTVYLQMNSLKPDDTAVYYCAAAGLGTVVSEWDYDYDYWGQGTQVTVSS

producing vhh72_anarci_H.csv, which this script reads.

Standard IMGT CDR loop boundaries (numeric IMGT position, inclusive) used
here: CDR1 27-38, CDR2 56-65, CDR3 105-117. ANARCI's own numbering already
accounts for IMGT insertion codes (e.g. 111A/111B/112C/112B/112A around the
long CDR3 loop) -- this script maps each numbered position back to its
0-indexed position in the original sequence by walking the CSV's per-column
residues in order and skipping gaps ('-'), not by re-deriving numbering
itself.

Usage (after running ANARCI as above):
    .venv/bin/python examples/anarci_vhh72_cdr_boundaries.py <path/to/vhh72_anarci_H.csv>
"""
import csv
import sys

WT_VHH72_SEQ = (
    "QVQLQESGGGLVQAGGSLRLSCAASGRTFSEYAMGWFRQAPGKEREFVATISWSGGSTYYTDSVKGRFTI"
    "SRDNAKNTVYLQMNSLKPDDTAVYYCAAAGLGTVVSEWDYDYDYWGQGTQVTVSS"
)

# Standard IMGT CDR boundaries (numeric IMGT position range, inclusive).
CDR_RANGES = {"CDR1": (27, 38), "CDR2": (56, 65), "CDR3": (105, 117)}

# What's currently hardcoded elsewhere in this project (0-indexed into
# WT_VHH72_SEQ), to diff against.
EXISTING_HARDCODED_0IDX = {
    "CDR1": set(range(25, 33)),
    "CDR2": set(range(50, 58)),
    "CDR3": set(range(96, 114)),
}

_META_COLS = {
    "Id", "domain_no", "hmm_species", "chain_type", "e-value", "score",
    "seqstart_index", "seqend_index", "identity_species", "v_gene",
    "v_identity", "j_gene", "j_identity",
}


def _imgt_num(label):
    i = 0
    while i < len(label) and label[i].isdigit():
        i += 1
    return int(label[:i])


def parse_anarci_csv(csv_path):
    """Returns (reconstructed_sequence, imgt_labels) -- imgt_labels[i] is the
    IMGT position label for WT_VHH72_SEQ[i]."""
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        row = next(reader)
    pos_cols = [(h, v) for h, v in zip(header, row) if h not in _META_COLS]

    seq_chars, imgt_labels = [], []
    for label, aa in pos_cols:
        if aa == "-":
            continue
        seq_chars.append(aa)
        imgt_labels.append(label)
    return "".join(seq_chars), imgt_labels


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <path/to/vhh72_anarci_H.csv>")

    reconstructed, imgt_labels = parse_anarci_csv(sys.argv[1])
    print(f"=== Verifying ANARCI reconstruction against the real WT VHH72 sequence ===", flush=True)
    if reconstructed != WT_VHH72_SEQ:
        raise SystemExit(
            f"ANARCI-reconstructed sequence does not match WT_VHH72_SEQ -- "
            f"got {reconstructed!r}, expected {WT_VHH72_SEQ!r}. Do not trust "
            f"the CDR boundaries below until this is resolved."
        )
    print("reconstruction matches exactly (125/125 residues).\n", flush=True)

    print("=== IMGT CDR boundaries, derived fresh from this ANARCI run ===", flush=True)
    all_match = True
    for cdr, (lo, hi) in CDR_RANGES.items():
        idxs = [i for i, label in enumerate(imgt_labels) if lo <= _imgt_num(label) <= hi]
        derived = set(idxs)
        existing = EXISTING_HARDCODED_0IDX[cdr]
        match = derived == existing
        all_match &= match
        residues = "".join(WT_VHH72_SEQ[i] for i in idxs)
        print(
            f"{cdr}: 0-idx {min(idxs)}-{max(idxs)} (n={len(idxs)}) -- {residues}\n"
            f"  matches existing hardcoded {sorted(existing)[0]}-{sorted(existing)[-1]}: {match}",
            flush=True,
        )

    print(
        f"\n{'All three CDRs match the existing hardcoded boundaries exactly.' if all_match else 'MISMATCH -- existing hardcoded boundaries need updating.'}",
        flush=True,
    )
