# Site-restricted and site-restricted interface confidence metrics

Scope note: all source code quoted below was read directly from the upstream repositories during this
session (2026-10-05). Where a fact comes from a paper abstract or a search-engine summary rather than
a primary read, it is labelled as such.

---

## Q1. ipSAE: exact definition, d0 normalisation, asym/max variants, and whether the implementation supports a residue subset

### Takeaway

ipSAE is a per-residue, PAE-cutoff-masked, d0-renormalised pTM over cross-chain residue pairs, aggregated
by **max over the aligned residue index** and then **max over the two chain directions**. The published
implementation (`ipsae.py`, v4, 3 Jan 2026) takes exactly four positional arguments and has **no option
to restrict scoring to a specified residue subset** — but the subset hook is a single boolean matrix
(`valid_pairs_matrix`) and is trivial to intersect with a specified site mask.

### Cited Findings

- ipSAE's pTM transform is defined verbatim as `def ptm_func(x,d0): return 1.0/(1+(x/d0)**2.0)`, i.e.
  `f(PAE, d0) = 1 / (1 + (PAE/d0)^2)` — [ipsae.py L111-112](https://raw.githubusercontent.com/DunbrackLab/IPSAE/main/ipsae.py)
- The d0 normalisation is the Yang & Skolnick TM-score form, clamped at a floor:
  `calc_d0(L) = max(1.0, 1.24*(L-15)^(1/3) - 1.8)` for `L > 27`, else `d0 = 1.0`; the floor is 2.0 for
  nucleic-acid chain pairs. The vectorised form clamps `L = max(26, L)` first and, per a comment dated
  `01.03.2026`, "now returns 1.00 instead of 1.04 for minimum value" —
  [ipsae.py L115-137](https://raw.githubusercontent.com/DunbrackLab/IPSAE/main/ipsae.py)
- The interface mask is `valid_pairs_matrix = np.outer(chains == chain1, chains == chain2) & (pae_matrix < pae_cutoff)`
  — a hard, non-differentiable boolean AND of (i) cross-chain membership and (ii) PAE below cutoff —
  [ipsae.py L737 and L783](https://raw.githubusercontent.com/DunbrackLab/IPSAE/main/ipsae.py)
- Per-residue ipSAE (the headline `ipSAE` column, i.e. the `d0res` variant) is
  `ipsae_d0res_byres[c1][c2][i] = mean_{j in valid_pairs(i)} f(PAE[i,j], d0res(i))`, where
  `d0res(i) = calc_d0_array(n0res(i))` and `n0res(i) = np.sum(valid_pairs_matrix, axis=1)[i]` =
  the number of chain2 residues whose PAE to residue i is below cutoff —
  [ipsae.py L786-800](https://raw.githubusercontent.com/DunbrackLab/IPSAE/main/ipsae.py)
- The three d0 variants differ **only** in which residue count feeds d0:
  - `n0chn = np.sum(chains==chain1) + np.sum(chains==chain2)` → `ipSAE_d0chn` (d0 from the summed chain lengths) — [ipsae.py L731-732](https://raw.githubusercontent.com/DunbrackLab/IPSAE/main/ipsae.py)
  - `n0dom = |{i in chain1 : any valid pair}| + |{j in chain2 : any valid pair}|` → `ipSAE_d0dom` — [ipsae.py L775-778](https://raw.githubusercontent.com/DunbrackLab/IPSAE/main/ipsae.py)
  - `n0res(i)` as above → `ipSAE` (the recommended score) — [ipsae.py L786-790](https://raw.githubusercontent.com/DunbrackLab/IPSAE/main/ipsae.py)
  The README confirms: "**ipSAE**=ipSAE value for given PAE cutoff and d0 determined by number of residues
  in 2nd chain with PAE<cutoff"; "**ipSAE_d0chn** … d0 = sum of chain lengths"; "**ipSAE_d0dom** … d0 =
  total number of residues in both chains with any interchain PAE<cutoff" —
  [IPSAE README L55-59](https://raw.githubusercontent.com/DunbrackLab/IPSAE/main/README.md)
- **asym** = `max_i` over aligned residues in chain1: `max_index = np.argmax(interchain_values); ipsae_d0res_asym[c1][c2] = interchain_values[max_index]`
  — [ipsae.py L821-846](https://raw.githubusercontent.com/DunbrackLab/IPSAE/main/ipsae.py)
- **max** = `max(asym(A→B), asym(B→A))`, taken explicitly with `maxvalue=max(ipsae_d0res_asym[chain1][chain2], ipsae_d0res_asym[chain2][chain1])`
  — [ipsae.py L878-895](https://raw.githubusercontent.com/DunbrackLab/IPSAE/main/ipsae.py). The README's
  `Type` column is documented as `"asym" or "max"; asym means asymmetric ipTM/ipSAE values; max is maximum
  of asym values` — [IPSAE README L53](https://raw.githubusercontent.com/DunbrackLab/IPSAE/main/README.md).
  **There is no `min` variant in the published implementation** (see Inferences).
- The CLI is strictly four positional arguments — `pae_file_path = sys.argv[1]; pdb_path = sys.argv[2];
  pae_cutoff = float(sys.argv[3]); dist_cutoff = float(sys.argv[4])`, guarded by `if len(sys.argv) < 5`.
  No `argparse`, no chain-selection flag, no residue-list flag, no specified site file —
  [ipsae.py L38-59](https://raw.githubusercontent.com/DunbrackLab/IPSAE/main/ipsae.py)
- README examples use PAE cutoffs of both 15 (AF2) and 10 (AF3, Boltz); the script does not hard-code a
  recommendation — [IPSAE README L12-23](https://raw.githubusercontent.com/DunbrackLab/IPSAE/main/README.md)
- `dist_cutoff` does **not** enter ipSAE itself; it only defines the reported `dist1`/`dist2` counts
  ("number of residues in chain 1 with PAE<cutoff and dist<cutoff from chain2") used for diagnostics and
  PyMOL output — [IPSAE README L87-89](https://raw.githubusercontent.com/DunbrackLab/IPSAE/main/README.md),
  computed at [ipsae.py L758-768](https://raw.githubusercontent.com/DunbrackLab/IPSAE/main/ipsae.py)
- The same script also emits pDockQ, pDockQ2 and LIS, with the original citations in its header —
  [ipsae.py L6-8](https://raw.githubusercontent.com/DunbrackLab/IPSAE/main/ipsae.py)
- Primary paper: Dunbrack, "Rēs ipSAE loquunt: What's wrong with AlphaFold's ipTM score and how to fix it",
  bioRxiv 2025.02.10.637595 — [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.02.10.637595v2);
  [PubMed 39990437](https://pubmed.ncbi.nlm.nih.gov/39990437/). The paper's framing is that ipTM
  "averages predicted alignment quality across all residue pairs between chains, including disordered
  regions and non-interacting domains" and that ipSAE "separate[s] true from false complexes more
  efficiently than AlphaFold2's ipTM score" — [search summary of the bioRxiv preprint / GitHub](https://github.com/dunbracklab/IPSAE)
- ipSAE is packaged on PyPI (`pip install ipsae`) — [PyPI: ipsae](https://pypi.org/project/ipsae/) — and
  reimplemented in Neurosnap as `neurosnap.algos.ipsae` —
  [Neurosnap docs](https://neurosnap.ai/docs/neurosnap.algos.ipsae.html)
- ipSAE has been vendored into ColabFold as `colabfold/alphafold/ipsae.py`, called via
  `ipsae.get_interface_scores(...)` and reported alongside `pDockQ2` in the batch log —
  [ColabFold batch.py L611-624](https://raw.githubusercontent.com/sokrypton/ColabFold/main/colabfold/batch.py)

### Inferences

- **ipSAE is max-pooled twice and is therefore structurally the worst possible site-agnostic metric.**
  `asym` takes `argmax` over *all* chain1 residues; `max` then takes the better of the two directions.
  A designed chain that forms one confident contact patch anywhere on the target saturates the score. This is the
  exact mechanism behind the reported failure mode (0.77 vs a true-complex 0.795 while docked 14-27 Å away),
  and it also explains why a permuted-specified site negative control can score *higher* than the real target:
  nothing in the formula references the intended site.
- The project description says "minimum over the two chain directions". That is **not** what `ipsae.py`
  does — it takes the maximum. Taking the min is a stricter, defensible modification (it requires both
  directions to be confident), but it is a local variant and should not be cited as "ipSAE" without
  qualification. Worth double-checking which the project's code actually computes.
- **Restricting ipSAE to a specified site is a one-line change.** Replace
  `valid_pairs_matrix = np.outer(chains==chain1, chains==chain2) & (pae_matrix < pae_cutoff)`
  with `... & epitope_col_mask[None, :]` (and, if the designed chain side should also be constrained,
  `& binder_row_mask[:, None]`). Everything downstream — `n0res`, `d0res`, the per-residue mean, the asym
  and max aggregations — then operates only over (designed-chain residue, specified-site residue) pairs. Call the result
  `ipSAE@epitope`. Note that this also shrinks `n0res`, hence shrinks `d0res`, hence makes the score
  *harsher* (smaller d0 means the PAE must be lower to score the same) — so a site-restricted ipSAE
  is not directly comparable in absolute value to the unrestricted one and needs its own reference scale.
- A cheap complementary guard: replace the `max_i` with a sum/mean over specified-site residues, or require a
  minimum `n0res` within the specified site, so a single lucky residue cannot carry the score.

### Gaps

- The bioRxiv PDF fetch returned garbled values for the `n0` definitions (it claimed `n0 = 4 / 2 / 0.5`
  for d0chn/d0dom/d0res, which is inconsistent with the source code and with the README's own example
  table showing `n0chn=1571, n0dom=553, n0res=290`). I did **not** resolve the paper's own equation
  numbering; the source-code definitions above should be treated as authoritative and the PDF extraction
  as unreliable.
- I found no published ipSAE variant, fork, or issue thread that adds specified site/residue-subset restriction.
  If one exists it is not discoverable by the searches run here.

---

## Q2. ipTM and pTM; is there a residue-restricted variant? actifpTM

### Takeaway

ipTM is already a *max over aligned-residue index i* of a weighted average over cross-chain partners j,
and the AlphaFold implementation exposes both a 1-D `residue_weights` vector and (in the ColabFold fork)
a full 2-D `pair_residue_weights` matrix. **actifpTM is exactly a residue-subset-restricted ipTM** — it
restricts the residue set to the predicted contact interface. The published code therefore already
contains the machinery to restrict ipTM to an *arbitrary user-specified* specified site; only the residue-set
selection step would need changing.

### Cited Findings

- AlphaFold's `predicted_tm_score` computes, per residue i:
  `d0 = 1.24*(max(num_res,19)-15)^(1/3) - 1.8`; `tm_per_bin = 1/(1 + bin_centers^2/d0^2)`;
  `predicted_tm_term = sum_bins softmax(logits) * tm_per_bin`; `pair_mask = asym_id[:,None] != asym_id[None,:]`
  for the interface case; `normed_residue_mask = pair_residue_weights / pair_residue_weights.sum(-1)`;
  `per_alignment = (predicted_tm_term * normed_residue_mask).sum(-1)`;
  `residuewise_iptm = per_alignment * residue_weights`. ipTM is the max over i of that vector —
  [ColabFold extra_ptm.py L42-103](https://raw.githubusercontent.com/sokrypton/ColabFold/main/colabfold/alphafold/extra_ptm.py)
- Critically, the ColabFold fork's signature is
  `predicted_tm_score_modified(logits, breaks, residue_weights=None, asym_id=None, pair_residue_weights=None, use_jnp=False)`
  with the docstring `pair_residue_weights: [num_res, num_res] unnormalized weights for actifptm calculation`
  — i.e. an **arbitrary user-supplied pairwise weight matrix** —
  [ColabFold extra_ptm.py L42-60](https://raw.githubusercontent.com/sokrypton/ColabFold/main/colabfold/alphafold/extra_ptm.py)
- actifpTM is published as Varga, Hegedűs et al. (AlphaPulldown group), "actifpTM: a refined confidence
  metric of AlphaFold2 predictions involving flexible regions", *Bioinformatics* 41(3):btaf107 (2025) —
  [Bioinformatics](https://academic.oup.com/bioinformatics/article/41/3/btaf107/8075121);
  [arXiv:2412.15970](https://arxiv.org/abs/2412.15970); [PMC11925850](https://pmc.ncbi.nlm.nih.gov/articles/PMC11925850/)
- Paper's own description of the modification: "for actifpTM we modify the masking of the original ipTM
  calculations to take into account the predicted distance probabilities as residue-pair weights", with
  the interface defined from "the contact probability map output by AF2 … the probabilities of each
  residue-pair to be within 8Å distance to each other (Cβ-Cβ atoms, Cɑ is taken for glycines)". It is
  "calculated both for the full complex as well as for each pair of chains" —
  [PMC11925850](https://pmc.ncbi.nlm.nih.gov/articles/PMC11925850/)
- Two implementations exist in `extra_ptm.py`:
  1. **Probability-weighted** (`get_actifptm_probs`): `cmap_copy` is zeroed everywhere except the
     chain-pair block, then passed as `pair_residue_weights`; `seq_mask` is set to 1 only on the two
     chains' residues. `cmap = get_contact_map(result, 8)` where
     `get_contact_map = (jax.nn.softmax(dist_logits) * (dist_bins < dist)).sum(-1)` — a **soft** contact
     probability from the distogram —
     [extra_ptm.py L22-27, L122-159](https://raw.githubusercontent.com/sokrypton/ColabFold/main/colabfold/alphafold/extra_ptm.py)
  2. **Binary-contact** (`get_actifptm_contacts`, the *default*, since `get_chain_and_interface_metrics`
     has `use_probs_extra=False`): `contacts = np.where(cmap[block] >= 0.6)`; the union of contacting
     global positions becomes `seq_mask = 1`; standard ipTM is then run on that subset; "In case of no
     confident contacts, the interface PTM score is set to 0" —
     [extra_ptm.py L162-203, L250-260](https://raw.githubusercontent.com/sokrypton/ColabFold/main/colabfold/alphafold/extra_ptm.py)
- `get_pairwise_iptm(result, asym_id, start_i, end_i, start_j, end_j)` — "This will calculate ipTM as
  usual, just between given chains" — sets `seq_mask = 1` on the two chains' index ranges only. This is a
  **ready-made residue-range-restricted ipTM**, currently driven by chain boundaries
  (`get_chain_indices(asym_id)`) rather than by a user specified site list —
  [extra_ptm.py L206-221, L29-39](https://raw.githubusercontent.com/sokrypton/ColabFold/main/colabfold/alphafold/extra_ptm.py)
- Activation in ColabFold: `calc_extra_ptm: bool = False` flag; `extra_ptm.get_chain_and_interface_metrics(result, input_features['asym_id'], ...)`;
  outputs `result['actifptm']`, `pairwise_actifptm`, `pairwise_iptm`, `per_chain_ptm`, plus a pairwise plot
  via `extra_ptm.plot_chain_pairwise_analysis` —
  [ColabFold batch.py L84, L438, L536-543, L556, L602-603, L1272-1334, L1700-1703](https://raw.githubusercontent.com/sokrypton/ColabFold/main/colabfold/batch.py).
  The paper says it is enabled by "checking the `calc_extra_ptm` checkbox in the Jupyter notebook" or
  "`--calc_extra_ptm` flag when running localcolabfold" —
  [PMC11925850](https://pmc.ncbi.nlm.nih.gov/articles/PMC11925850/)
- AlphaPulldown2 reports "the weighted ipTM+pTM metric, actifpTM, new generation ipSAE scoring, and
  FoldSeek-Multimer clustering" together — [bioRxiv 2025.09.09.675126](https://www.biorxiv.org/content/10.1101/2025.09.09.675126v2.full)
- The reference d0 in `predicted_tm_score_modified` is computed from `num_res = residue_weights.shape[0]`,
  i.e. the **full array length**, not the size of the restricted subset —
  [extra_ptm.py L70-78](https://raw.githubusercontent.com/sokrypton/ColabFold/main/colabfold/alphafold/extra_ptm.py)
- ColabDesign's in-training `get_ptm(inputs, outputs, interface=False)` calls the stock
  `confidence.predicted_tm_score(**pae, use_jnp=True)` with `residue_weights = inputs["seq_mask"]` and
  `asym_id` for the interface case — so ipTM during design is computed with a 1-D mask only —
  [ColabDesign loss.py L204-213](https://raw.githubusercontent.com/sokrypton/ColabDesign/main/colabdesign/af/loss.py)

### Inferences

- **actifpTM is the closest published template for what the project needs**, and the gap is one function
  call wide. `get_actifptm_contacts` picks the residue subset from `cmap >= 0.6`; swapping that for a
  user-supplied site index list (and keeping the designed chain's own residues in `seq_mask`) yields
  **"site-restricted ipTM"** directly, with zero new maths. Equivalently, use the probability-weighted
  path and set `pair_residue_weights = cmap ⊙ (binder_mask ⊗ epitope_mask)` so only designed chain↔specified site pairs
  carry weight.
- Because `d0` is derived from the *full* array length rather than the subset, a site-restricted ipTM
  built this way keeps the same d0 as the unrestricted ipTM, so values stay on a comparable scale — the
  opposite of the ipSAE case. This is an advantage for interpretability: `actifpTM@epitope` can be compared
  against the unrestricted `actifpTM` for the same complex, and their ratio is a clean "is the confident
  interface at the intended site?" statistic.
- actifpTM as published does **not** solve the project's problem. It restricts to whatever interface was
  formed, so it is still site-agnostic — in fact it is *more* permissive than ipTM, because it discards the
  non-interface residues that would otherwise dilute the score. Using stock actifpTM as a filter would make
  the off-target problem worse, not better.
- The `>= 0.6` contact-probability threshold and the `8 Å` distance are hard-coded; the paper does not
  expose them, and the residue set is not user-specifiable in the shipped code.

### Gaps

- The paper does not print a standalone closed-form equation for actifpTM (confirmed by direct read of
  PMC11925850); the authoritative definition is the code above.
- I did not find whether `actifptm` is reduced by `max` or by `mean` over the residuewise vector in the
  `get_chain_and_interface_metrics` tail (lines beyond 290 were not read). `get_per_chain_ptm` uses
  `.max()`, and stock ipTM uses max, so max is the strong presumption — but this should be verified
  before relying on it.
- No published "site-restricted ipTM" / "site-specific ipTM" metric was found under any of the search
  terms tried. This appears to be a genuine gap in the literature as of 2026.

---

## Q3. Interface PAE (iPAE / i_pAE) and i_pDAE (BindCraft2)

### Takeaway

i_pAE is a plain masked mean of the interface PAE block divided by 31; it is restrictable to a specified site by
construction (the mask is an outer product of two residue-weight vectors) and is differentiable. **i_pDAE
is the single most important finding here: it is ipSAE with the `PAE < cutoff` mask replaced by a
geometric `CA-CA ≤ 8 Å` contact mask**, implemented as a plain boolean matrix — so intersecting it with an
specified site column mask gives a site-restricted, ipSAE-grade score in one line.

### Cited Findings

- **i_pAE** (BindCraft2) = `chain_pair_pae_loss(..., aligned_chains=(binder,), reference_chains=(target,))`,
  whose body is `pae = metrics['pae'] / 31.0` then
  `_masked_mean(pae[aligned_rows[:,None], reference_rows[None,:]], aligned_weights[:,None] * reference_weights[None,:])`,
  with `_masked_mean(values, mask, eps=1e-8) = (values*mask).sum() / (mask.sum() + eps)` —
  [BindCraft2 loss.py L120, L216-225, L358-361](https://raw.githubusercontent.com/PacesaLab/BindCraft2/main/bindcraft/loss.py)
  and the filter wrapper at
  [BindCraft2 filters.py L95-99](https://raw.githubusercontent.com/PacesaLab/BindCraft2/main/bindcraft/filters.py)
- The same `/31.0` normalisation and mask structure is in ColabDesign:
  `get_pae_loss(outputs, mask_1d=None, mask_1b=None, mask_2d=None)` → `p = get_pae(outputs)/31.0; p = (p+p.T)/2; mask_2d = mask_2d * mask_1d[:,None] * mask_1b[None,:]; return mask_loss(p, mask_2d)`
  — [ColabDesign loss.py L251-259](https://raw.githubusercontent.com/sokrypton/ColabDesign/main/colabdesign/af/loss.py)
- BindCraft1 exposes `i_pae` both as a design-loss weight (`"i_pae": advanced_settings["weights_pae_inter"]`)
  and as a post-hoc filter (`'i_pAE': round(prediction_metrics['i_pae'], 2)`, compared with `'<='`) —
  [BindCraft colabdesign_utils.py L42, L267, L277](https://raw.githubusercontent.com/martinpacesa/BindCraft/main/functions/colabdesign_utils.py)
- **i_pDAE** is BindCraft2's ranking metric: `RANKING_METRIC = 'i_pDAE'` and
  `LEADING_CONFIDENCE_COLUMNS = (RANKING_METRIC, 'i_pTM', 'pLDDT', 'pTM', 'i_pAE', ...)` —
  [BindCraft2 campaign_output.py L39, L89](https://raw.githubusercontent.com/PacesaLab/BindCraft2/main/bindcraft/campaign_output.py);
  the README calls `3_Ranked/!_Ranked.csv` "Accepted designs ordered by `i_pDAE`, a distance-masked
  interface confidence score" —
  [BindCraft2 README L202](https://raw.githubusercontent.com/PacesaLab/BindCraft2/main/README.md)
- i_pDAE's exact definition (read verbatim):
  ```python
  def tm_score_distance_scale(partner_count):
      return jnp.maximum(1.24 * (jnp.maximum(partner_count, 19.0) - 15.0) ** (1/3) - 1.8, 1.0)

  def anchored_interface_tm_scores(pae, contact):
      partner_counts = contact.sum(-1).astype(jnp.float32)
      tm_scores = 1.0 / (1.0 + (pae / tm_score_distance_scale(partner_counts)[:, None]) ** 2)
      return jnp.where(contact, tm_scores, 0.0).sum(-1) / jnp.maximum(partner_counts, 1.0)
  ```
  with `contact = (pairwise CA-CA distance <= cutoff) & resolved`, `cutoff = 8.0` by default,
  `interface_pae = pae[binder_slice, target_slice]`, both directions returned as
  `((interface_pae, contact), (interface_pae.T, contact.T))`, and
  `i_pDAE = max over both directions of anchored_interface_tm_scores(...).max()` —
  [BindCraft2 filters.py L118-169](https://raw.githubusercontent.com/PacesaLab/BindCraft2/main/bindcraft/filters.py)
- A **per-residue i_pDAE track** is already produced: `residue_confidence_tracks(...)` writes
  `tracks['i_pDAE'] = interface`, an array of per-residue anchored interface TM scores with `nan` where a
  residue has no contacts —
  [BindCraft2 filters.py L143-162](https://raw.githubusercontent.com/PacesaLab/BindCraft2/main/bindcraft/filters.py)
- `Interface_Residues` is reported as a metric: `binder_assembly_contact_masks(..., cutoff=4.0)` returns
  per-residue boolean contact masks for **both** designed chain and target, and the metric is the designed chain-side count
  — [BindCraft2 filters.py L101-116](https://raw.githubusercontent.com/PacesaLab/BindCraft2/main/bindcraft/filters.py)
- Reported BindCraft2 defaults, from search summary of the BindCraft2 docs (not verified against the
  primary file, which 404'd at the `docs/reference.md` path): "i_pDAE is a distance-masked interface
  confidence metric on a scale of 0–1, where higher values are better"; "i_pAE represents the mean
  interface PAE / 31 Å, where lower values are better"; filter `max_ipae_final`; default "normalized
  interface PAE at most 0.35" —
  [search summary, BindCraft2 docs/reference.md](https://github.com/PacesaLab/BindCraft2/blob/main/docs/reference.md)

### Inferences

- **i_pDAE is the best off-the-shelf starting point for a site-restricted score**, better than ipSAE,
  for three reasons: (1) the mask is geometric (`CA-CA ≤ 8 Å`) rather than confidence-based, so it cannot
  be satisfied by confident-but-distant pairs; (2) the mask is an explicit boolean matrix passed as an
  argument, so `contact & epitope_mask[None, :]` is a legitimate one-line restriction; (3) a per-residue
  track already exists, so per-specified site-residue reporting is free.
- Restricting i_pDAE: pass `contact_restricted = contact & epitope_mask[None, :]` in the designed chain→target
  direction (and `& epitope_mask[:, None]` in the transposed direction). Because `partner_counts` then
  counts only specified site partners, `d0` shrinks and the score hardens — same caveat as for ipSAE. If a
  stable scale matters, freeze `partner_counts` at the unrestricted value and only restrict the summation.
- **The `max` in i_pDAE has the same site-agnosticism failure as ipSAE.** `anchored_interface_tm_scores(...).max()`
  over both directions means one good residue suffices. For site targeting, replace with a mean (or a
  soft-max at low temperature) over the specified-site residues, and additionally require a minimum number of
  specified-site residues in contact.
- i_pAE's hard `/31.0` is just the PAE head's max bin, used to map PAE into [0,1]; it carries no
  site information and should not be expected to discriminate sites.
- A useful composite the code already supports: report `i_pDAE@epitope / i_pDAE` as a *site fidelity ratio*.
  Near 1 means the confident interface is the intended one; near 0 means the designed chain found a different patch.

### Gaps

- `docs/reference.md` does not exist at the path the search engine returned (404 on both the GitHub blob
  and the raw URL); the documented thresholds above are therefore second-hand. The code-level definitions
  are first-hand.
- I did not locate a BindCraft2 paper/preprint in this session, so i_pDAE has no citable primary
  publication here — only the repository.

---

## Q4. LIS, pDockQ, pDockQ2, DockQ — restrictability to a specified specified site

### Takeaway

pDockQ2 and LIS are both plain masked averages over an interface block and are therefore naturally
restrictable to a specified site sub-block; pDockQ is restrictable in principle but its fitted sigmoid breaks
down because the contact count enters logarithmically. DockQ cannot be restricted to a user-specified
specified site in any useful way for design, because it requires a reference complex — Fnat is already a
*native-interface* recall, not an *arbitrary-site* recall.

### Cited Findings

- **pDockQ** (Bryant, Pozzati, Elofsson, *Nat Commun* 13:1265, 2022 — [Nature Communications](https://www.nature.com/articles/s41467-022-28865-w)),
  as implemented in ipSAE: interface defined by `distances[i] <= pDockQ_cutoff` with `pDockQ_cutoff = 8.0`
  (CB-CB, CA for Gly); `nres = |unique interface residues|`; `mean_plddt = cb_plddt[interface].mean()`;
  `x = mean_plddt * math.log10(npairs)`; and
  `pDockQ = 0.724 / (1 + exp(-0.052*(x - 152.611))) + 0.018` —
  [ipsae.py L643-669](https://raw.githubusercontent.com/DunbrackLab/IPSAE/main/ipsae.py)
- **pDockQ2** (Zhu, Shenoy, Kundrotas, Elofsson, *Bioinformatics* 39(7):btad424, 2023 —
  [Bioinformatics](https://academic.oup.com/bioinformatics/article/39/7/btad424/7219714)), as implemented
  in ipSAE: `pae_list_ptm = ptm_func_vec(pae_list, 10.0)` (**d0 fixed at 10 Å**, not length-dependent);
  `x = mean_plddt * mean_ptm`; `pDockQ2 = 1.31 / (1 + exp(-0.075*(x - 84.733))) + 0.005` —
  [ipsae.py L672-700](https://raw.githubusercontent.com/DunbrackLab/IPSAE/main/ipsae.py).
  The paper's form is `X_i = <1/(1 + (PAE_int/d0)^2)> × <pLDDT>_int` with optimised `d0 = 10 Å`, then a
  logistic fitted against ground-truth DockQ on the AlphaFold-Multimer benchmark; it is computed
  **per chain pair** and aggregated — [search summary of btad424](https://academic.oup.com/bioinformatics/article/39/7/btad424/7219714)
- **LIS** (Kim, Hu, Comjean, Rodiger, Mohr, Perrimon — [bioRxiv 2024.02.19.580970](https://www.biorxiv.org/content/10.1101/2024.02.19.580970v1)),
  as implemented in ipSAE: over the interchain PAE block, keep entries with `PAE < 12`, map each to
  `(12 - PAE)/12`, and take the **mean over the surviving entries only** (not over all pairs); 0 if none
  survive — [ipsae.py L702-720](https://raw.githubusercontent.com/DunbrackLab/IPSAE/main/ipsae.py)
- **DockQ** (Basu & Wallner, *PLoS ONE* 11:e0161879, 2016; v2 Mirabello & Wallner,
  *Bioinformatics* 40(10):btae586, 2024 — [Bioinformatics](https://academic.oup.com/bioinformatics/article/40/10/btae586/7796530)).
  The DockQ README defines its components as `fnat`: "Fraction of retrieved native contacts (same as
  Recall or TPR)"; `iRMSD`: "RMSD of interfacial residues"; `LRMSD`: "Ligand RMSD"; with CAPRI-style
  bands `<0.23` incorrect, `0.23-0.49` acceptable, `0.49-0.80` medium, `>=0.80` high —
  [DockQ README](https://raw.githubusercontent.com/bjornwallner/DockQ/master/README.md)
- DockQ's only subsetting controls are **chain-level**, not residue-level: `--mapping` restricts to
  specific chain pairs/interfaces (e.g. `--mapping :WX`), plus `--capri_peptide` and `--small_molecule`
  modes — [DockQ README](https://raw.githubusercontent.com/bjornwallner/DockQ/master/README.md)
- The standard DockQ combination is `DockQ = (Fnat + 1/(1+(LRMSD/8.5)^2) + 1/(1+(iRMSD/1.5)^2)) / 3`
  (standard published definition; **not re-verified from a primary source in this session** — the DockQ
  README does not print the constants and points to [btae586](https://academic.oup.com/bioinformatics/article/40/10/btae586/7796530))
- All of pDockQ, pDockQ2 and LIS are computed by `ipsae.py` and emitted in the same per-chain-pair table
  as ipSAE, with `asym` and `max` rows — [IPSAE README L30-41, L69](https://raw.githubusercontent.com/DunbrackLab/IPSAE/main/README.md)

### Inferences

- **Restrictability ranking (easiest → hardest):**
  1. **LIS** — trivially restrictable. The mask is `(chains[:,None]==chain1) & (chains[None,:]==chain2)`;
     AND with `epitope_mask[None,:]`. The score is a mean over survivors with no length-dependent
     normalisation at all, so the restricted value stays on the same 0-1 scale and is directly comparable
     to the unrestricted one. For a quick, interpretable "LIS@specified site" this is the lowest-effort option in
     the whole set.
  2. **pDockQ2** — restrictable, and "naturally restrictable" is correct: it is already per-interface-pair,
     `d0` is a fixed 10 Å (so no length renormalisation artefact), and both factors (`mean_ptm` over
     interface PAEs, `mean_plddt` over interface residues) are plain averages over an index set. Restrict
     by taking `pae_list` over designed chain×specified site pairs only and `mean_plddt` over the specified-site residues in
     contact. Caveat: the logistic constants (1.31, -0.075, 84.733, 0.005) were fitted on *whole*
     interfaces, so a restricted pDockQ2 is no longer calibrated to DockQ and should be used as a rank
     statistic, not as a probability.
  3. **pDockQ** — restrictable but ill-advised. `x = mean_plddt * log10(npairs)` makes the score grow with
     interface size; restricting to a specified site shrinks `npairs` and drags `x` down mechanically,
     confounding "wrong site" with "small site". It also uses no PAE at all, which is why pDockQ2 exists.
  4. **DockQ** — not restrictable in a design-relevant way. It needs a reference complex; `Fnat` is
     defined against the *native* contact set. For de novo or redesigned designed chains against a specified
     specified site with no experimental complex, DockQ is simply unavailable. Its `--mapping` is chain-level.
- The right way to borrow from DockQ is to borrow **Fnat's shape, not DockQ itself**: define
  `Fnat_epitope = |predicted contacts ∩ (binder × epitope)| / |binder × epitope contacts possible|`, or
  more usefully the specified site-recall form in Q5.
- Because LIS, pDockQ2 and i_pDAE all restrict cleanly while ipSAE and ipTM carry normalisation artefacts,
  a pragmatic recommendation is: use **LIS@specified site** or **i_pDAE@specified site** as the restricted confidence
  term, and keep unrestricted ipSAE as a sanity check that *some* confident interface exists at all.

### Gaps

- I did not verify the DockQ scaling constants (8.5 Å, 1.5 Å) from a primary source in this session.
- pDockQ2's paper-level aggregation rule across multiple chain pairs (mean? min?) was taken from a search
  summary, not a primary read.

---

## Q5. Specified site-specific / site-specific scoring in antibody-antigen prediction benchmarks

### Takeaway

Benchmarks do report site-agreement measures, but they are **structure-comparison** measures (DockQ,
specified site shift, antibody displacement) or **classification** measures (specified site precision/recall/F1 over
antigen surface residues) — not confidence metrics. No benchmark in 2024-2026 that I found defines an
site-restricted *confidence* score. The closest named quantities are "specified site recall / EpiRec",
"specified site shift", and the interface-residue precision/recall/MCC family from AsEP.

### Cited Findings

- A 2026 AF3 antibody benchmark explicitly separates confidence from site correctness: 3401
  experimentally validated SAbDab complexes plus 23798 negative controls; confidence assessed via
  "Predicted Aligned Error and Interface Predicted Template Modeling score"; "maximum recall of 53% at 100
  inference seeds"; "an innate false positive rate of approximately 3%"; AF3 "hallucinate[s] plausible
  binding interfaces across the surface of decoy targets while avoiding disordered regions"; site
  correctness assessed with **"DockQ, specified site shift, and antibody displacement"**; and "approximately 34%
  of false negatives retained correct specified site location despite poor structural alignment" —
  [bioRxiv 2026.07.30.741792](https://www.biorxiv.org/content/10.64898/2026.07.30.741792v1);
  [PubMed 42620036](https://pubmed.ncbi.nlm.nih.gov/42620036/)
- **AsEP** (NeurIPS 2024 Datasets & Benchmarks) curates 1723 non-redundant antibody-antigen complexes and
  "the task is formulated as binary classification over antigen surface residues to predict epitopes",
  scored with precision/recall (and MCC / AUC-PR) — [arXiv:2407.18184](https://arxiv.org/html/2407.18184);
  [NeurIPS proceedings PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/15add6732964d5b1f0954058bf3ccc88-Paper-Datasets_and_Benchmarks_Track.pdf)
- Illustrative precision/recall trade-off on AsEP: WALLE at 0.926 recall / 0.114 precision vs EpiFormer at
  0.720 recall / 0.363 precision — [search summary of EpiFormer, arXiv:2606.04154](https://arxiv.org/pdf/2606.04154)
  (**numbers taken from a search-result summary, not verified against the primary table**)
- **CHIMERA-Bench: "A Benchmark Dataset for Specified site-Specific Antibody Design"** defines
  **Specified site Recall (EpiRec)** as "the fraction of true specified-site residues that the design contacts" —
  [arXiv:2603.13431](https://arxiv.org/pdf/2603.13431) (**definition from a search-result summary; the
  primary PDF was not read**)
- DockQ's own documentation names `fnat` as "Fraction of retrieved native contacts (same as Recall or
  TPR)" — i.e. the canonical interface-contact recall —
  [DockQ README](https://raw.githubusercontent.com/bjornwallner/DockQ/master/README.md)
- A 2025 AlphaFold/TCR-antibody benchmarking review exists and is the right place to look for
  consolidated site-agreement definitions — [PMC13370930](https://pmc.ncbi.nlm.nih.gov/articles/PMC13370930/)
  (**listed as a lead; not read in this session**)
- In sequence optimization practice, the site is specified on the *input* side rather than scored on the output
  side. BindCraft1: `af_model.prep_inputs(..., hotspot=target_hotspot_residues, ...)` —
  [BindCraft colabdesign_utils.py L36](https://raw.githubusercontent.com/martinpacesa/BindCraft/main/functions/colabdesign_utils.py).
  BindCraft2 README: `"hotspots": "54,56,66-70"` "using residue numbers from your structure"; chain-prefixed
  anchor residues `"A54,B12-16"`; `coldspots` to "Keep a target region free"; and `--forced-targeting` /
  `"forced_targeting": true` for a "**Focused specified site** — concentrate binding on named anchor residues", with
  "the method and acceptance criteria" documented under "targeting options" —
  [BindCraft2 README L65, L134-155](https://raw.githubusercontent.com/PacesaLab/BindCraft2/main/README.md)
- BindCraft2's forced-targeting mechanism is structural, not metric-based: `bindcraft/epitope_targeting.py`
  defines `epitope_residues(atoms, atom_mask, hotspot_residues, epitope_cutoff)` as all residues whose
  minimum atom-atom distance to any anchor residue atom is `<= epitope_cutoff`, then `lysinate_target()` mutates
  every surface-exposed residue **outside** that specified site to lysine
  (`lysinated_residues = surface_exposed_residues(...) & ~epitope_residue_mask & (sequence != K)`), raising
  `ValueError('Epitope-focused design needs target hotspots to define the region kept at wild type')` if no
  anchor residues are set, and logging
  `f'forced targeting epitope={...} lysinated={...} residues={...}'` —
  [BindCraft2 epitope_targeting.py L21-83](https://raw.githubusercontent.com/PacesaLab/BindCraft2/main/bindcraft/epitope_targeting.py)
- Anchor residue specification is standard across pipelines: "Anchor residues are residues that should be targeted on the
  target protein and can be defined according to residue numbering, either individually or as residue
  ranges, or left empty to let the pipeline determine an optimal binding site" —
  [BindCraft wiki](https://github.com/martinpacesa/BindCraft/wiki/De-novo-binder-design-with-BindCraft)

### Inferences

- The field's answer to "did it bind the right site?" is currently **geometry-vs-reference** (DockQ,
  specified site shift) or **set-overlap** (specified site recall / precision), never a confidence metric. The project's
  problem — a confidence score that is high only when the confident interface is at specified residues —
  is not a solved, named, published quantity. The honest framing for a write-up is: this is a gap, and the
  nearest publishable precedent is actifpTM's `pair_residue_weights` generalised from
  "predicted contacts" to "specified specified site".
- The AF3 benchmark's ~3% false-positive rate on 23798 decoys and its finding that AF3 "hallucinate[s]
  plausible binding interfaces across the surface of decoy targets" is a strong, citable independent
  confirmation of the project's permuted-specified site negative-control result. It also means the project's
  observation (control scoring ~2× the real target) is consistent with a known model pathology, not an
  artefact of the local pipeline.
- The AF3 benchmark's "~34% of false negatives retained correct specified site location despite poor RMSD" is the
  mirror-image warning: site correctness and pose accuracy are partly decoupled, so a specified site-overlap
  metric and an RMSD/DockQ metric are **not** substitutes. The project should carry both.
- BindCraft2's `coldspot` / `coldspot_repel` / forced-targeting design is informative: the authors chose to
  enforce the site through the *input* (anchor residue contact losses, coldspot repulsion, lysinating the rest of
  the surface) rather than through a site-restricted confidence score. That is evidence the metric gap is
  real and that practitioners route around it.

### Gaps

- Neither the CHIMERA-Bench nor the EpiFormer primary PDFs were read, so the EpiRec definition and the
  precision/recall figures are second-hand. These should be verified before being quoted in a report.
- No formal definition of "specified site shift" was found. A targeted search for the term returned nothing with a
  formal metric definition; the AF3 benchmark uses the term without an abstract-level definition. Likely
  defined as the distance between predicted and native specified site centroids, but **this is unverified**.
- The AF3 benchmark's cutoffs for calling a specified site "correct" were not recoverable from the abstract.

---

## Q6. Metrics that directly measure overlap between a predicted interface and a specified specified site

### Takeaway

These are set-comparison statistics over residue sets, not confidence metrics: recall (EpiRec / Fnat-style),
precision, F1 and Jaccard over {predicted target-side interface residues} vs {specified specified-site residues}.
They are cheap, require no reference complex, and the contact masks needed to compute them are already
produced by BindCraft2 and ColabFold code. They are inherently hard-thresholded, hence non-differentiable
as written, but each has an obvious soft relaxation via the distogram contact probability.

### Cited Findings

- Let `P` = set of target residues in contact with the designed chain in the prediction, and `E` = the specified
  specified site. The field's named forms are:
  - **Recall / EpiRec** = `|P ∩ E| / |E|` — "the fraction of true specified-site residues that the design
    contacts" — [arXiv:2603.13431, CHIMERA-Bench](https://arxiv.org/pdf/2603.13431) (search summary)
  - **Fnat** = "Fraction of retrieved native contacts (same as Recall or TPR)" — the contact-pair rather
    than residue version — [DockQ README](https://raw.githubusercontent.com/bjornwallner/DockQ/master/README.md)
  - **Precision** and **Recall** over antigen surface residues, as the standard AsEP reporting pair for
    specified site prediction treated as "binary classification over antigen surface residues" —
    [arXiv:2407.18184](https://arxiv.org/html/2407.18184)
- The target-side predicted interface set `P` is already computed in BindCraft2:
  `binder_target_contact_masks(binder, target, cutoff=4.0)` builds
  `contact = (pairwise_atom_distances(binder_atoms, target_atoms) <= cutoff) & binder_atom_mask[:,None] & target_atom_mask[None,:]`
  and returns `(contact.any(-1)...any(-1), contact.any(0)...any(-1))` — the designed chain-side and **target-side**
  per-residue boolean masks. `binder_assembly_contact_masks` unions this over designed chain copies —
  [BindCraft2 filters.py L101-110](https://raw.githubusercontent.com/PacesaLab/BindCraft2/main/bindcraft/filters.py)
- The specified specified site set `E` can be expanded from anchor residues by the same code used for forced targeting:
  `epitope_residues(atoms, atom_mask, hotspot_residues, epitope_cutoff)` = residues within
  `epitope_cutoff` of any anchor residue atom —
  [BindCraft2 epitope_targeting.py L25-29](https://raw.githubusercontent.com/PacesaLab/BindCraft2/main/bindcraft/epitope_targeting.py)
- BindCraft1 has an equivalent helper imported as `hotspot_residues` from `biopython_utils`, used as
  `binder_contacts = hotspot_residues(model_pdb_path)` —
  [BindCraft colabdesign_utils.py L16, L204](https://raw.githubusercontent.com/martinpacesa/BindCraft/main/functions/colabdesign_utils.py)
- Contact-probability (soft) versions of `P` are available from the distogram:
  `get_contact_map(outputs, dist=8.0) = (jax.nn.softmax(dist_logits) * (dist_bins < dist)).sum(-1)` —
  [ColabDesign loss.py L223-230](https://raw.githubusercontent.com/sokrypton/ColabDesign/main/colabdesign/af/loss.py)
  and identically in [ColabFold extra_ptm.py L22-27](https://raw.githubusercontent.com/sokrypton/ColabFold/main/colabfold/alphafold/extra_ptm.py)
- The AF3 antibody benchmark pairs site-overlap reasoning with structural measures ("DockQ, specified site shift,
  and antibody displacement") rather than using overlap alone —
  [bioRxiv 2026.07.30.741792](https://www.biorxiv.org/content/10.64898/2026.07.30.741792v1)

### Inferences

- Definitions the project can adopt directly, with the citations above as precedent:
  - `EpiRecall = |P ∩ E| / |E|`
  - `EpiPrecision = |P ∩ E| / |P|` (penalises designed chains that also smear over off-specified site surface)
  - `EpiF1 = 2·EpiPrecision·EpiRecall / (EpiPrecision + EpiRecall)`
  - `EpiJaccard = |P ∩ E| / |P ∪ E|`
  - `EpiCoverageFraction` — the contact-weighted version: `(Σ_{j∈E} c_j) / (Σ_{j∈target} c_j)` where
    `c_j = Σ_i cmap[i,j]` over designed-chain residues i. This is the natural soft analogue and is differentiable.
- **A clean two-factor filter is the right shape for the project's problem**: one factor says "the
  interface is confident" (ipSAE / i_pDAE / LIS), the other says "the interface is here"
  (EpiJaccard or EpiCoverageFraction), and the product or the min of the two is the selection score. This
  is strictly more informative than any single restricted metric, because it separates the two failure
  modes the project is seeing (confident-but-wrong-site vs right-site-but-unconfident). It also makes the
  permuted-specified site control a natural sanity check: a permuted specified site should drive the overlap factor to
  ~chance while leaving the confidence factor unchanged.
- The hard-threshold versions are post-hoc-only. The soft versions (`cmap`-weighted coverage, or
  `Σ_{i,j∈E} cmap[i,j]` as a "soft specified site contact count") are smooth in the distogram logits and can be
  used as a gradient objective — this is exactly what BindCraft's `i_con` with anchor residues already does
  (see Q7).

### Gaps

- I found no paper that reports a **Jaccard** index specifically for predicted-interface vs specified-specified site
  agreement under that name; precision/recall/F1 dominate. The Jaccard form is a reasonable construction
  but should be presented as the project's own, not as a cited standard.
- No standard cutoff convention emerged for defining `P`. The codebases read here use 4.0 Å atom-atom
  (BindCraft2 `Interface_Residues`), 8.0 Å CA-CA (BindCraft2 i_pDAE), and 8.0 Å CB-CB (pDockQ, actifpTM).
  Any specified site-overlap metric must state its cutoff; results will not be comparable across the three.

---

## Q7. Differentiability — which of these work as a gradient objective

### Takeaway

The hard `PAE < cutoff` mask in ipSAE/LIS and the hard distance mask in i_pDAE are piecewise-constant in
the model outputs and give zero gradient through the mask, so none of the published *scores* is a usable
objective as written. But the **site-restricted differentiable objectives already exist**: ColabDesign's
`i_pae` with the `hotspot` option is a site-restricted interface-PAE loss, and BindCraft2's
`interface_contacts` and `target_plddt` losses are anchor residue-masked. These are the production-grade answer
to the differentiability question.

### Cited Findings

- **ColabDesign already implements a site-restricted differentiable interface-PAE loss.** In
  `_loss_binder`:
  ```python
  binder_id = zeros.at[-bL:].set(mask[-bL:])
  if "hotspot" in opt:
    target_id = zeros.at[opt["hotspot"]].set(mask[opt["hotspot"]])
    i_con_loss = get_con_loss(inputs, outputs, opt["i_con"], mask_1d=target_id, mask_1b=binder_id)
  else:
    target_id = zeros.at[:tL].set(mask[:tL])
    i_con_loss = get_con_loss(inputs, outputs, opt["i_con"], mask_1d=binder_id, mask_1b=target_id)
  ...
  "i_pae":   get_pae_loss(outputs, mask_1d=binder_id, mask_1b=target_id),
  ```
  — when `hotspot` is supplied, `target_id` **is** the anchor residue mask, so `i_pae` becomes a mean of
  `PAE/31` over (designed-chain residue × anchor residue) pairs only —
  [ColabDesign loss.py L35-58](https://raw.githubusercontent.com/sokrypton/ColabDesign/main/colabdesign/af/loss.py)
- The masked mean is smooth: `mask_loss(x, mask) = (x*mask).sum() / (1e-8 + mask.sum())`, with an optional
  straight-through variant `jax.lax.stop_gradient(x.mean() - x_masked) + x_masked` —
  [ColabDesign loss.py L232-240](https://raw.githubusercontent.com/sokrypton/ColabDesign/main/colabdesign/af/loss.py)
- PAE itself is a differentiable expectation over bins:
  `get_pae(outputs) = (softmax(pae_logits) * bin_centers).sum(-1)` —
  [ColabDesign loss.py L196-202](https://raw.githubusercontent.com/sokrypton/ColabDesign/main/colabdesign/af/loss.py)
- The contact loss is smooth in the distogram but uses a hard top-k selection:
  `_get_con_loss` returns `jnp.where(binary, con_loss_bin_ent, con_loss_cat_ent)` with
  `con_loss_bin_ent = -jnp.log((bins*px + 1e-8).sum(-1))` and
  `con_loss_cat_ent = -(px_ * jax.nn.log_softmax(dgram)).sum(-1)`; the aggregation is
  `min_k(x, k, mask)` using `jnp.sort` with a NaN-masked array, applied twice —
  `p = min_k(p, con_opt["num"], m); return min_k(p, con_opt["num_pos"], mask_1d)` —
  [ColabDesign loss.py L261-308](https://raw.githubusercontent.com/sokrypton/ColabDesign/main/colabdesign/af/loss.py)
- `get_contact_map(outputs, dist=8.0) = (jax.nn.softmax(dist_logits) * (dist_bins < dist)).sum(-1)` — a
  **differentiable soft contact probability**; the `dist_bins < dist` comparison is on fixed bin centres,
  not on model output, so it does not break the gradient —
  [ColabDesign loss.py L223-230](https://raw.githubusercontent.com/sokrypton/ColabDesign/main/colabdesign/af/loss.py)
- **BindCraft2 has anchor residue-masked differentiable losses.** `target_plddt_loss` uses
  `hotspot_mask = has_residue_flag(..., ResidueFlags.HOTSPOT)` and
  `_masked_mean(1 - plddt[target_slice], jnp.where(hotspot_mask.any(), hotspot_mask, real_residue_mask(...)))`
  — an anchor residue-restricted target-pLDDT objective with a graceful fallback when no anchor residues are set —
  [BindCraft2 loss.py L193-197](https://raw.githubusercontent.com/PacesaLab/BindCraft2/main/bindcraft/loss.py)
- `interface_contacts_loss` is explicitly site-restricted when anchor residues exist:
  `hotspot_mask = expand_chain_residue_mask(residue_count, chain_slices[target], has_residue_flag(..., ResidueFlags.HOTSPOT))`;
  `hotspot_contacts = mean_selected_contact_loss(pair_loss, contacts_per_residue, contact_residue_count, hotspot_mask, hotspot_mask[:,None] & binder_mask[None,:])`;
  `return jnp.where(hotspot_mask.any(), hotspot_contacts, surface_contacts)` —
  [BindCraft2 loss.py L539-551](https://raw.githubusercontent.com/PacesaLab/BindCraft2/main/bindcraft/loss.py)
- The complementary negative-site objective exists too: `coldspot_repel_loss` builds a
  `coldspot_mask` from `ResidueFlags.COLDSPOT` and calls
  `nonbinding_residue_repulsion_loss(distogram, coldspot_mask, binder_mask, cutoff, contacts_per_residue)`
  — [BindCraft2 loss.py L505-514](https://raw.githubusercontent.com/PacesaLab/BindCraft2/main/bindcraft/loss.py)
- BindCraft2's top-k selection explicitly stops the gradient through the *ranking* while keeping it through
  the *values*: `contact_ranking = jax.lax.stop_gradient(jnp.argsort(jnp.where(mask, values, jnp.inf)))`,
  then `jnp.where(selected, ranked_values, 0).sum(-1) / (selected.sum(-1) + eps)` —
  [BindCraft2 loss.py L417-421](https://raw.githubusercontent.com/PacesaLab/BindCraft2/main/bindcraft/loss.py)
- BindCraft2 also wraps ipTM directly as a loss: `@loss('iptm_loss') ... return 1 - metrics['iptm']`, and
  `metrics['iptm']` is produced by
  `confidence.predicted_tm_score(pae_head['logits'], pae_head['breaks'], residue_weights=seq_mask, asym_id=interface_asym_id, use_jnp=True, return_per_alignment=True)`
  — so ipTM is differentiable in-graph, and `seq_mask` is the restriction hook —
  [BindCraft2 loss.py L386-388](https://raw.githubusercontent.com/PacesaLab/BindCraft2/main/bindcraft/loss.py),
  [BindCraft2 af2.py L161](https://raw.githubusercontent.com/PacesaLab/BindCraft2/main/bindcraft/af2.py)
- BindCraft2 uses soft sigmoid relaxations for geometric predicates elsewhere, e.g.
  `jax.nn.sigmoid((radius - pairwise_atom_distances(...)) / contact_temperature)` for neighbour counting
  and `soft_maximum(values, temperature)` — demonstrating the house style for smoothing a hard cutoff —
  [BindCraft2 loss.py L274-280, L227-228](https://raw.githubusercontent.com/PacesaLab/BindCraft2/main/bindcraft/loss.py)
- actifpTM's probability-weighted path is written in `jax.numpy` and called with `use_jnp=True`, and its
  `pair_residue_weights` come from the **soft** `get_contact_map` — so that path is differentiable
  end-to-end; the default `cmap >= 0.6` binary path is not —
  [ColabFold extra_ptm.py L122-159 vs L162-203](https://raw.githubusercontent.com/sokrypton/ColabFold/main/colabfold/alphafold/extra_ptm.py)
- ipSAE's implementation is pure NumPy with `np.vectorize` and a Python `for i in range(numres)` loop,
  reading a JSON/npz PAE file from disk — there is no gradient path of any kind —
  [ipsae.py L32-34, L111-113, L740-800](https://raw.githubusercontent.com/DunbrackLab/IPSAE/main/ipsae.py)

### Inferences

- **Summary table (a) restrictable to specified target residues, (b) implementation exists, (c) differentiable:**

  | Metric | (a) Restrictable to specified residues | (b) Implementation | (c) Differentiable |
  |---|---|---|---|
  | ipSAE (`d0res`) | Not in the released code; one-line change to `valid_pairs_matrix`; d0 shrinks so scale shifts | `ipsae.py` v4, PyPI `ipsae`, Neurosnap, vendored in ColabFold | No — NumPy, hard `PAE < cutoff` mask, double `argmax` |
  | ipSAE `d0chn` / `d0dom` | Same as above; `d0chn` is the most scale-stable of the three under restriction | same | No |
  | ipTM | Yes, via `residue_weights`/`seq_mask`; no published specified site variant | AF2/AF3/ColabFold/ColabDesign/BindCraft2 | Yes (in-graph, `use_jnp=True`) |
  | pTM | Yes, same mechanism (`get_per_chain_ptm` already slices) | ColabFold `extra_ptm.py` | Yes |
  | actifpTM | **Yes — it is literally a residue-subset-restricted ipTM**, but the subset is auto-chosen from `cmap >= 0.6`, not user-specified | ColabFold `extra_ptm.py`, `--calc_extra_ptm`; AlphaPulldown2 | Probability path yes (jnp + soft cmap); binary path no |
  | i_pAE (interface PAE/31) | Yes — mask is `mask_1d[:,None]*mask_1b[None,:]`, and `mask_1b` can be the specified site | ColabDesign `get_pae_loss`, BindCraft1/2 `chain_pair_pae_loss` | **Yes** |
  | i_pAE with `hotspot` | **Already site-restricted today** | ColabDesign `_loss_binder` | **Yes** |
  | i_pDAE | Yes — `contact & epitope_mask`; best geometric grounding of the set | BindCraft2 `filters.py`, plus a per-residue track | No as written (hard 8 Å mask, `.max()`); soft version straightforward |
  | LIS | Yes, trivially; no length renormalisation so scale is preserved | `ipsae.py`; original Kim et al. code | No (hard `PAE < 12`); soft version straightforward |
  | pDockQ | In principle; `log10(npairs)` confounds site with size | `ipsae.py`, Elofsson lab | No (fitted sigmoid on hard contact counts) |
  | pDockQ2 | Yes — per-chain-pair by construction, fixed `d0 = 10 Å`, both factors are plain means | `ipsae.py`, ColabFold, Elofsson lab | No (hard 8 Å contacts); the PAE factor alone is |
  | DockQ / Fnat | No — needs a reference complex; `--mapping` is chain-level only | `bjornwallner/DockQ` | No |
  | EpiRecall / EpiPrecision / EpiF1 / EpiJaccard | Yes by definition | Masks exist in BindCraft1/2; metric itself must be written | No as written; soft `cmap`-weighted coverage is |
  | Anchor residue `i_con` contact loss | **Already site-restricted today** | ColabDesign `get_con_loss(mask_1d=hotspot)`; BindCraft2 `interface_contacts_loss` | Yes, a.e. (`jnp.sort`/`argsort` top-k is piecewise differentiable; BindCraft2 stops gradient through the ranking) |

- **The non-smooth `PAE < cutoff` problem has a standard fix in this codebase family**: replace the
  indicator `1[PAE_ij < c]` with `sigmoid((c - PAE_ij)/T)`, exactly as BindCraft2 does for distance
  predicates (`jax.nn.sigmoid((radius - d)/contact_temperature)`). Doing this to ipSAE gives a "soft ipSAE":
  `w_ij = sigmoid((c - PAE_ij)/T) · epitope_j · binder_i`;
  `n0res(i) = Σ_j w_ij`; `d0(i) = max(1, 1.24·(max(26, n0res(i)) - 15)^{1/3} - 1.8)`;
  `score_i = Σ_j w_ij · 1/(1 + (PAE_ij/d0(i))²) / Σ_j w_ij`; and replace the outer `max_i` with a
  temperature-controlled soft-max or a mean over specified site-contacting designed-chain residues. Every term is smooth;
  the only remaining discontinuity (the `max(26, ·)` and `max(1, ·)` clamps) is piecewise-linear and
  gradient-safe away from the kink.
- **Practical recommendation, strongest first:**
  1. For a *post-hoc filter* that fixes the reported failure: compute `i_pDAE@epitope` (or `LIS@epitope`)
     **and** `EpiJaccard`, and gate on both. `LIS@epitope` is the cheapest to implement and the only one
     whose absolute scale survives restriction unchanged.
  2. For a *gradient objective*: use ColabDesign/BindCraft2's existing anchor residue-restricted `i_pae` and
     `i_con` rather than inventing a differentiable ipSAE. They are production-tested, and the anchor residue path
     through `target_id` makes `i_pae` a site-restricted PAE loss with no code change.
  3. If a *restricted ipSAE-like* score is specifically wanted for continuity with existing numbers, build
     it on actifpTM's `pair_residue_weights` interface, because that keeps `d0` tied to the full sequence
     length and therefore keeps the restricted and unrestricted values on one scale.
- One caution on calibration: every metric that renormalises `d0` by the number of surviving partners
  (ipSAE's `d0res`/`d0dom`, i_pDAE's `tm_score_distance_scale`) becomes *harsher* when restricted, because
  a smaller `d0` demands lower PAE for the same score. Thresholds learned on unrestricted scores (e.g. a
  0.795 "true complex" reference) will not transfer. Re-derive the reference by computing the restricted
  metric on the true complex with the same site definition.

### Gaps

- I did not find any published work that defines or evaluates a *differentiable, site-restricted
  interface confidence* score under a name. The components all exist; the composite does not appear in the
  literature I could reach.
- I did not read BindCraft2's `nonbinding_residue_repulsion_loss` or `mean_selected_contact_loss` bodies,
  so the exact smoothing used in the coldspot and anchor residue contact losses is unverified beyond the
  `best_contact_mean` helper quoted above.
- The `ipsae` PyPI package and the Neurosnap reimplementation may expose options the reference script does
  not; I checked only the reference `ipsae.py`. Worth a direct look at `pip show -f ipsae` before
  concluding that no packaged residue-subset option exists anywhere.
