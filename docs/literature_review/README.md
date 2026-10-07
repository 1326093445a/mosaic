# Literature review: reaching a *specified* arrangement with predictor gradients

Reviewed 2026-10-05. Entry point for this notes folder.

## Why this review exists, in proof-of-concept terms

This project is a **proof of concept for gradients**, not an optimization campaign
(see [P17_JN1.md §19](../P17_JN1.md)). The question is whether a frozen
structure predictor's gradient with respect to designed-chain sequence can restore that
predictor's own two readouts — interface confidence and target-aligned designed chain
pose — from a degraded starting point. No biological claim is attached, and
affinity, expression, developability and wet-lab outcomes are **not** the
yardstick here.

Measured state before this review (§§20–22):

| Readout | Degraded start | Best recovered | Reference |
|---|---:|---:|---:|
| Interface confidence (mean min-directional ipSAE) | 0.00 | **0.77** | 0.795–0.807 |
| Target-aligned designed-chain pose RMSD | 48 Å | **14 Å** | 2.6 Å |

Confidence recovers; pose does not. Three controls then qualified the
confidence half: a permuted-site decoy scored ~2× the real target at matched
compute (§21.2); removing the pose term entirely barely changed pose (§21.3);
and on a control where the answer provably sat 5 edits away inside the budget,
recovery reached 58% on confidence and 19% on pose (§22.1).

**So the question put to the literature was a controllability question, not a
biology question:** by what mechanism does any published pipeline make a
gradient-guided search converge on a *specified* geometric arrangement rather
than any plausible one? "Specified site" below is simply the name the field gives to
that specification.

## The five findings that bear on the proof of concept

**1. Site specification lives in four distinct layers, and we occupy one.**
Conditioning (an input the model was *trained* to respond to), gradient-time
loss, in-trajectory rejection, and selection filtering. We have a gradient-time
loss, and we rank on a selection metric that is site-agnostic. **No surveyed
pipeline uses that combination.** The two closest analogues keep a gradient
term as weak as ours and put the decisive enforcement in selection, as hard
booleans checked *before* any confidence metric.

**2. Interface confidence measures the self-consistency of a packing, not the
nativeness of a location.** This is the most proof-of-concept-relevant
conclusion in the whole review, because it says our retention metric cannot in
principle carry the pose claim. ipSAE's pair set is
`outer(chains==c1, chains==c2) & (PAE < cutoff)` — it never references the
intended site, so no reduction over it can be made site-aware. We had already
applied both obvious hardenings (mean rather than `max_i`, min rather than max
over directions, both stricter than published ipSAE) and they changed nothing,
which is now explained rather than puzzling. **The decisive object is the mask.**

**3. "Stronger pose weight" is answered no, with measurements.** Three
independent lines: a 10×-dominant RMSD term needed ~5× *more* optimisation
steps with low RMSD "not guaranteed"; weight-based guidance achieved 0%
constraint satisfaction across ~100k samples where a dual-variable method
reached 100%; and Rosetta ramps coordinate restraints *down*, not up. What
these pipelines schedule is the sequence *parameterisation*, not geometric
weights. A larger edit budget is also the wrong lever — and should be expected
to make off-site convergence *more* likely absent a site gate.

**4. Our pose term has no published precedent.** No published work backprops an
inter-chain, align-on-target/measure-on-designed chain reference-pose RMSD through a
predictor to optimise sequence. Every placement-specifying pipeline works in
**distogram or contact space** instead. The mechanistic reason is gradient
density: a distogram term feeds every pair directly from a network output head,
whereas a global pose RMSD yields one scalar per step and the Kabsch step
removes precisely the rigid-body component that constitutes most of the error.
A hinge on that scalar is worse — `relu(rmsd − tol)` has gradient magnitude
exactly 1 or 0 and carries no distance information at all.

**5. Our two negative results are the novel contributions.** The failure mode
has no accepted name; no design paper runs a compute-matched permuted-site
control; and the weight-0 pose ablation has no published counterpart. Both of
our negative results sit in genuine holes in the literature.

## What the literature cannot do for us

Not gaps in the search — gaps in the field, and they bound what a comparison to
published work can establish:

- **No pipeline publishes a site-recovery rate.** There is no "X% of designs
  reach the intended site" figure for any of them.
- **Fewer than ~10 designed complexes have been solved experimentally**, across
  all flagship papers, all drawn post hoc from confirmed designed chains and all
  reported on-site. That is survivorship, so neither a rate nor a counterexample
  exists.
- **No published false-positive rate or threshold for ipSAE**, and none for the
  distribution of ipSAE over *designed* rather than natural complexes. Our
  0.77-vs-0.795 comparison therefore sits outside what that metric's own paper
  validated.
- **No published off-site rate before filtering**, from any pipeline, though
  the existence of dedicated site filters and abort counters implies it is not
  small.
- **No benchmarked gain from held-out-predictor re-prediction**, and no
  head-to-head ablation of RMSD versus distogram as a design objective.

Consequently: published success rates are **not** a yardstick for this project.
They measure complete pipelines including redesign, filtering and wet-lab
triage, under budget accounting that is not commensurable with a score-call
count. What transfers is mechanism.

## Implementable items, mapped to this repository

Ordered by how much each changes the proof-of-concept readout per unit of work.

| # | Change | Where | Grounding |
|---|---|---|---|
| 1 | Site gate in **retention**, as booleans evaluated before any confidence metric | `src/mosaic/search.py` retention; `examples/p17_confidence_search.py` scoring | Germinal's initial filter contains **no confidence metric at all**: `binder_near_hotspot`, `cdr3_hotspot_contacts > 0`, `percent_interface_cdr > 0.5`, at 5.3 Å / 6.0 Å / ≥3 contacts. It *computes* ipsae and declines to filter on it. |
| 2 | Flip the contact term's reduction to per-**site** residue | `BinderTargetContact`, `src/mosaic/losses/structure_prediction.py:316-339` | Ours reduces `.mean()` over designed-chain residues — ColabDesign's *no-anchor residue* semantics. Every working restraint reduces per site residue; ColabDesign's `hotspot` option literally swaps the reduction axis of the same loss. Our 8 Å cutoff is already in the right band. |
| 3 | Site-masked confidence metric for ranking | new metric beside `ipsae_d0res_asym_max` | BindCraft2's `i_pDAE` replaces the PAE mask with a geometric `CA-CA ≤ 8 Å` boolean matrix, so `contact & site_mask[None,:]` is a one-line restriction. **Calibration warning:** every `d0`-renormalising variant hardens under restriction, so the 0.795 reference will not transfer and must be re-derived. |
| 4 | Off-site repulsion at gradient time | composite loss | We reward site contacts and charge nothing for an interface elsewhere. Germinal has the contact face-side form (`L_CDR − λ·L_framework`); the target-side analogue is implemented by nobody at gradient time, only as a BindCraft2 filter. |
| 5 | In-trajectory abort | search loop | RFantibody discards a trajectory when the closest specified residue is >10 Å from any designed loop, and aborts the job after 20 failures. Off by default; the existence of a failure counter implies the authors saw it fire often. |
| 6 | Input-space site restriction | featurisation | BindCraft2 lysinates exposed surface outside a 10 Å shell; target-trimming to site fragments is reported to help. Both make an off-site pose unattractive or geometrically unavailable **without touching a loss weight**, which suits a sequence-gradient pipeline since it acts where we are already free to act. |

Scale reference: a 14–27 Å displacement fails **every** published site-contact
threshold — 4 Å heavy-atom, 5 Å atom / 15 Å token, 6 Å, 8 Å Cβ, 10 Å Cβ. Even
the loosest criterion in the literature rejects these candidates.

Two things not to copy: BindCraft 1's `Hotspot_RMSD` compares a design to its
own hallucinated trajectory, unaligned, so an off-site trajectory is *rewarded*
for being reproduced faithfully; and a fast proxy was measured to diverge from
the full model by up to 7 Å in site distance, so any site metric must be
recomputed on the expensive predictor at selection time — our own §6 proxy gap,
independently confirmed.

## Files in this folder

| File | Covers |
|---|---|
| `diffusion_and_inverse_folding_epitope_specification.md` | RFdiffusion, RFantibody, AlphaProteo, EvoBind, Boltz-2/BoltzGen: conditioning vs loss vs filter, with training definitions and parameter values |
| `hallucination_pipeline_loss_composition.md` | Germinal, BindCraft 1/2, EasyNano, BindEnergyCraft, ColabDesign: exact loss terms, weights, filter thresholds, file paths and line numbers |
| `site_restricted_confidence_metrics.md` | ipSAE, actifpTM, i_pDAE, LIS, pDockQ/pDockQ2, DockQ: definitions, restrictability, differentiability |
| `pose_rmsd_objectives_and_restraint_scheduling.md` | Prior art for RMSD-through-predictor, FAPE saturation, Kabsch/SVD gradients, penalty continuation and annealing evidence |
| `wrong_epitope_failure_mode_and_controls.md` | Whether the failure mode is documented, what negative controls the field runs, reported detection and mitigation |
| `distance_restraint_conditioning.md` | Distance/pocket conditioning as a model input: what the predictor was trained to respond to, and parameter values |
| `can_mutations_reorient_binding.md` | Whether sequence substitutions alone can reorient an existing arrangement, and the evidence behind it |
| `continuous_relaxation_and_the_discretization_gap.md` | **Added 2026-10-07.** Relaxed vs discrete sequence optimization: the measured efficiency win, the cost of rounding back, initialization when a parent exists, and position-vs-identity proposals (RSO, RSS, Fast SeqProp, BindCraft staging, CoSiNE) |
| `../site_targeting_synthesis.md` | Full synthesis, ~6,600 words, 48 inline citations |

## Caveats to carry forward

Each note file distinguishes cited findings from the researcher's own
inferences, and flags claims that came from search summaries rather than
verified full text. Specifically unverified and worth reading before any
novelty claim: **Nat Commun 2025** (s41467-025-67361-9), reportedly describing
"cases where the highest-affinity candidates bind outside the intended
specified site", and the **Nature 2025 RFantibody** paper, reportedly describing a VHH
with a framework-mediated binding mode classified as a design failure. Both
paywalled; the RFantibody preprint contains no such case.

Computational-only, no wet-lab validation: EasyNano, BindEnergyCraft, and the
target-trimming result. Germinal has experimental binding and alanine-scanning
data but **no experimental structures** — it verifies site targeting with
predictions from the same model family that designed the candidates, which is a
circularity worth naming when citing it as precedent.

One unresolved tension between threads: whether `actifpTM` escapes the
`d0`-renormalisation hardening (one thread reports its `d0` uses full chain
length and so is scale-stable; the synthesis reports `LIS@epitope` as the only
scale-stable option). Check before choosing a metric.
