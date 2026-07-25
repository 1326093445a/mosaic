# Gradient Guidance: Empirical Assessment Notes (AlphaSeq)

Companion to `docs/guidance_design_notes.md` (coordinate-gradient controller
design) and `docs/guidance_search_summary.md` (search-object/selection
policy). Those two documents establish the design and explicitly leave one
question open: **how do we know the gradient guidance is actually
working** — not fighting the model's own prior, and actually predictive of
real binding. This note is a session log of trying to answer that, grounded
in real code (line references current as of this session, may drift), real
literature (checked in depth, not just abstracts), a real dataset, and (as
of §9) real GPU runs with a real quantitative result. It still ends without
a conclusion — read §10 (What this does and doesn't tell us) before
assuming anything here is settled; §9e's result is real but is one point on
several axes that remain untested.

---

## 1. The core question

> What gradient combination, in what manner, controls the model so it does
> not do out-of-distribution work and unrealistic design?

This splits into two genuinely separate, stacked unknowns:

1. **Mechanism**: does the in-loop guidance controller stay consistent with
   what BoltzGen's own denoiser considers plausible, or can it drag the
   trajectory off the model's learned manifold?
2. **Objective**: even with a perfect mechanism, is the thing being
   optimized (`L_bind` = ipTM-family proxy in-loop, ipSAE post-refold)
   actually predictive of real binding affinity?

Neither is resolved on paper. What follows is what's actually checkable —
in code, in the literature, and now in real data.

---

## 2. Mechanism: what's actually implemented in `guided_partial_diffusion`

(`src/mosaic/models/boltzgen.py`, `step_body` at line 1035)

### 2a. Trust-region clipping + noise-dependent decay — REAL, implemented

```python
# boltzgen.py:698-716
def _clip_rms(delta, tau, atom_partial_mask, eps=1e-8):
    ...
    scale = jnp.minimum(1.0, tau / rms)
    return delta * scale
```
```python
# boltzgen.py:1114-1116
delta = lambda_fn(t_hat) * g_total
delta = _clip_rms(delta, tau_fn(t_hat), atom_partial_mask)
x0_guided = x0_hat - delta
```
`lambda_fn`/`tau_fn` default to `lam_max * t_hat` and
`max(tau_max * t_hat, tau_min)` (lines 731-757) — both shrink as noise
decreases, matching `guidance_design_notes.md` §9's "late diffusion:
sharply reduce guidance magnitude." Enforced unconditionally every step.
**Correction (caught in review): that describes `default_lambda_schedule`
as a standalone function, not what the production workflow actually runs.**
`boltzgen_vhh_guided.py`'s `run()` always passes an explicit
`guidance_lambda_fn = lambda_schedule_fn(cfg.lambda_schedule, cfg.lambda_max)`,
and `VHHDesignConfig.lambda_schedule` defaults to `"sigma_squared"`
(`lam_max * sigma**2`, `boltzgen_vhh_guided.py:152,299-306`), not the linear
`default_lambda_schedule` described above. `default_lambda_schedule` only
governs `lambda` when a caller passes `None` explicitly — the production
driver never does. `tau`/`alpha`/`beta` genuinely do resolve to their
`boltzgen.py` defaults (the driver never overrides those three), so only
the `lambda` claim needed fixing.

All four schedules, functions of `t_hat` (current noise level):

| schedule | role | default form | production override |
|---|---|---|---|
| `lambda` | overall guidance strength | `lam_max * t_hat` (linear) | `cfg.lambda_schedule="sigma_squared"` -> `lam_max * t_hat**2` |
| `tau` | trust-region radius | `max(tau_max * t_hat, tau_min)` | none |
| `alpha` | naturalness (`g_nat`) weight | `alpha_max / (1 + t_hat)` | none |
| `beta` | edit-budget (`g_edit`) weight | `beta_max / (1 + t_hat)` | none |

Qualitative shape: early in denoising (`t_hat` large) — wide trust region,
strong overall guidance, but `alpha`/`beta` still small, so `g_bind` mostly
drives `x0_guided`. Late (`t_hat` small) — trust region shrinks, overall
strength drops steeply (quadratic in production), but naturalness/edit
weight ramps up as the structure settles.

### 2b. PCGrad-style conflict control — REAL, implemented

```python
# boltzgen.py:679-695
def _compat_project(g_aux, g_anchor, eps=1e-8):
    dot = jnp.sum(g_aux * g_anchor, axis=(1, 2), keepdims=True)
    anchor_sq = jnp.sum(g_anchor * g_anchor, axis=(1, 2), keepdims=True) + eps
    conflict = jnp.minimum(0.0, dot) / anchor_sq
    return g_aux - conflict * g_anchor
```
Applied asymmetrically — `g_bind` is the anchor, never modified; `g_nat`/
`g_edit` get the conflicting component projected out before merging (lines
1104, 1110).

### 2c. Prior-compatibility / staying in-distribution — NOT implemented

`unguided_direction` (what BoltzGen's denoiser would have done with zero
guidance) is computed every step (`boltzgen.py:1094`) but **never
referenced** in the merge block that produces `g_bind`/`g_nat`/`g_edit`/
`g_total`/`delta`/`x0_guided` (1096-1122). It only appears afterward, inside
`if return_diagnostics:` (1142-1159) — logged, not enforced. Matches
`guidance_design_notes.md` §6.4 verbatim ("this document does not yet
specify a final prior-compatibility mechanism"), confirmed true in code.

Separately: `x0_hat = jax.lax.stop_gradient(x0_hat)` (line 1073, deliberate,
documented) blocks one literature-standard fix — see §3a.

### 2d. Two real bugs found and fixed this session (both verified, both tested)

1. **Diagnostics computed cosine/norm over the whole complex, not just
   designable atoms.** Because `_mask_center_normalize` zeros the merged
   gradient on every non-designable atom before it reaches `delta`,
   `x0_guided ≈ x0_hat` on every frozen atom — so `guided_direction`/
   `unguided_direction` agree near-exactly there regardless of what
   guidance is doing. For a small CDR inside a large complex, that frozen
   majority dilutes real disagreement toward `cos≈1.0`. Fixed:
   `per_step_metrics(diagnostics, atom_partial_mask=...)` now also computes
   `cos_guided_unguided_designable`/`norm_ratio_designable` restricted to
   designable atoms; `summarize()` uses these as the primary stat when
   present, reporting the whole-complex values as explicit
   `*_whole_complex` secondary stats instead of silently conflating the
   two. Driver (`boltzgen_vhh_guided.py`) now passes `atom_partial_mask`
   through. Tests: `tests/test_guidance_diagnostics.py` (10 new/updated
   cases, including a synthetic frozen-majority-dilutes-real-disagreement
   regression guard).
2. **Edit-budget anchor promotion can silently collapse guidance to
   naturalness-only.** When no Boltz2/OpenDDE bind signal is configured,
   `edit_loss` is promoted into the `bind` anchor slot
   (`boltzgen_vhh_guided.py:1528-1540`). If `weight_edit_budget <= 0` too,
   that anchor's gradient is all-zero — traced through `step_body`:
   `_mask_center_normalize(zeros) = zeros`, then
   `_compat_project(g_nat, g_bind=0)` returns `g_nat` **completely
   unprojected** (a zero anchor can't conflict with anything), so
   `g_total` collapses to `alpha_fn(t_hat) * g_nat` alone — naturalness
   silently becomes the sole driver despite the code's own "bind is the
   anchor" framing. Fixed: `guidance_anchor_is_empty_edit_budget(cfg)`
   predicate + a `run()`-time warning (before any model loading, same
   fail-fast pattern as the ipSAE/mutual-exclusivity guards). Tests:
   `tests/test_boltzgen_vhh_guided_config.py` (4 new cases).

Full suite: 55/55 passing after both fixes.

---

## 3. Literature check: does either cited paper actually solve §2c?

Both cited in `guidance_design_notes.md` §13. Checked in depth (full-text
via ar5iv), not just abstracts.

### 3a. Gradient Guidance for Diffusion Models: An Optimization Perspective
(arXiv:2404.14743)

Proposes "look-ahead loss" guidance:
`G_loss(x_t,t) := -β(t)·∇_{x_t}(y - g^T E[x_0|x_t])²`, where `E[x_0|x_t]`
is exactly BoltzGen's `x0_hat`. **Theorem 1** proves
`G_loss(x_t,t) ∈ Span(A)` — differentiating through the denoiser's own
Jacobian automatically confines guidance to the subspace the model learned.

**Doesn't transfer here.** Theorem 1 requires **Assumption 1**: data lies
in a *linear* subspace, `X = AU`. Their experiments use a linear score
function, not a real neural denoiser — Section 7's numerical experiments
are synthetic rewards with known analytic optima (chosen so the theory is
provable), not real-world validated objectives. Their own toy setup: ~76
min per run, 4.6s per backward-sampling pass, **no discussion of cost
reduction** (no truncated backprop, no checkpointing, no selective-step
guidance). To even attempt the heuristic version (no formal guarantee, just
the intuition) requires removing mosaic's `stop_gradient`
(`boltzgen.py:1073`) and backpropagating through the entire BoltzGen
denoiser every guided step — real added cost, no guarantee it keeps a deep
nonlinear model on-manifold. **Verdict: theoretically motivating, not
directly applicable. The formal guarantee is a toy-data artifact for our
setting.** It also does not validate its reward against anything external
— it assumes a trusted objective and studies guidance behavior under it,
which is a different problem than ours (§4).

### 3b. NOS — Protein Design with Guided Discrete Diffusion (arXiv:2305.20009)

Guides an intermediate **hidden state** via iterative Langevin updates:
`h^(i+1) ← h^i + λ₁∇_h v_θ(h^i) + λ₂∇_h KL(p(w_{t-1}|w_t) || p_h) + λ₃z^i`.
The `λ₂∇_h KL(...)` term explicitly penalizes divergence from the model's
own unguided prediction — enforced, not just proven structurally.

**More relevant, but not a solved recipe either.** Section 5.4 is real
SPR-measured wet-lab data (374 designs, 3 rounds, "99% expression rate and
40% binding rate" in the final round) — real skin in the game, unlike 3a.
Explicitly names the exact risk under discussion: *"Genetic algorithms
easily hack empirical models by leaving the support of natural
sequences... leading to poor quality solutions that nevertheless attain
high acquisition value."* Their fix — a naturalness/likelihood term
fighting the optimizer — is structurally the same role AbLang2 plays as
mosaic's `L_nat`. Cheaper than 3a (K inner steps at one hidden layer, not a
full trunk backward pass); does **not** map 1:1 onto continuous coordinate
diffusion (NOS is discrete-sequence, token-embedding hidden states) —
needs real adaptation, not a transplant.

**Checked specifically whether NOS validated its reward model before
deploying it, since that's the precedent we'd want:** it did not. No
held-out correlation/AUC number is reported for the binding discriminator
`v_θ` before the wet-lab campaign started. What they did instead: ran the
real loop and let real failures correct it (e.g. Round 2's expression drop
traced to a hyperparameter, `λ` too small, not to the reward model being
wrong). **Nobody in this literature validated a guidance objective against
ground truth before deploying it — including the most real-world-grounded
paper available.** One more data point from their own ablation, directly
relevant to "what gradient combination matters": isolating guidance
(step size → 0) vs. their saliency-based position selection, **"selecting
positions using saliency has a much larger effect than guidance"** — in the
one paper that tested this for real, *where* you apply pressure mattered
more than *how* the gradient was shaped once there.

**Concrete, cheap adaptation path for mosaic (candidate, not decided):**
`unguided_direction` (§2c) is already computed for free every step, so
*some* NOS-style term reusing it is cheap. But the exact form is genuinely
unsettled: a separate KL-like penalty added before differentiation (closer
to NOS's actual term), a `_compat_project`-style projection of `g_total`
against `unguided_direction` (reusing existing PCGrad machinery), or a
post-merge correction after the existing trust-region clip — these are not
equivalent, differ in where in the graph the constraint acts, and trade off
differently against the clip already in §2a. Picking one is exactly the
open decision `guidance_implementation_todo.md` leaves unresolved.

---

## 4. The objective itself: ipTM/ipSAE evidence gap

Independent of mechanism — is what's being optimized even the right
target? mosaic's in-loop `L_bind` uses ipTM-family terms
(`DistogramIPTMProxy` for OpenDDE, `IPTMLoss`/PAE-based for Boltz2); ipSAE
is deliberately post-refold-only (enforced by a `ValueError` guard, ipSAE's
hard PAE-cutoff masking makes it unsuitable for step-wise gradient
guidance — matches `guidance_design_notes.md` §5.3).

Checked the actual ipSAE paper (Dunbrack, bioRxiv 10.1101/2025.02.10.637595):
it fixes a real bug in ipTM (score contaminated by disordered regions /
whole-chain length — trimming a construct to the interacting domains
inflates ipTM with no actual interface change). Validated on **40 true vs.
70 false heterodimer structures** — better at classifying real vs. spurious
interface, not at predicting binding strength. **Zero correlation data
against KD, SPR, or ITC is reported**, and the paper says so itself:
*"additional benchmarking is certainly required to demonstrate that the
ipSAE metric is able to rank the structural accuracy of models"* — even
structural-accuracy ranking, let alone affinity, is unproven.

**Net: mosaic's in-loop objective (ipTM-family) is the metric with the
least published evidence behind it in this whole stack; the post-refold
objective (ipSAE) is a real improvement at detecting spurious interfaces
but has never been shown to correlate with real binding strength either.
Nobody has published that number for either metric, for anyone.**

---

## 5. The decided approach: two paths, refined once (with codex)

First pass (mine) proposed a per-position additive AA-preference regression
as ground truth. Reviewed and corrected: the ~24k designed VHH72 variants
are campaign-biased (some prior guided-design process chose what to try,
not a random/exhaustive DMS) and mostly multi-mutant — an additive
regression imposes a no-epistasis assumption the data doesn't support, and
positions/substitutions nobody tried would be silently absent rather than
scored. Corrected approach, agreed:

**Path 1 — mutation-direction benchmark (AlphaSeq, no full trajectory
needed).** Build **local contrast pairs**: variant A vs. B differ by 1-2
CDR mutations, both have a real KD measurement against the same target —
that pair has direct empirical support for "this substitution/background
direction helps or hurts here." Then ask each guidance config: at the
WT/reference structure, when guidance shifts `p_seq`, does probability mass
move toward the empirically-beneficial substitution? Scored via sign
agreement, top-k hit rate, effect-size-weighted agreement, and coverage.
This tests **which gradient points toward beneficial mutations without
pretending ipTM/ipSAE are affinity** — directly avoiding the circularity of
"does guidance improve ipTM" as a benchmark, which would be meaningless
per §4.

**Correction (caught in review, verified directly — §9e): "no model, no
extrapolation" overstated it.** Scoring a WT-structure gradient against
pairs measured on *other* (non-WT) genetic backgrounds still assumes the
substitution's effect transfers to the WT background — that's exactly the
epistasis assumption this framing claimed to avoid. Checked directly: of
879 (position, substitution, target) keys with ≥3 independent background
measurements in the AlphaSeq VHH72 data, **61.9% show mixed-sign effects
across backgrounds** — the same substitution helps in some genetic contexts
and hurts in others. Path 1 is a **heuristic proxy that assumes rough
background-transferability**, not assumption-free ground truth. It's still
more direct than a fitted additive model (§5's original reasoning for
preferring pairs over regression still holds relative to that alternative),
but it is not extrapolation-free in absolute terms.

**Path 2 — full proposal benchmark (needs WT structures, more expensive).**
Once trajectories are cheap enough to run small batches: full guided
trajectories under a few configs (full controller vs. raw/naive sum,
OpenDDE-bind vs. Boltz2-bind, with/without AbLang2, with/without a future
prior-consistency term from §3b), evaluate final mutations against the same
AlphaSeq-derived empirical support. Tests the actual generation process,
not just the first-order gradient direction.

**Agreed priority: Path 1 first.** Cheaper, sharper, and directly answers
"what gradient is really working here" without waiting on full-trajectory
compute. If a gradient's first-order direction already pushes away from
AlphaSeq-beneficial mutations, that's reason enough not to trust it for
real inference regardless of anything else. Passing Path 1 earns the right
to be tested in Path 2 — it does not prove the mechanism is safe on its
own (§8 still applies).

---

## 6. Real data now in hand

### 6a. WT VHH72 → WT RBD structure

Copied into the repo: `vhh72_wt_wt_rbd.cif` (chain A = VHH72, 125 residues;
chain A2 = WT SARS-CoV-2 RBD). Source:
`/home/yfeng17/SBSAb/dataset/alphaseq/best_cifs/boltz_asq_scv2_ym_0005_sars_vhh72_sars_cov2_rbd_6lzg_bdfff4e22f_e21c75a70d.cif`.
**Caveat: this is a Boltz-predicted structure** (`structure_status:
raw_boltz`), not experimentally solved — no crystal/cryo-EM structure of
this complex exists in the dataset. Any error in that predicted pose
propagates into whatever's built on top of it.

### 6b. WT KD — thin, but usable

Exactly **one** row for genuinely unmutated VHH72 vs. WT RBD:
`KD_nM=3.65`, `neg_log10_KD_nM=-0.562`. **Zero** unmutated-WT-vs-Gamma or
-vs-Delta rows exist anywhere in the dataset (confirmed by direct query) —
no baseline for those two targets, only variant-vs-variant comparisons.

Bonus found in the same query: **12 "WT_synonymous" rows** — identical WT
protein sequence, different silent-codon DNA constructs, all against WT
RBD. Since the protein is identical, their spread is pure assay noise:
`range=0.360, std≈0.117` (neg_log10_KD_nM units). This is the noise-floor
estimate used below to sanity-check contrast-pair deltas. All 13 structures
(1 WT + 12 synonymous) exist as CIFs under `best_cifs/`.

### 6c. VHH72 CDR boundaries (IMGT, via ANARCI on the real WT sequence)

```
CDR1: 1-indexed 26-33  (0-indexed 25-32)   seq=GRTFSEYA
CDR2: 1-indexed 51-58  (0-indexed 50-57)   seq=ISWSGGST
CDR3: 1-indexed 97-114 (0-indexed 96-113)  seq=AAAGLGTVVSEWDYDYDY
```
Chain type H (heavy/VHH), as expected. CDR1/CDR2 land at the same raw
indices as a different nanobody (P17_JN1) checked earlier in this
project — consistent with typical VHH framework lengths, not a
coincidence specific to either molecule. Rederivation method documented in
`examples/alphaseq_vhh72_cdr_contrast_pairs.py`'s module docstring (needs
the separate `anarci` conda env present on this machine; not a mosaic
dependency).

### 6d. Contrast-pair counts — raw vs. CDR-restricted

Full reproducible script: `examples/alphaseq_vhh72_cdr_contrast_pairs.py`
(pure gemmi + numpy, no GPU/jax, ~35s for the full ~24k-variant set;
verified to reproduce these exact numbers).

**Naive whole-sequence pairs are mostly not CDR mutations.** Of all
distance≤2 pairs among the 19,775 length-125 (no-indel) variants (301,101
total), **78% involve at least one framework position** (132,649 mixed
CDR+framework, 102,974 framework-only) — restricting to real CDR
boundaries is not optional, it changes the answer by ~4x.

**CDR-only counts** (the actually-relevant subset, since mosaic's guidance
only touches designable/CDR positions):
- Distance-1 (clean, single substitution, no confound): **9,248 sequence
  pairs → 24,450 pair×target instances** (a pair can have shared KD
  measurements across up to 3 targets: WT/Gamma/Delta)
- Distance-2 (two simultaneous substitutions, effect not attributable to
  either position alone): 56,230 sequence pairs → 150,089 pair×target
  instances
- By CDR loop (distance-1, pair×target): **CDR1: 12,034, CDR2: 7,182,
  CDR3: 5,234** — CDR1 best covered, CDR3 thinnest (it's >2x longer, so any
  single-position match is rarer among 24k designs sampled by a biased
  campaign, not an exhaustive scan).

**Signal vs. noise floor (§6b):** median |Δneg_log10_KD_nM|=0.217,
mean=0.272, only **35.7%** of pairs clear the ~0.3 noise-floor threshold
from the WT-synonymous replicates. So the trustworthy subset is smaller
than the raw pair count — before using these as ground truth, filter by
this threshold (or, better, by each measurement's own
`ci_width_log10_KD_nM`, not yet wired into the script) rather than treating
every nonzero delta as real signal.

---

## 7. The comparison mechanism, specified precisely

Naming the guidance gradient's "does it move toward affinity" isn't a bare
score comparison — it has to be a **shift relative to unguided**, computed
through the same differentiable IF bridge every config uses, so configs are
compared on equal footing:

```python
x0_unguided = x0                      # no delta applied
x0_guided_A = x0 - delta_A            # config A's merged, clipped step
x0_guided_B = x0 - delta_B            # config B's merged, clipped step

p_seq_unguided = IF_phi(x0_unguided)  # (CDR positions, 20) probs
p_seq_A        = IF_phi(x0_guided_A)
p_seq_B        = IF_phi(x0_guided_B)

shift_A = p_seq_A - p_seq_unguided    # which residues gained/lost mass, config A
shift_B = p_seq_B - p_seq_unguided    # same, config B
```

At each CDR position where §6d's contrast-pair table has evidence, check
whether `shift_A`/`shift_B` moves mass toward the empirically-better
residue(s) and away from empirically-worse ones — aggregated via the
sign-agreement/top-k/weighted-agreement/coverage metrics from §5.

**Open question in this protocol, not yet decided: what is `x0`?**
1. Feed the real WT coordinates straight in as `x0`, skip BoltzGen's
   denoising call entirely. Cheapest; tests "if the model were exactly at
   this real structure, which way does guidance push it" — but is not what
   `step_body` actually does in production (never treats a structure as
   `x0_hat` without it coming from the denoiser).
2. Add noise to the real coordinates the way `guided_partial_diffusion`
   does at the start of a partial-diffusion run (some `sigma`/
   `start_sigma_frac`), run the actual denoiser forward pass to get a real
   `x0_hat`, then proceed as above. More faithful — exercises the real
   code path — but requires picking a noise level (or several), since
   `alpha`/`beta`/`lambda`/`tau` all depend on `sigma` and are explicitly
   designed to behave differently early vs. late in the trajectory.

Current lean: option 2, starting at a low-noise level (closest to "model is
nearly converged on something real, does guidance still push the right way
here" — the most directly interpretable regime), with higher-noise levels
as a later extension. Not yet built.

---

## 8. Honest epistemic framing (do not skip this when reading the rest)

Every layer above — intrinsic diagnostics, ipTM/ipSAE proxies, even the
AlphaSeq contrast pairs — rests on the same **static-structure assumption**.
A single predicted conformation, scored once, standing in for a dynamic
binding process (induced fit, conformational ensembles, entropy, solvation,
kinetics). Nothing in this stack (BoltzGen, Boltz2, OpenDDE) models any of
that, and the WT reference structure itself (§6a) is a Boltz *prediction*,
not a solved structure.

AlphaSeq is a better rung on the ladder than ipTM/ipSAE — real wet-lab
measurement — but it's a **yeast-display assay** with its own confounds
(expression level, avidity, surface presentation), not equivalent to a
solution SPR KD or in vivo efficacy. Agreement between the gradient and
AlphaSeq outcomes is evidence, not proof.

**No offline check in this document gets to "we know the gradient works."**
What each check retires is one specific, named risk:
- mechanical correctness — narrower than "verified" implies.
  `examples/opendde_smoke_test.py` checks a directional descent property
  (stepping along `-grad` lowers a smooth probe loss), for **one isolated
  path only**: a soft-sequence PSSM straight into OpenDDE's
  `build_distogram_only_loss`. Never touches `x0_hat`, BoltzGen's denoiser,
  or the BoltzGen-IF bridge — a genuinely different code path from what
  `step_body` actually runs in production. Not a literal finite-difference
  numerical-vs-analytic gradient comparison either (no such check exists
  anywhere in the repo, for either path). Should be built for the actual
  production path before treating mechanical correctness as settled there.
- internal consistency (masking/normalization/conflict-projection/
  trust-region) — verified in code, §2a/2b, and re-verified after this
  session's two bug fixes (§2d).
- prior-compatibility — not implemented as control, only measurable after
  the fact (§2c); best literature mechanism doesn't formally transfer to
  our setting (§3a); a cheaper literature-grounded alternative exists but
  is unbuilt and unvalidated for this domain (§3b).
- objective validity — unmeasured for both ipTM and ipSAE against real
  affinity, for anyone, anywhere (§4), and no paper in the guidance
  literature validated its own reward model before deploying it either
  (§3b) — the field's answer so far is iteration, not upfront proof.

This is not a design problem solvable on paper. It's empirical and
iterative. Offline checks (this whole document, Path 1 especially) exist to
filter out obviously broken configs *before* spending real assay budget —
not to replace the design→test→measure loop, which nothing here
substitutes for.

---

## 9. Real GPU results: Path 1 built, run, and scored

Everything in this section actually happened on GPU against the real
structure — not a plan anymore. Decisions from the old §9 (now superseded):
`x0` = noise-then-denoise at `start_sigma_frac=0.3` (real BoltzGen forward
pass, not the coords-direct shortcut); first configs compared = raw sum vs.
production controller vs. production controller + NOS-style consistency
term; NOS term's form = `_compat_project`-style projection (reused existing
PCGrad machinery); noise-floor filter = the blunt 0.3 threshold from §6d
(per-measurement `ci_width_log10_KD_nM` still not wired in).

### 9a. Interface crop: built, and one real bug found + reverted

`examples/crop_vhh72_wt_rbd.py`: crops the target to residues within 10Å of
a **CDR atom specifically** (not the whole binder chain — guidance only
touches the 34 CDR positions, so framework-proximal target residues are
irrelevant to what's being tested). 209 → 56 target residues, 2,619 → 1,402
total atoms. Also identifies the top-8 closest target residues as a
"hotspot" reference (TYR51, SER57, PHE59, LYS60, CYS61, GLY86, ARG90,
TYR190, all ≤2.96Å from a CDR atom) — diagnostic only, not wired into any
loss weighting.

**Bug found, "fixed," then correctly reverted.** `gemmi`'s
`setup_entities()` does not rebuild `entity.full_sequence` /
`_entity_poly.pdbx_seq_one_letter_code` from post-crop residue content — the
written CIF kept declaring the full 209-residue target even though
`_atom_site` correctly had only 56. This is not a bug in general — it's
standard mmCIF semantics (`full_sequence` = the complete construct including
unresolved regions) that BoltzGen's own parser
(`load_features_and_structure_writer`) explicitly relies on: it aligns
`full_sequence` positionally against residue numbering and treats the gaps
as "unresolved" (visible in its own log: `Removing leading and/or trailing
unresolved residues...`). First pass force-truncated `full_sequence` to fix
a naive reader (OpenDDE's `opendde json` CLI, which takes this field at face
value) — this silently broke BoltzGen's parser outright
(`AssertionError: polymer[i].name == res_name`), since it destroyed the
positional correspondence the gap-handling depends on. **Reverted** —
`full_sequence` is left untouched; readers that need a genuinely truncated
sequence (the raw OpenDDE CLI) get a hand-built JSON input instead, not a
truncated CIF. `vhh72_wt_wt_rbd_cropped.cif` in the repo is the correct
(reverted) version.

**Revised since (interface criterion + CDR verification, §12.0):** the
above used any-heavy-atom minimum distance at 10Å. Re-derived on request
with a tighter, more standard criterion — Calpha-Calpha distance — and the
CDR boundaries independently re-verified via a real ANARCI run rather than
trusted from memory (`examples/anarci_vhh72_cdr_boundaries.py`: exact match
against the existing 0-indexed boundaries, CDR1 25-32/CDR2 50-57/CDR3
96-113). Calpha at 5Å gave only 3 target residues — too small a shell to
carry any real local backbone context, a real and informative negative
result, not just a rejected parameter. Widened to 8Å: 15 target residues
(seqids 51-62, 64-66, mostly contiguous), residues `YNSASFSTFKCYVSP`, top
hotspot contacts TYR51(5.17Å)/ASN52(4.54Å)/ALA54(5.01Å)/SER57(5.41Å)/
THR58(4.24Å)/PHE59(5.13Å)/LYS60(4.08Å)/CYS61(5.03Å). **This Calpha/8Å
definition is now the real interface/hotspot characterization** (used for
reporting, not for building a crop to feed the model — per §12.0, inference
runs use the full complex, no crop).

`examples/vhh72_opendde_structure_prediction.py`'s full structure-prediction
path (`opendde_forward_from_trunk`, real diffusion sampling + confidence
heads, not the distogram-only guidance path) OOM'd catastrophically when
called eagerly — 223GB requested for the full 334-residue complex, 64GB for
the 181-residue crop. Isolated stage-by-stage (trunk / `expand_to_structural_
tokens` / `sample_coordinates` / `run_confidence_head`) — the trunk works
fine eagerly; `expand_to_structural_tokens` is where it blows up.

**Root cause: not a correctness bug, a usage-pattern requirement.**
Wrapping `expand_to_structural_tokens` in `eqx.filter_jit` fixed it
completely — 6.8s, normal memory (~180MB for the actual `z_st` tensor).
Without JIT, XLA can't fuse operations across the structural-token
refiner's transformer-style layers, so every intermediate over `(N_struct,
N_struct, C)` gets fully materialized with no fusion. Production code
(`refold_pareto_with_opendde`) was never exposed to this — it already
wraps this call in `eqx.filter_jit`. The standalone smoke tests
(`opendde_smoke_test.py`, `opendde_refold_smoke_test.py`) call it eagerly
too, and only ever "worked" because they use tiny synthetic sequences
(~56 residues) where unfused eager execution happens to still fit — **an
eager call at toy scale is not evidence a complex is tractable at real
scale.** Fixed in `vhh72_opendde_structure_prediction.py` by JIT-wrapping
the forward call; even so, the full 334-residue complex still genuinely
needs ~45-50GB (JIT reduced but did not eliminate the memory need) — more
than one 24GB GPU has regardless of allocator tuning, so that script now
runs against the cropped structure. `jopendde`'s `sample_diffusion` itself
(the actual diffusion loop) is well-written — real `jax.lax.scan`, explicit
loop-invariant hoisting with comments anticipating exactly this class of
bug — the problem is specifically the eager/JIT boundary at the call site,
not the vendored implementation.

### 9c. Does OpenDDE actually predict this real complex with confidence?

Motivating question: if OpenDDE can't confidently predict a real, known
binder, no gradient derived from it is worth trusting regardless of
mechanism. Ran the **raw torch OpenDDE CLI** (`opendde pred`, bypassing
mosaic's JAX port entirely) with the ABAG checkpoint, no MSA:

| | Full structure (209-res target) | Cropped (56-res target) |
|---|---|---|
| Raw torch | **ipTM=0.93, pLDDT=93.5** | ipTM=0.46, pLDDT=76.2 |
| Mosaic JAX port | not tested (too large even w/ JIT) | ipTM≈0.28, pLDDT≈72.3 (mean of 3 samples) |

**Confidence is genuinely high on the real, full complex** — OpenDDE isn't
confused about this pair. **Cropping measurably hurts confidence, on both
implementations, independent of JAX vs. torch** — a 56-residue fragment
made of four disconnected loop segments doesn't fold/present like a real
domain, so lower confidence there is expected, correct model behavior, not
an artifact. This is a real cost of the crop needed for the gradient test
(§9b/9d) — accepted, not fixed. The remaining raw-torch-vs-JAX gap on the
*identical* crop (0.46 vs 0.28) is unconfirmed — plausibly just fewer
refinement steps in the JAX test (`recycling_steps=1`/`n_step=8` vs. torch's
defaults `cycle=10`/`step=200`), not verified either way.

### 9d. Gradient-path comparison: built, run, real numbers

`examples/vhh72_gradient_path_comparison.py`: one real `x0_hat` (real noise
+ churn + BoltzGen denoiser forward pass, `start_sigma_frac=0.3`,
`t_hat=5.617`) on the cropped structure, fixed objective (OpenDDE bind +
AbLang2 nat + edit budget, identical across configs), three merges from the
*same* raw gradients:

```
g_bind_raw rms=0.006850   g_nat_raw rms=0.000473   g_edit_raw rms=0.505169
delta_raw rms=15.84   delta_full rms=5.04   delta_consistent rms=5.04
cos(delta_raw, delta_full) = -0.4304
cos(delta_full, delta_consistent) = 1.0000
cos(delta_full, unguided_direction) = -0.0037 -> cos(delta_consistent, unguided_direction) = 0.0000
```

Two things worth naming: (1) `g_nat_raw`'s magnitude is ~1000x smaller than
`g_edit_raw`'s at this step — AbLang2's raw signal is nearly negligible
before normalization; whether the merge handles that imbalance *sensibly*
(vs. just erasing real information by force-normalizing a near-zero
gradient to unit RMS) is untested. (2) the NOS-style consistency term had
**almost no effect** — `delta_full` was already nearly orthogonal to
`unguided_direction` (`cos=-0.0037`, razor-thin conflict), so there was
almost nothing for `_compat_project` to correct. Real finding *for this one
step* — the production controller isn't fighting the model's prior much
here — but it's one point on the seed/noise-level axis, not a general claim.

**Correction (caught in review, verified directly): "raw vs. controller" is
not a clean single-variable ablation.** They differ in at least five things
simultaneously, not just "the merge mechanism" as a unit: masking,
de-meaning, RMS-normalization, PCGrad conflict-projection, and trust-region
clipping. Checked directly and found a sixth, uncounted difference: `raw`
has **no masking at all**, so `delta_raw` has nonzero components on frozen
(non-designable) atoms too — and this script never re-anchors frozen atoms
after applying it (unlike real `step_body`, which resets them to the parent
position every step regardless of guidance). So `x0_guided_raw` genuinely
corrupts frozen framework/target geometry before it's read by the IF
bridge, on top of everything else. `raw`'s much larger magnitude
(`rms=15.84` vs. `5.04`) is itself a consequence of comparing an
unnormalized three-term sum against a normalized one under the "same"
`lambda(t_hat)` scaling — magnitude and direction are confounded, not
isolated. §9e's result should be read as "the full controller bundle
outperforms doing none of it," not as evidence for which specific
ingredient matters.

### 9e. Contrast-pair scoring: the actual quantitative answer

`examples/vhh72_score_gradient_vs_contrast_pairs.py`: for every clean
(distance=1) CDR contrast pair clearing the noise floor, does a config's
`p_seq` shift move probability toward the empirically better residue?
Position→token mapping via `binder_token_indices`; amino-acid→column
mapping via `mosaic.common.TOKENS` — sanity-checked before trusting it
(`p_seq_unguided`'s argmax matched the real WT residue at 20/34 CDR
positions, 59% vs. 5% chance for a wrong ordering; imperfect match is
expected, not concerning — BoltzGen-IF is inverse folding, not sequence
recovery, and the noised/denoised `x0_hat` has already drifted from exact
WT geometry, most visibly in the longer, floppier CDR3).

```
raw                    : unweighted=0.4776  weighted=0.4657  (n=7,835)
controller             : unweighted=0.5040  weighted=0.4984  (n=7,835)
controller+consistency : unweighted=0.5016  weighted=0.4945  (n=7,835)
```

**Correction — the original statistical claim here was wrong, caught in
review and independently reproduced.** The first pass treated `n=7,835` as
i.i.d. Bernoulli trials (SE≈0.0056, "raw is ~4 SE below chance, statistically
significant"). That's invalid: the 7,835 pairs are **not independent** —
they all draw on just **27 unique CDR positions** (the number of positions
this one gradient evaluation actually touches), with up to 1,665 pairs
sharing a single position's `shift` vector (min 2, median 97, max 1,665
pairs per position). The true unit of independence is the position, not the
pair.

Re-ran with a cluster bootstrap (resampling *positions* with replacement,
5,000 draws, pooling that position's pairs each time):

```
raw                    : 95% CI = [0.382, 0.564]  (contains 0.5)
controller             : 95% CI = [0.423, 0.575]  (contains 0.5)
controller+consistency : 95% CI = [0.418, 0.577]  (contains 0.5)
```

**All three configs' confidence intervals contain chance.** The
"statistically significant anti-correlation" claim for `raw` does not
survive proper dependency accounting — withdrawn. What the point estimates
still show, without over-claiming significance: `raw`'s point estimate
(0.478) sits below `controller`'s (0.504) and `controller+consistency`'s
(0.502), consistent with (not proof of) the controller bundle not being
harmful — but none of the three is distinguishable from random at this
sample size once the real degrees of freedom (27 positions, not 7,835
pairs) are used. Per-position agreement is also highly variable (min=0.0,
median≈0.39-0.47, max up to 1.0 depending on config) — a handful of
positions swing the aggregate substantially, another symptom of the same
small-effective-n problem.

Honest read, corrected: this result does not show the controller "rescues"
anything in a statistically defensible sense — it shows the point estimates
are *directionally* consistent with that story, on a sample too small (27
independent positions) to confirm it. It does not distinguish whether the
*mechanism* is insufficient or the *objective* (OpenDDE's distogram-only
`L_bind`, whose correlation with real affinity has never been measured by
anyone — §4) is simply not a useful compass. One seed, one noise level, one
crop, ~27 effective data points — see §10.

---

## 10. What this does and doesn't tell us / open next steps

**What §9e actually establishes, corrected:** at one specific point (one
seed, one `t_hat`, this crop, ~27 independent CDR positions), none of the
three configs' sign-agreement rates are statistically distinguishable from
chance — `raw`'s point estimate is directionally lower than `controller`'s,
but the confidence intervals overlap heavily and all three contain 0.5.
That's real, checked-against-real-data information, but it's weaker than
the first pass claimed, and it is still one point on the seed/noise axis:
"we can't really see it until we do a real run and compare" (raised
directly, and correct) — a single-step local probe, however carefully
scored, is not the same claim as "a finished, multi-step-guided design is
good."

**Concretely still open, roughly in priority order:**

1. **Path 2 — full trajectory, not one step.** Run `guided_partial_diffusion`
   for real (the full 200-step schedule) with the actual production
   controller, decode a finished sequence (not just a `p_seq` shift), score
   *that* against the same contrast-pair table. This is the direct answer to
   "we can't see it until we do a real run" — nothing in §9 substitutes for
   it. No raw-mode equivalent exists for a full trajectory yet (`step_body`
   always runs the full controller) — building a full-trajectory raw
   comparison is separate, additional work, not yet started.
2. **More seeds / noise levels / starting structures, for real statistical
   power.** §9e's effective sample size is ~27 independent CDR positions
   from one gradient evaluation — nowhere near enough to resolve a modest
   effect. Multiple seeds and noise levels would each add ~27-34 more
   positions per run; this is the direct fix for the small-n problem found
   in review, not just "nice to have."
3. **A cleaner mechanism ablation.** §9d's `raw` vs. `controller` changes
   masking, de-meaning, normalization, PCGrad-projection, trust-region
   clipping, and frozen-atom corruption all at once. Isolating which
   ingredient(s) actually matter needs one-change-at-a-time variants (e.g.
   masking-only, masking+normalize-only, etc.), not just the two
   bundled endpoints tested so far.
4. **Edit budget as a real variable.** `weight_edit_budget=5.0` was held
   fixed throughout §9d/9e. Different budget pressure changes how hard
   `L_edit` fights `L_bind`/`L_nat` in the merge — entirely unexplored.
5. **L_nat/L_bind interaction, not just L_bind's affinity-agreement.** The
   ~1000x magnitude gap between `g_nat_raw` and `g_edit_raw`/`g_bind_raw`
   (§9d) raises a real question about whether normalizing a near-zero
   gradient to unit RMS is meaningful or just noise amplification — untested.
6. **Realism/naturalness as its own axis, separate from affinity agreement.**
   Everything scored in §9e only checks "does the shift favor higher
   affinity." Nothing yet checks whether the resulting sequence still looks
   like a real antibody (e.g. AbLang2 PLL on the final decoded sequence, on
   its own terms) — the "not out-of-distribution / unrealistic design" half
   of the original question (§1) has no empirical check at all yet, only
   the affinity half does.
7. Per-measurement `ci_width_log10_KD_nM`-based noise filtering (§6d),
   still using the blunt 0.3 threshold.
8. Whether the raw-torch-vs-JAX-port confidence gap on the identical crop
   (§9c, 0.46 vs 0.28) is a step/cycle-count artifact or something worth
   chasing further.
9. **`alpha`/`beta`/`tau` are not config/CLI-exposed in the driver.** Only
   `lambda_max`/`lambda_schedule` are — confirmed directly (grepped
   `boltzgen_vhh_guided.py` for `guidance_alpha_fn`/`guidance_beta_fn`/
   `guidance_tau_fn`: zero hits). The driver never passes them, so they
   silently always resolve to `boltzgen.py`'s hardcoded defaults
   (`alpha_max=1.0`, `beta_max=1.0`, `tau_max=2.0`, `tau_min=0.05`). This is
   a real experimental-design limitation, not just a coding detail: it
   affects what can actually be swept, it means the "controller" family is
   only partially configurable from the workflow, and a reader could
   otherwise assume these schedules were available to tune alongside
   `lambda`. Only `lambda_max`/`lambda_schedule` are config-exposed in the
   current driver. `alpha`, `beta`, and `tau` still silently use
   `boltzgen.py` defaults, so schedule-level ablations on those terms are
   not yet possible without code edits.

Priority lean: (1) and (2) first — (1) is what everything else is a
variation on, and (2) is required before any version of §9e's result (from
any config) can be trusted as more than a single noisy draw. (9) doesn't
block the first sweep (§12c holds `alpha`/`beta`/`tau` fixed at defaults
deliberately) but should be resolved before claiming the "controller" family
was fully explored.

**Note on review provenance:** the statistical, epistasis, and ablation
corrections above came from an external review round. Its citations for
"the Gradient Guidance paper" and "the NOS paper" (arXiv:2502.07892,
2505.02111) were checked directly and are **wrong papers entirely** (cat
qubits and multimorbidity clustering, respectively — unrelated to
diffusion guidance). Its empirical claims (statistical dependency, epistasis
mixed-sign rate, missing re-anchoring in `raw`) were each independently
reproduced from scratch before being accepted here, not taken on faith —
same standard as everything else in this document. Citations from that
review should not be trusted without checking; its specific numerical/code
claims held up this time, but verify again next time rather than assuming
the pattern holds.

---

## 11. Reproducibility index

All scripts below are real, run on GPU this session, verified to produce
the numbers quoted above (not aspirational). Re-run order:

1. `examples/alphaseq_vhh72_cdr_contrast_pairs.py` — CDR-only contrast
   pairs from real AlphaSeq data (§6d). No GPU needed, ~35s.
1b. `examples/anarci_vhh72_cdr_boundaries.py` — independent re-verification
   of the CDR boundaries above from a real ANARCI run (§9a revision).
   Requires ANARCI first (separate `anarci` conda env, not scripted into
   this repo — see the module docstring for the exact command), then this
   script parses its CSV output. No GPU needed.
2. `examples/crop_vhh72_wt_rbd.py` — Calpha/8Å interface + hotspot
   characterization (§9a revision; **not used to build the structure fed to
   the model** — inference runs use the full, uncropped complex per §12.0).
   Still writes `vhh72_wt_wt_rbd_cropped.cif` for reference/diagnostics. No
   GPU needed.
3. `examples/vhh72_opendde_structure_prediction.py` — OpenDDE JAX-port
   confidence check on the crop (§9c bottom row). Needs
   `XLA_PYTHON_CLIENT_MEM_FRACTION=0.88`, GPU.
4. `examples/vhh72_gradient_path_comparison.py` — the actual gradient-path
   comparison (§9d), writes `vhh72_gradient_path_comparison_cache.pkl`
   (repo root) for step 5. Needs `XLA_PYTHON_CLIENT_MEM_FRACTION=0.88`, GPU.
5. `examples/vhh72_score_gradient_vs_contrast_pairs.py` — the final scoring
   (§9e). No GPU needed, reads the pickle from step 4.

Raw-torch (non-mosaic) OpenDDE CLI comparisons (§9c) used the separate
`OpenDDE` conda env (`conda activate OpenDDE`) and `opendde json`/`opendde
pred` directly — not scripted into this repo, commands are in-session only;
rerun via `opendde json -i <cif> -o <dir>` then `opendde pred -i <json> -o
<dir> -n opendde_v1 --load_checkpoint_path ~/.cache/opendde/checkpoint/
opendde_abag.pt --use_msa False -e 1` if needed again.

---

## 12. Real NOS-style and look-ahead mechanisms — built and tested, not yet run for real

### 12.0 Why this plan, not a different one

**Why real mechanisms now, not another proxy.** Testing `_compat_project`
against `unguided_direction` (§9d/9e) answered "does a cheap post-hoc
directional correction do anything" — it doesn't tell us whether either
paper's actual mechanism would, because neither was implemented. A result
built on a proxy can't license a conclusion about the thing it's a proxy
for. The only way to get a real answer is real code.

**Why production code, not another one-off script.** Everything scored so
far (`vhh72_gradient_path_comparison.py`) hand-rebuilds one step of
`step_body` inside a test script — useful for a first probe, but it means
the mechanism being tested and the mechanism that would ever actually run
in a real design campaign are two different code paths that can silently
drift apart. Building NOS-style (§12a) and look-ahead (§12b) as config-gated
fields on `VHHDesignConfig`, reused by both real workflow runs and the
assessment scripts, closes that gap — the thing being tested is the thing
that would ship.

**Why both mechanisms, not one.** They address different failure modes, not
overlapping ones. NOS-style (§12a) is an explicit penalty — it directly
regularizes the guided update toward what the unguided model would have
predicted, enforced by adding a term to the objective being differentiated.
Look-ahead (§12b) is structural — it argues that differentiating all the way
through the denoiser's own Jacobian (instead of stopping at `x0_hat`)
confines guidance to the subspace the network itself can express, with no
explicit penalty term needed. A design could plausibly need "stay close to
the prior" (NOS) without also needing "route the gradient through the
network's own structure" (look-ahead), or vice versa, or both, or neither —
that's an empirical question, not something arguable on paper, hence
building both and testing together rather than picking one on intuition.

**Why Path 1 needs to be well-powered before it can gate Path 2.** The
two-path structure (§5) was never meant to weight Path 1 and Path 2 equally
from the start: Path 1 is supposed to act as a cheap pre-inference filter,
and Path 2 (full multi-step trajectory, real decoded sequence, §10 item 1)
is the expensive, higher-fidelity test that's only worth running once Path 1
isn't obviously broken. But "isn't obviously broken" requires Path 1 to
actually have statistical power — right now it doesn't (§9e's ~27
independent CDR positions from one seed, all three configs' cluster-bootstrap
CIs containing 0.5). A gate that can't distinguish signal from noise isn't a
gate. That's why §10 item 2 (more seeds/noise levels) is being fixed
*before* trusting any config's Path 1 result, real or new, as grounds to
commit Path 2's compute.

**Why Path 2 still matters regardless of what Path 1 shows.** Path 1 —
even fully powered — is structurally a single-step local probe: it cannot
see accumulated drift across steps, re-anchoring interactions, schedule
decay, or what an actually-decoded final sequence looks like. A clean Path 1
pass would not prove the full generator works; a clean Path 1 null would not
prove it doesn't. Path 2 is not optional follow-up work contingent on Path 1
succeeding — it answers a different question that Path 1 cannot answer by
construction. Path 1 being underpowered blocks trusting it as *this
round's* gating decision, not the eventual need for Path 2 itself.

**Why combine the seed/noise-level power fix with the new-mechanism build,
not run them separately.** Purely a compute-efficiency call: both changes
require the same real GPU pipeline (real `x0_hat`, real gradients, real
`p_seq` shift, real contrast-pair scoring) — running that pipeline once
across `{raw, controller, controller+NOS-real, controller+look-ahead-real}`
x many seeds/noise-levels costs one GPU round, not two.

**On hardware and cropping — decided, not still open.** Earlier runs
(§9a-9d) targeted a 24GB-class GPU, which is why the CDR-interface crop
exists at all — the full 334-residue complex needed ~45-50GB for full
diffusion sampling even under JIT (§9b), more than that budget could close.
This work now runs on an H200 (140GB). **Decision: the real inference runs
(§12c and beyond) use the full, uncropped complex, not a crop.** The crop
scripts and the Cα/8Å interface definition (§9a, revised — see below) stay
in the repo as the real, ANARCI-verified interface/hotspot characterization
used for reporting and sanity-checking, not as what gets fed to the model.
The feasibility smoke test (does look-ahead's full-backward-pass cost
actually fit/run at a reasonable speed on the full complex) is still real,
outstanding work — a decision to use the full complex is not the same
claim as "verified tractable," and that check has not been run yet.

`controller+consistency` in §9d/9e was **not** a real implementation of
either paper checked in §3 — it was `_compat_project(g_total,
unguided_direction)`, a cheap post-hoc directional correction reusing
existing PCGrad machinery. It measured as a near-no-op at the one tested
step (`cos(delta_full, unguided_direction) = -0.0037`, almost nothing to
project away) because it isn't shaped like either paper's actual mechanism.
§2c (prior-compatibility) remains genuinely unimplemented: `unguided_direction`
is computed every step (`boltzgen.py:1094`) but never enters the merge —
it only appears afterward in diagnostics. This section plans the two real
candidates from §3, decided directly rather than left as a proxy.

**Both will be built as production code**, new fields on `VHHDesignConfig`
gating each on/off (default off, matching this project's fail-fast/no-silent-
default pattern), wired through the same CLI/build pattern
`weight_opendde_contact` etc. already use — reused by both real workflow
runs and the assessment scripts, not hand-duplicated per script.

### 12a. NOS-style consistency term — built, real iterative version (not the §9d proxy)

**The one-shot translation originally planned here was mathematically a
no-op, caught before implementation, not after.** The first design —
`L_consistency = ||x0_hat_candidate - x0_hat_unguided||^2` with
`x0_hat_unguided = stop_gradient(x0_hat)` — was checked directly in JAX
before writing any production code: value `0.0`, gradient exactly
`[0., 0., 0.]`, not approximately. Reason: `x - stop_gradient(x)` is
identically the zero function in value, and differentiating a squared
function at `u=0` gives `2u=0` regardless of the (nonzero) `du/dx` the
stop-gradient trick otherwise produces. A "fixed" two-stage variant
(penalize a *candidate* `x0_hat - s*g_aux` against the anchor) was also
checked directly and found to be **exactly proportional to `g_aux`**
(`jnp.allclose` confirmed) — i.e. a disguised rescale of the existing
merged gradient, not new directional information. More generally: **any
one-shot squared-distance penalty built from a candidate that is an affine
function of the same auxiliary gradients being regularized can only ever
produce a gradient proportional to those gradients** — a distance penalty's
gradient always points along the displacement vector, and if that
displacement is `-s·g_aux`, the derivative is forced to point along `g_aux`
too. There is no one-shot construction that escapes this.

**Why NOS's real mechanism doesn't hit this wall.** NOS iterates: `h^(i+1)
← h^i + λ₁∇_h v_θ(h^i) + λ₂∇_h KL(...) + λ₃z^i`, K inner Langevin steps
per outer diffusion timestep. Each inner step re-evaluates `v_θ` at the
*already-moved* `h^i`, not the original `h^0`, while the KL term keeps
pulling back toward the fixed `h^0` — that feedback loop (plus injected
noise) is what breaks the one-shot degeneracy. Separately, NOS's KL term
acts on a genuinely nonlinear object (a *categorical* distribution over
discrete tokens from a decoder head), where KL is not equivalent to
Euclidean distance between hidden states, unlike BoltzGen's fixed-covariance
Gaussian coordinate parameterization, where mean-KL genuinely does reduce to
squared L2 distance. Both properties — iteration, and a term whose gradient
carries direction beyond plain distance — are structurally required; a
faithful port needed both, not a distance term bolted onto one merge step.

**Built: genuine K-step iterative version**, `_nos_iterative_merge` in
`src/mosaic/models/boltzgen.py`. Each inner step: re-run the existing
mask/center/normalize/PCGrad merge (`_merge_aux_gradients`, factored out of
the one-shot path so both share the same code) at the *current* point, add
a consistency gradient `2·λ_kl·(x0_i - anchor)` (nonzero from inner step 1
onward, since `x0_i` has moved by then), take one small inner step. `K=0`
(default) is byte-identical to the prior one-shot behavior — verified by
`_merge_aux_gradients` reproducing the exact pre-refactor computation
(`test_merge_aux_gradients_matches_manual_pcgrad_merge`).

**Real, verified nuance found while testing this:** for a *single*
objective, once `_mask_center_normalize` rescales every re-evaluated
gradient to unit RMS, the direction along a straight-line path to that
objective's minimum is invariant — so `K` re-evaluated steps with
`lambda_kl=0` coincide exactly with `K` copies of the one-shot step
(`test_nos_iterative_merge_pure_reevaluation_matches_naive_repeat_for_one_objective`).
Bare re-evaluation is *not* by itself a source of new dynamics when only one
objective is active — the consistency term itself, and PCGrad's asymmetric
conflict projection when ≥2 objectives are active (the realistic production
case), are the two real sources of iteration-dependent behavior here.

**Real stability bug found and fixed during testing, not from the source
paper.** The consistency-only recursion is `(x_i+1 - anchor) = (1 -
2·step·λ_kl)·(x_i - anchor)` — a linear system that diverges geometrically
whenever `2·step·λ_kl > 1`. Confirmed directly: `step=0.1, lambda_kl=50`
grew a small perturbation to a distance of ~2000 over 5 steps instead of
shrinking it. Fixed by capping each inner step's raw displacement via the
existing `_clip_rms` guardrail, scaled to `tau_fn(t_hat)/K`, so a bad
`(inner_step, lambda_kl)` combination from the sweep in §12c can't silently
produce exploded garbage (`test_nos_inner_step_clip_bounds_unstable_hyperparameters`
verifies the same combination stays bounded post-fix).

**Wired as real, reusable production code**, not test-script-only:
`VHHDesignConfig.nos_inner_steps` (default `0`, off) / `nos_inner_step_size`
/ `nos_lambda_kl` / `nos_langevin_noise`, CLI flags `--nos-inner-steps` etc.,
threaded through to `guided_partial_diffusion`'s new
`guidance_nos_inner_steps`/`guidance_nos_inner_step_fn`/
`guidance_nos_lambda_kl_fn`/`guidance_nos_noise_fn` kwargs. `lambda_kl` and
inner step size are swept in §12c's test plan, including `nos_inner_steps=0`
as the exact-regression baseline against everything already in §9.

Status: implemented and unit-tested (15 tests, `tests/test_guidance_controller.py`,
no GPU/model required — pure-JAX primitive tests against hand-built gradient
closures, same style as the existing controller tests). **Not yet run
against real BoltzGen/OpenDDE/AbLang2** — that's §12c.

**Real bug found by review, verified directly, fixed:** the NOS branch in
`step_body` originally set `delta = x0_hat - x0_candidate` with no
`lambda_fn(t_hat)` multiplication, unlike the one-shot path's `delta =
lambda_fn(t_hat) * g_total`. Confirmed by reading the code directly (not
taken on the review's word) — `--lambda-max`/`--lambda-schedule` were
silently dead for any `nos_inner_steps > 0` run: guidance strength was
controlled only by `nos_inner_step_size`/`nos_lambda_kl`/the inner clip,
with the documented, swept-in-§12c outer schedule doing nothing. This
confounds any controller-vs-controller+NOS comparison, since it changes two
things at once (the prior-compatibility mechanism *and* the meaning of the
main strength knob), not one. Fixed: `delta = lambda_fn(t_hat) * (x0_hat -
x0_candidate)`, then the same `_clip_rms(delta, tau_fn(t_hat), ...)` as
before. This does not erase the consistency term's shrinking effect (a run
with strong `nos_lambda_kl` still produces a smaller pre-`lambda_fn`
displacement, which `lambda_fn` then scales same as any other) — it
restores `lambda_fn` as the one outer amplitude knob with consistent meaning
across both paths; `nos_inner_step_size`/`nos_lambda_kl` remain the inner
solver's own step size and pull-back strength, mechanism-internal parameters
analogous to how `alpha_fn`/`beta_fn` already are, not a substitute for the
outer schedule. Existing unit tests (§12a above) were unaffected — they
exercise `_nos_iterative_merge` directly, below where `lambda_fn` applies;
full suite re-verified green after the fix (60 passed, 7 deselected).

### 12b. Look-ahead gradient — built (arXiv:2404.14743's actual mechanism)

Not a multi-step rollout (a different technique used elsewhere in the
guidance literature) — specifically: differentiate `∂L/∂x_t` through the
full denoiser network, not just through the guidance loss's own small graph
attached to a frozen `x0_hat` (which is what `x0_hat =
jax.lax.stop_gradient(x0_hat)`, `boltzgen.py:1073`, blocks for the one-shot
and NOS paths). Per Theorem 1, this confines guidance to the subspace the
model's own Jacobian can express — a structural argument for staying
in-distribution, distinct from NOS's explicit penalty. **Caveat already
established in §3a**: the theorem is proved only under a linear-subspace
data assumption that does not hold for a real nonlinear denoiser like
BoltzGen — treat as a heuristic worth measuring, not a guaranteed fix. Real,
uncosted cost: a full backward pass through the entire denoiser trunk per
guidance term per guided step, not just through the auxiliary-loss subgraph.

**Code split, deliberately:** `src/mosaic/models/guidance_lookahead.py`
holds only `build_lookahead_grad_fn` — compose "denoiser forward → guidance
loss" and differentiate with respect to the denoiser's input. This piece is
genuinely self-contained (independently testable with a synthetic denoiser)
and reusable, so it lives in its own module rather than inline in
`boltzgen.py`. The actual merge and where-to-apply-the-result logic stays in
`step_body` and reuses `_merge_aux_gradients`/`_clip_rms` unchanged (both
already shared with §12a) — that logic is tightly coupled to `step_body`'s
per-step state (`atom_coords_noisy`, `t_hat`, `network_condition_kwargs`,
re-anchoring, diagnostics) and pulling it into a second file would just
create a circular import back into `boltzgen.py`'s internals, or force
duplicating the merge machinery — exactly the test/production drift §12.0
was trying to avoid.

**A real sign bug, found by review, verified empirically, fixed — the
original "verified numerically" claim below was wrong in a specific,
instructive way.** Look-ahead's gradient is with respect to
`atom_coords_noisy`, not `x0_hat` — the one-shot and NOS paths both only
ever adjust `x0_guided`, never touching `atom_coords_noisy` itself. The
first version of this code hand-derived an x0-equivalent conversion by
solving for what `x0_guided` would reproduce "literally nudge
`atom_coords_noisy` in the direction calculation only," landed on `delta =
-t_hat * lambda_fn(t_hat) * g_total` (note the minus), and checked that
this construction was *internally self-consistent* (`jnp.allclose` against
the literal-nudge formulation) — which it was. But self-consistency of a
derivation is not the same as correctness of its premise, and the premise
itself was never checked against the one thing that actually matters: does
it make the guidance loss go down. It didn't. Reproduced directly on a toy
nonlinear denoiser + quadratic loss: `guidance_loss(x0_hat) = 9.12`,
`guidance_loss(x0_hat + t_hat·λ·g_total)` (the shipped, buggy formula)
`= 12.67` — **worse than doing nothing** — versus `guidance_loss(x0_hat -
λ·g_total) = 5.17`, a genuine decrease. `g_total` here (the gradient through
the denoiser, w.r.t. `atom_coords_noisy`) is empirically well-aligned in
direction with the direct x0-space gradient at `x0_hat` (cosine ≈0.93 in the
toy case) — descending along it works exactly the way descending `g_bind`
does in the one-shot path. **Fix: no special sign, no extra `t_hat` factor —
treat it identically to the one-shot/NOS merged gradient:**

```python
delta = lambda_fn(t_hat) * g_total   # identical in form to the one-shot path
x0_guided = x0_hat - delta            # same shared line as the other two paths
```

Simpler than what was originally built, and empirically verified (not
re-derived by hand a third time) via the same direct check: `guidance_loss`
after one guided step, compared against the unguided baseline, using the
*real* `guided_partial_diffusion` code path with a fake denoiser — not just
a standalone script.

**Wired as real, reusable production code**: `VHHDesignConfig.lookahead`
(default `False`), CLI flag `--lookahead {0,1}`, threaded through to
`guided_partial_diffusion`'s new `guidance_lookahead` kwarg. **Mutually
exclusive with `guidance_nos_inner_steps > 0`** — raises `ValueError` if
both are set (combining them would mean `K` full denoiser backward passes
per step, not scoped or tested; matches the existing Boltz2/OpenDDE
guidance mutual-exclusivity precedent in this codebase).

**Test gap found by the same review, also fixed**: the original tests
checked that look-ahead "runs and changes the result" relative to no
guidance — a property the sign-flipped bug also satisfied, so it passed
despite being wrong. Added
`test_lookahead_actually_reduces_the_guidance_loss`, which runs the real
`guided_partial_diffusion` end to end and asserts the final structure's
guidance loss is actually lower than the unguided baseline's — the property
that was missing, and the one that would have caught this.

Status: implemented and tested — `tests/test_guidance_lookahead.py` (6
tests): `build_lookahead_grad_fn` matches direct composition and (the actual
point of the mechanism) genuinely differs from the stop-gradient gradient
the other two paths compute; a full `guided_partial_diffusion` run with a
synthetic fake `structure_module`/`Sampler` (deterministic nonlinear
function standing in for the real denoiser — no GPU/checkpoint needed)
confirms `guidance_lookahead=False` is byte-identical to prior behavior,
`guidance_lookahead=True` runs end to end, and — the test that used to be
missing — actually reduces the guidance loss relative to no guidance at all.
This is an integration-level check, not just the isolated closure builder,
since the sign derivation above is the most subtle part of section 12 and a
wiring mistake would not show up in a unit test of `build_lookahead_grad_fn`
alone. Full suite green: 66 passed, 7 deselected. **Not yet run against
real BoltzGen/OpenDDE/AbLang2** — that's §12c.

### 12c. Combined test plan

Decided: fix §10 item 2's statistical-power problem (27 independent CDR
positions from one seed) and test the two new real mechanisms in the same
pass, rather than two separate GPU rounds. Test matrix: `{raw, controller,
controller+NOS-real (12a), controller+look-ahead-real (12b)}` swept across
multiple seeds and noise levels on the **full, uncropped complex** (§12.0 —
decided, not the crop) (concretely proposed: ~8 seeds x 3 `start_sigma_frac`
values spanning early/mid/late in the schedule — open to adjustment once
the first pass runs).

**Compute note:** `raw`, `controller`, and `controller+NOS-real` all share
the same `stop_gradient`-based `x0_hat` and raw gradients per `(seed,
t_hat)` draw — only the merge differs between them, which is cheap.
`controller+look-ahead-real` needs a genuinely separate, more expensive
gradient computation (full backprop through the denoiser) per draw. So the
real GPU cost is approximately two expensive computations per draw (one
`stop_gradient`-based, one full-backprop-based), not four, but both are now
on the full complex, not the crop — §9b already measured ~45-50GB for the
full complex under the one-shot path's much smaller backward graph; H200
(140GB) should cover that with room to spare, but look-ahead's full-denoiser
backward pass on the full complex is a real unknown, not yet measured. The
feasibility smoke test (§12.0) should run before committing to the full
sweep's size/cost.

Scoring stays the protocol from §7 (sign-agreement vs. real AlphaSeq
contrast pairs), but aggregation changes: seeds now give genuine independent
replicates per CDR position (fixing the pseudo-replication problem in §9e),
so the cluster bootstrap resamples positions with real within-position
variance across seeds, reported per noise level and pooled across all three.

**Status: both §12a and §12b are implemented and tested** (real, reusable
production code, config/CLI-gated, default off — `_nos_iterative_merge`/
`_merge_aux_gradients` and the look-ahead wiring in `boltzgen.py`,
`build_lookahead_grad_fn` in the new `guidance_lookahead.py`, config/CLI
fields in `boltzgen_vhh_guided.py`). 21 new tests total (15 in
`tests/test_guidance_controller.py` for §12a, 6 in the new
`tests/test_guidance_lookahead.py` for §12b, the latter including a full
`guided_partial_diffusion` run against a synthetic fake denoiser that
directly checks the guidance loss actually decreases, not just isolated
primitives — a real sign bug in §12b was caught this way, see above).
Full suite green: 66 passed, 7 deselected slow.
**Neither has been run against real BoltzGen/OpenDDE/AbLang2 or scored
against AlphaSeq contrast pairs yet** — same standard as the rest of this
document; this section will be updated with real numbers once that has
happened, not before.

---

## 13. Decision: setting aside BoltzGen-guided diffusion for this problem

**§12's mechanisms are real, tested, and stay in the repo** — this is not a
retraction of that work. It's a decision that BoltzGen-guided diffusion is
not the right tool for *this specific problem* (edit-budget-constrained
redesign of a known VHH72 starting point), based on concrete findings, not
a vague sense of difficulty. The distinction matters: **this is not a claim
that diffusion as a paradigm is wrong for protein design** — it's that
BoltzGen's specific pretrained checkpoint, trained for general
structure/binder generation, has several concrete mismatches with a
constrained-few-edit redesign task specifically. A model actually trained
with an edit-budget-aware curriculum could plausibly not have any of these
problems; that's a different, uninvestigated question.

**The concrete findings that drove this, each independently verified this
session, not asserted:**

1. **The rounds-of-iteration ceiling.** `alpha_fn`/`beta_fn` ramp up
   naturalness/edit-locality weight as `t_hat` decreases — by design, since
   early-schedule structure is mostly noise and position-level signal is
   unreliable there. So genuinely well-informed guidance steps are
   concentrated toward the *end* of an already-truncated
   (`start_sigma_frac`) schedule, not spread evenly across it. Any scheme
   for progressively narrowing which positions get edited needs real rounds
   to work with, and BoltzGen's pretrained schedule wasn't built with that
   curriculum in mind — it was built for generic denoising.
2. **`EditBudget` is blind to concentration vs. diffusion of edits.**
   `E = sum_i (1 - <s_i, s_ref_i>) * designable_i` is a linear sum — `0.1`
   deviation spread across 50 positions and `1.0` deviation concentrated on
   5 give the identical `E` and identical hinge penalty. Nothing in the
   objective prefers "a few confident mutations" over "everything drifts a
   little" — that's a real gap in the loss, not a tuning problem.
3. **The IF bridge actually used (`differentiable_jax_boltzgen_if` →
   `JaxBoltzGenIF.__call__`, `boltzgen_if_jax.py:306`) does hard, sequential,
   argmax-committed autoregressive decoding**, not the soft mean-field
   process an earlier pass through this document wrongly assumed (traced to
   the wrong function initially — `differentiable_inverse_fold`/ProteinMPNN,
   corrected directly from source). Each designable position's residue is
   chosen via `jnp.argmax` and written into `decoded_sequence` as a real,
   fixed commitment that every later position (in a random order, fixed
   once per outer iteration) conditions on. No revisiting mechanism exists.
   A position decided early can commit to something a later position's
   context then can't undo.
4. **No existing literature solves the reconciliation problem this creates**
   for this specific architecture. Real precedent exists for progressive
   discrete commitment (Mask-Predict, arXiv:1904.09324; ReMDM,
   arXiv:2503.00307; ProtLiD²'s MCM-ReMask, arXiv:2605.27413) but all of it
   assumes a true discrete-diffusion or NAR generative process with its own
   timesteps to hook a remasking schedule into — which the sequence side
   here does not have (sequence is a single autoregressive IF pass, not a
   diffusion process; conflating the two was a real error caught mid-session).
   The closest real antibody-design precedent, NOS/LaMBO-2 (arXiv:2305.20009,
   re-verified directly from the paper text), sidesteps the problem
   differently — it selects edit positions *once, upfront, before generation
   even starts* — a structurally different design than either "revert
   repeatedly" or "decide once at the end," and one this pipeline doesn't
   currently implement either.
5. **mosaic's own validated track record is entirely hallucination-based,
   not guided-diffusion-based** (confirmed directly from `README.md`): the
   minibinder blog post (Boltz-2 + soluble ProteinMPNN + `simplex_APGM`,
   real wet-lab 8/10 and 7/10 hit rates), the Adaptyv Nipah competition win,
   ORBIT winning GEM×Adaptyv — all hallucination. Guided diffusion through a
   generative model gets one passing mention in the README as a secondary
   use case, with no cited track record. This is independent evidence about
   which paradigm is actually proven inside this exact toolkit, not just an
   architectural argument.

**Decided path forward: hallucination-based design** — direct gradient
optimization of a soft sequence through a fixed structure-confidence
predictor (OpenDDE or Boltz-2), combined with naturalness losses
(AbLang2/ESM-family) and `EditBudget`, with no BoltzGen diffusion schedule
involved at all. This sidesteps finding (1) entirely (no borrowed schedule —
the number and pacing of optimization rounds is fully controlled, not
dictated by a pretrained curriculum) and gives finding (2)'s sparsity gap a
much simpler place to be fixed (the same `EditBudget`/L0-style sparsity
question applies, but without also fighting a noise schedule at the same
time). It trades away diffusion's implicit realism prior (the
noise-injection process that keeps samples near the data manifold by
construction) — a real cost, not a free lunch — but mosaic's own cited case
studies already manage that risk in practice via naturalness losses plus
edit-budget constraints, with real wet-lab results to show it works, which
the guided-diffusion path does not have.

**That next round has now been scoped and built as a first real pass.**
There is now a concrete VHH72 hallucination pipeline in
`examples/vhh72_hallucination_search.py`: OpenDDE specifically, paired with
an explicit search policy, using the existing loss-agnostic mosaic search
machinery in `src/mosaic/optimizers.py` rather than BoltzGen's diffusion
loop. The current implementation is:

- stage 1: `simplex_APGM` continuous relaxation on a composite loss
  (OpenDDE contact + pose-anchor + AbLang2 + `EditBudget`)
- stage 2: exact hard-budget discrete search from the continuous result via
  `edit_budgeted_greedy_descent` or `edit_budgeted_gradient_mcmc`

This is exactly the kind of budget-constrained search-policy augmentation
the paragraph above anticipated; it is no longer just a proposed direction.
What remains open is empirical calibration, not whether the path exists:
the weights are still first-pass choices, the pose-drift margin is still a
heuristic, and the AbLang2 `stop_grad` setting remains a live ablation axis.

The sharper long-term point underneath this whole investigation remains the
same: every workaround explored here (NOS, look-ahead, sparsity
regularization, search-policy-augmented hallucination) is a way of
retrofitting constrained-edit behavior onto a model that wasn't trained for
it — the real fix would be a model actually co-trained for joint
structure/sequence generation under an edit-budget curriculum, which
doesn't exist yet for this problem.

**Next step:** run and assess the hallucination-based approach concretely,
not scope it from scratch. The real open questions now are which discrete
search policy works better here (`greedy` vs `mcmc`), whether AbLang2 should
backprop through its own encoder (`stop_grad=0`) or act only as fixed
per-token reweighting (`stop_grad=1`), and how sensitive the result is to
the current first-pass loss weighting and pose-drift tolerance choices.
