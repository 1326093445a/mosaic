# Guidance Implementation TODO

This is a companion to `docs/guidance_design_notes.md` (coordinate-gradient
controller) and `docs/guidance_search_summary.md` (search object and
selection policy). Those two documents establish the design; this document
tracks the concrete, sequenced implementation plan derived from them, plus
the three papers most directly informing the open questions:

- Gradient Guidance for Diffusion Models: An Optimization Perspective
  (arXiv:2404.14743)
- PCGrad: Gradient Surgery for Multi-Task Learning (arXiv:2001.06782)
- Protein Design with Guided Discrete Diffusion / NOS (arXiv:2305.20009)

The phases below are ordered deliberately. Do not start a phase before the
previous one's decision gate (where one exists) is satisfied.

---

## Phase 1 — Mechanical controller rewrite

Rationale: this fixes a demonstrated flaw in the current implementation
(single merged scalar loss, unclipped raw coordinate gradient, clipping
applied at the wrong point in the graph — `dL/d(p_seq)` instead of
`dL/d(x0_hat)`). It does not depend on resolving the prior-compatibility
question below, and every piece is standard, well-understood technique.

- [x] Restructure `guidance_fn` from one merged scalar into three separate
      objective evaluations: `L_bind`, `L_nat`, `L_edit` — done via
      `guidance_fn_bind`/`guidance_fn_nat`/`guidance_fn_edit` params on
      `guided_partial_diffusion` (`src/mosaic/models/boltzgen.py`)
- [x] Compute three separate gradients (`g_bind`, `g_nat`, `g_edit`) via
      separate `jax.grad` calls — no more single merged `g`
- [x] Apply `M_design` mask to each gradient independently
- [x] De-mean each gradient over designable atoms (kill rigid-translation
      component)
- [x] RMS-normalize each gradient over designable atoms
      — masking, de-meaning, and normalization implemented together in
      `_mask_center_normalize`, numerically unit-tested (frozen atoms exactly
      zero, pure rigid-translation gradient fully cancels, RMS = 1.0 on
      designable atoms)
- [x] Implement `compat()` — PCGrad-style projection, asymmetric:
      `g_nat`/`g_edit` get projected against `g_bind`, `g_bind` itself is
      never modified — `_compat_project`, unit-tested (agreeing gradients
      pass through unchanged, conflicting gradients have the conflicting
      component removed to exactly zero dot product)
- [x] Merge:
      `g_total = g_bind + alpha(sigma)*compat(g_nat,g_bind) + beta(sigma)*compat(g_edit,g_bind)`
- [x] Trust-radius clip: `clip_rms(lambda(sigma)*g_total, tau(sigma))`
      — `_clip_rms`, unit-tested (caps RMS at tau, no-ops when already under)
- [x] Pin down actual functional forms for `alpha(sigma)`, `beta(sigma)`,
      `lambda(sigma)`, `tau(sigma)` — implemented as `default_alpha_schedule`,
      `default_beta_schedule`, `default_lambda_schedule`, `default_tau_schedule`.
      Explicitly documented as first-pass defaults to be tuned against Phase 2
      diagnostics, not final answers. Unit-tested for the qualitative shape
      design notes specify: `lambda` decreases monotonically as sigma -> 0,
      `alpha`/`beta` increase as sigma -> 0, `tau` respects a floor so the
      cap never collapses to zero at the final step.
- [x] Expose the **actual unguided reverse-update direction as computed by
      the step body** as an auxiliary output of `step_body`, even if
      nothing consumes it yet. This must match the real quantity the
      sampler would use with `lambda=0` — i.e. built from
      `atom_coords_noisy` (post-churn, and post rigid-realignment to
      `x0_hat`'s frame when `alignment_reverse_diff=True`, which is the
      case for BoltzGen-1 release checkpoints), not a hand-derived
      shorthand like `(x0_hat - x_t)/t_hat`. `x_t` and `atom_coords_noisy`
      are not the same tensor once churn noise and optional rigid
      realignment are applied, and comparing against the wrong one would
      quietly measure the diagnostic in the wrong reference frame. This
      value already exists inside the current computation; costs nothing
      now, costly to retrofit once this block is rewritten and settled.
      Note: the cosine-similarity diagnostic in Phase 2 is invariant to
      the `step_scale*(sigma_t - t_hat)` scaling factor, but the norm-ratio
      diagnostic is not — compute the norm ratio on the fully-scaled
      coordinate delta, not the raw `denoised_over_sigma` term, or it will
      not correspond to an actual physical displacement.
      — Implemented as `return_diagnostics=True` on `guided_partial_diffusion`;
      `step_body` computes `unguided_direction` from `atom_coords_noisy`
      (matching the note above) and `jax.lax.scan` stacks it per step when
      requested. `False` by default and resolved at trace time (plain Python
      bool), so no diagnostic computation is added to the graph when unused.
      Also exposes `guided_direction`, `g_bind`, `g_nat`, `g_edit` per step —
      the last three are what Phase 2's pairwise objective-conflict logging
      needs.
- [x] Confirm `ipSAE` stays post-refold only, never an in-loop gradient
      target (agreed independently across both design docs and both the
      Gradient Guidance and NOS papers) — holds: no `ipSAE` term appears
      anywhere in the new controller; `L_bind`/`L_nat`/`L_edit` are the only
      in-loop objectives.

**Caller wired up:** `src/mosaic/workflows/boltzgen_vhh_guided.py`'s
`build_guidance_loss` now returns a `GuidanceLosses(bind, nat, edit)`
dataclass instead of one pre-summed loss — `bind` = the Boltz2
interface-confidence surrogate (`L_bind`), `nat` = combined ESM2 + AbLang2
(`L_nat`), `edit` = the edit-budget/locality term (`L_edit`). The driver
builds three separate `guidance_fn_*` closures (one per non-`None` field)
and calls `guided_partial_diffusion` with the new signature. v1's
"EditBudget-only guidance" behavior is preserved via an explicit fallback:
when no Boltz2 binding signal is configured, `edit` is promoted to fill the
`bind` (anchor) slot instead of being dropped, since the new controller
requires a bind objective for guidance to be active at all. Full module
import verified end-to-end (`import mosaic.workflows.boltzgen_vhh_guided`
succeeds, including `GuidanceLosses` and the updated `build_guidance_loss`).
Not yet done: an actual end-to-end run with real BoltzGen weights (needs the
checkpoint download and GPU time SETUP.md's v0 smoke test describes) —
everything above is verified at the import/unit-test level, not by running
the full sampler yet.

---

## Phase 2 — Diagnostics only

Rationale: cheap, high-value, and required before committing to *any*
prior-compatibility fix. The goal is to find out empirically whether
external guidance is actually fighting the BoltzGen prior often enough to
matter for this pipeline, not to assume it either way.

- [ ] Log cosine similarity: guided step direction vs. unguided direction
      (uses the exposed value from Phase 1)
- [ ] Log norm ratio: `‖guided step‖ / ‖unguided step‖`
- [ ] Log fraction of steps with strong directional disagreement (pick and
      justify a threshold)
- [ ] Stratify disagreement by `sigma` (noise axis)
- [ ] Stratify disagreement by **which objective** is driving it — separate
      stats for `g_bind` vs. `g_nat` vs. `g_edit`, not just the merged
      total. This determines whether an eventual fix should be global or
      objective-specific (e.g. only AbLang2's naturalness term may be the
      source of conflict, not the structurally-grounded binding term).
- [ ] Log per-step pairwise conflict *between the objectives themselves*:
      `cos(g_bind, g_nat)` and `cos(g_bind, g_edit)`, separate from the
      objective-vs-prior stratification above. This distinguishes two
      different problems that could otherwise be conflated: objectives
      disagreeing with *each other* (which Phase 1's PCGrad merge is
      already designed to handle) vs. the merged external signal
      disagreeing with the *prior* (the Phase 3 problem). It also doubles
      as a sanity check on whether Phase 1's conflict-projection logic is
      doing meaningful work at all, or solving a conflict that rarely
      occurs in practice.
- [ ] Correlate disagreement against final outcome, using **both** pose
      RMSD and ipSAE, not just one metric. (These two metrics have been
      directly observed to disagree on the same designs in this project's
      own validation work — do not trust either alone as ground truth for
      "good outcome.")
- [ ] Run on a modest sample of real trajectories, not synthetic/toy cases

**Decision gate — stated precisely, in three parts:**

1. Prior compatibility is a real, conceptually unresolved issue
   regardless of what Phase 2 finds. This is not up for revision by a
   single diagnostic run.
2. What Phase 2 actually decides is narrower: whether an *active
   corrective term* is needed **now**, and if so, which failure pattern
   (which sigma range, which objective) it needs to target.
3. If diagnostics show conflict is mild or infrequent, Phase 3 is
   **deferred, not cancelled** — keep the diagnostics running in the
   background (e.g. as part of routine trajectory logging) and revisit
   the decision as the pipeline is used on more targets, rather than
   treating a clean first read as a permanent all-clear.

---

## Phase 3 — Prior-compatibility mechanism

Gated by the three-part decision above. Begin active implementation only
when Phase 2's diagnostics indicate an active corrective term is needed
now; otherwise this phase stays deferred, with diagnostics still running.

- [ ] Implement a NOS-style KL-divergence penalty first (smaller
      architectural blast radius than reopening the denoiser's
      `stop_gradient` boundary)
- [ ] Adapt it to coordinate space — EDM step distributions are Gaussian,
      so a closed-form KL between guided and unguided step distributions
      is likely tractable, not just a heuristic approximation
- [ ] Resolve where it slots into the Phase 1 algorithm: separate additive
      term, PCGrad-projected like the other auxiliary objectives, or a
      post-merge correction to `delta`. This is an open design question,
      not an implementation detail.
- [ ] Only if the KL-style penalty proves insufficient, consider the
      Gradient Guidance paper's forward-prediction / Tweedie-formula
      restructuring (differentiating guidance back through the score
      network `s_theta` itself, removing the current `stop_gradient` on
      `x0_hat`). Its formal subspace-preservation guarantee (Theorem 1)
      is proven only for linear rewards; our actual losses (`L_iptm`,
      `L_ipae`, AbLang2 naturalness) are nonlinear, so treat any adoption
      of this route as heuristic, not as inheriting the paper's proof.

---

## Outer loop — search and selection

From `guidance_search_summary.md`. Can proceed in parallel with the inner
loop phases above; the two are largely decoupled implementation efforts.

- [ ] Demote `edit_budgeted_greedy_descent` from main search engine to
      optional shallow late-stage polish only
- [ ] Decide `K` (particle count) against an actual compute budget — this
      multiplies per-outer-iteration cost directly, not a default to guess
- [ ] Implement `K` independent guided-diffusion trajectories per outer
      iteration
- [ ] Implement proposal-level scoring (cheap intermediate reward)
- [ ] Audit that proxy reward against final refolded metrics on a modest
      sample before trusting it as a resampling criterion (same discipline
      as the Phase 2 gate above; this is `guidance_search_summary.md`
      §15.1's own requirement)
- [ ] Implement soft/hard resampling (soft early rounds, hard later —
      search-axis schedule, kept strictly separate from the `sigma`-axis
      schedules in Phase 1)
- [ ] Log particle-degeneracy diagnostics: ancestor identity, unique
      surviving ancestor count, effective sample size (ESS)
- [ ] First-pass mitigations: at most one resampling round, keep a
      no-resample baseline arm for comparison, optional cap on
      children-per-parent
- [ ] Refold only survivors (cost control — refolding is the expensive
      step)
- [ ] Final ranking: ipSAE + refolded ipTM + RMSD/pose filter + interface
      contacts (H-bonds, salt bridges)
