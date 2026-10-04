# OpenDDE validation history

Consolidated **2026-10-01** from the OpenDDE portions of the earlier mixed
experiment notes. These are historical observations recorded in July 2026,
not new runs or verification of the current implementation. The original notes
remain recoverable through Git history. For the current work, use
[status and next steps](p17_status_and_next_steps.md).

## What the earlier experiments established

The standalone OpenDDE/AbLang2 experiments distinguished execution success,
composite predicted loss, structural confidence and agreement with AlphaSeq
measurements. Those measures did not consistently improve together.

A three-seed discrete-search comparison completed 12 jobs. The best configuration
by composite loss changed with the seed; the original single-seed claim of a
stable policy winner did not replicate. Some sampled fronts lacked entries at
particular edit counts, so coverage was not identical across policies.

The experimental comparison used single-substitution contrast pairs. Only
65 of 166 mutation instances were testable, and generated candidates had no exact
CDR matches in that sweep's experimental table. A nearby measured sequence is
not a direct affinity measurement of a generated candidate.

Deduplicating recurring substitutions changed the interpretation of the apparent
agreement. Across all configurations, the recorded pooled favorable-vote fraction
was 48.4% (386 votes). One configuration reached 61.1% (190 votes), but pairs could
share variants and backgrounds. The reported binomial calculation therefore did
not establish independent-sample significance or a general policy advantage.

## Lower predicted loss did not establish better experimental agreement

Two later initialization comparisons completed without runtime failures. Their
historical aggregate results were:

| Recorded comparison | Mean composite loss at five edits | Pooled favorable experimental votes |
|---|---:|---:|
| Original starting point | 2.768 | 116/190 (61.1%) |
| Sampled initialization | 2.869 | 71/175 (40.6%) |
| Top-k initialization | 2.674 | 81/273 (29.7%) |

The lowest mean predicted loss coincided with the lowest favorable-vote fraction.
These are descriptive results from the earlier record; the vote counts have the
coverage and dependence limitations above. They do not measure affinity for the
complete generated sequences.

The record also corrected an architectural misunderstanding: the full OpenDDE
path can include differentiable coordinate sampling and confidence outputs;
the earlier distogram-only path was an implementation choice. The full-path
experiment was listed as built but unrun in the July notes. That dated status
must not be confused with the later P17 runs in the
[detailed project record](P17_JN1.md).

## Interpretation retained in the current handoff

- Composite loss and confidence are not calibrated affinity measurements.
- Mechanical changes to an optimizer do not establish biological improvement.
- Report experimental coverage and shared evidence instead of treating every
  repeated mutation instance as independent.
- Separate structural validity, numerical repeatability and predictive usefulness.
  The later P17 geometry failures make that distinction especially necessary.

The historical result directories named in the original notes were
`results/hallucination_sweep_discrete_3seed/`,
`results/hallucination_sweep_apgm_sample_mcmc_sg0/`, and
`results/hallucination_sweep_apgm_topk_mcmc_sg0/`.
Their availability is checkout-dependent; this cleanup did not rerun their analyses.
