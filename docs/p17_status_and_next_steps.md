# P17 optimization: status and next steps

Updated **2026-09-23**. Start here for the current handoff; the
[full project record](p17_jn1_redesign.md) retains the biology, infrastructure,
literature discussion, and Claude/Codex reviews.

## Current status

**The first search harness is implemented and tested on synthetic objectives.
The real OpenDDE + AbLang2 search has not yet passed a GPU smoke run.**
Full confidence-aware proposal gradients are now the default for both policies.
The reviewed memory-logging issues have been fixed and covered by failure tests.
We have not established that either search policy improves P17 or restores binding.

## Goal and agreed approach

Improve P17's predicted interface confidence against JN.1 while retaining the
framework and restricting edits to the allowed CDR positions. The current hard
WT-relative edit cap is 5; increasing it to 7 remains undecided.

![Presentation overview of the optimization loop](figures/p17_optimization_slide.svg)

1. **Guide:** frozen full OpenDDE supplies contact, coordinate-RMSD, ipTM,
   bidirectional interface-PAE and pTMEnergy losses. Combine these with frozen
   AbLang2 and the edit penalty; differentiate with respect to sequence inputs.
   The previous proxy remains available with `--proposal-path distogram`.
2. **Propose:** sample feasible CDR edits using gradient deltas and controlled
   proposal entropy. Allow reversions, replacements, and position exchanges.
3. **Score:** full OpenDDE runs forward-only. The provisional retention score is
   mean per-seed directional-min ipSAE across a fixed set of structural seeds.
4. **Select and repeat:** confidence affects retention and therefore future parents.

No additional model training is needed. ipSAE is a structural-confidence proxy,
not an affinity measurement. The separate affinity-model effort is not required
for this first comparison.

## Search policies

| Policy | Current role | Retention |
|---|---|---|
| Independent searches | First comparison arm | Each trajectory compares offspring with its own parent |
| Population with local competition | Preferred hypothesis; second comparison arm | Offspring competes with a nearby sequence; retain alternatives and protect the best active candidate |
| Complete WT-relative edit-combination sampling | Deferred follow-up | Reconsider whole edit allocations if the first two policies repeatedly settle on the same position sets |

The first two arms share models, proposal machinery, constraints and scoring.
Both can accept worse confidence according to a shared acceptance temperature;
population initialization also fills duplicate slots with alternatives. These
are proposed optimization heuristics, not exact MH samplers or reproductions
of a published algorithm. Equal call ceilings do not imply equal GPU time.

## What exists

| Artifact | Purpose |
|---|---|
| [Shared harness](../src/mosaic/search.py) | Backend-independent policy logic, feasibility checks, caching, budgets and event callbacks |
| [P17 runner](../examples/p17_confidence_search.py) | Full-gradient proposal objective, separate forward-only retention scoring, run metadata and memory snapshots |
| [Launcher](../examples/run_p17_confidence_search.sh) | Applies the existing OpenDDE outer-product, structural-token and bf16 patches |
| [Tests](../tests/test_confidence_search.py) | Policy behavior, constraints, score aggregation, reproducibility and memory-counter checks |
| [Presentation figure](figures/p17_optimization_slide.pdf) / [detailed figure](figures/p17_optimization_flow.pdf) | Model/search flow for slides or technical discussion |

Runs write `config.json`, `events.jsonl`, `memory.jsonl`, `candidates.csv`, and
`summary.json`. Search seeds and model/scoring seeds are separate. Repeated
sequences reuse evaluations under the fixed model-seed configuration.
The previous MCMC optimizer and its runners remain separate. The existing shared
loss builder accepts optional confidence terms without changing old callers.

Full-path defaults match the existing full-gradient example: contact 0.5,
coordinate pose 1.0, ipTM 0.025, interface PAE 0.05 per direction, pTMEnergy
0.025, AbLang2 0.10, and edit penalty 5.0. Confidence weights and RMSD tolerance
are CLI options. The RMSD tolerance defaults to 0 angstroms; this is a proposal
loss setting, not a retention threshold or a validated optimum. Ordinary global
pTM is not an additional term. The coordinate gradient is not detached.

Retention remains mean per-seed ipSAE-min. There is no RMSD retention gate.
`proposal_loss` replaces the runner's old `cheap_loss` output label; per-term
proposal diagnostics are logged in `events.jsonl`. The summary separately counts
full-gradient calls and standalone confidence predictions; each full-gradient
call includes a full forward and backward pass.

## What has actually been checked

- Before memory instrumentation: 35 targeted tests passed, comprising 25 new
  harness/scoring tests and 10 existing optimizer/loss tests.
- After instrumentation: the 27 tests in `test_confidence_search.py` passed
  again during Codex review, including memory-counter access and monotonicity.
  Ruff passed. These checks did not load or optimize with the real models.
- After full-gradient integration: **53 targeted tests passed; one GPU-counter
  test was skipped on the CPU backend**. Tests exercise each confidence term's
  sequence gradient, JIT integration through the shared loss builder, legacy
  compatibility, argument validation and failure logging. Ruff and CLI checks pass.
- Successful-call output conversion synchronizes results before memory snapshots.
- Both callback wrappers now include input creation inside the protected block,
  record sequence identity and success/error status, and preserve a model error
  when diagnostic logging also fails. No model checkpoints were loaded by these tests.

The GPU counter test is not a real-model smoke test. Peak memory is a running
allocator high-water mark, including previous work; a new shape does not reset it.
The two biological control cases with three prediction seeds each do not provide
six independent binding examples or validate a general candidate-ranking metric.

## Next moves, in order

1. **Run a minimal smoke test on the cluster for each policy.** Use the commands
   below, each with a fresh output directory. Inspect finite gradients/scores,
   retained-parent decisions, CDR/edit-cap compliance, memory and actual call cost.
2. **Run a small comparison pilot after the smoke checks pass.** Keep model and
   scoring settings fixed across arms. Use measured costs to match compute, and
   repeat search seeds to estimate variability before sizing a larger campaign.
3. **Evaluate selected candidates separately.** Reserve structural seeds not used
   for selection. Confidence improvements alone do not establish binding recovery.

Minimal smoke commands, from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 bash examples/run_p17_confidence_search.sh \
  --policy independent --proposal-path full --width 2 --seed 0 \
  --max-score-calls 2 --max-gradient-calls 1 --max-proposals 2 \
  --selection-seeds 0 --output-dir results/p17_smoke_independent

CUDA_VISIBLE_DEVICES=0 bash examples/run_p17_confidence_search.sh \
  --policy population --proposal-path full --width 2 --seed 0 \
  --max-score-calls 2 --max-gradient-calls 1 --max-proposals 2 \
  --selection-seeds 0 --output-dir results/p17_smoke_population
```

Each arm covers WT scoring, one parent gradient and at most one new candidate
score. This checks execution and logging; it cannot demonstrate policy superiority.

## Deferred decisions and reference map

Hold the full proposal objective and its weights fixed across the first policy
comparison. Check memory, finite gradients and each logged loss component before
scaling the pilot. A separate full-versus-distogram ablation can follow; do not
change the proposal objective between policy arms. Germinal-inspired gradient
balancing, an explicit pose-retention condition, alternative ranking metrics,
surrogate/BO evaluation allocation, batching, and policy 3 remain follow-ups.
Raw PAE persistence, automatic resume and a held-out rescoring runner are not
implemented in this version.

The [full handoff](p17_jn1_redesign.md) contains the **26-paper index plus a pinned
BindCraft2 source reference** in §8.13; policy rationale in §10; Claude's response
in §11; initial implementation in §12; memory instrumentation in §13; and the
latest full-gradient integration and logging fixes in §14.
The [literature survey](protein_search_policy_review.md) records broader context
and limits of the paper review. The main influences remain EvoProtGrad/PPDE,
AdaLead, ME-GIDE, PEX, LaMBO-2, BADASS and Germinal; their ideas are adaptations
here, not evidence that the current P17 implementation works.
