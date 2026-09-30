# P17 optimization: status and next steps

Updated **2026-09-29**. Start here for the current handoff; the
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
| [Multi-GPU launcher](../examples/run_p17_confidence_search_multi_gpu.sh) | Runs both policies on separate GPUs, with smoke/pilot presets, dry-run preview and worker exit codes |
| [Tests](../tests/test_confidence_search.py) | Policy behavior, constraints, score aggregation, reproducibility and memory-counter checks |
| [Presentation figure](figures/p17_optimization_slide.pdf) / [detailed figure](figures/p17_optimization_flow.pdf) | Model/search flow for slides or technical discussion |

Runs write `config.json`, `summary.json`, `logs/events.jsonl`,
`logs/memory.jsonl`, `tables/candidates.csv` and `tables/predictions.csv`,
plus PDB structures and compressed confidence arrays. See the layout below. Search seeds and model/scoring seeds are separate. Repeated
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
proposal diagnostics are logged in `logs/events.jsonl`. The summary separately counts
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

## H200 multi-GPU launcher

The launcher resolves the checkout and `.venv` relative to its own location,
so `/storage/frank/mosaic` works without replacing paths in the script. Run
inside an existing GPU allocation with the cluster environment and model assets
available:

```bash
cd /storage/frank/mosaic
bash examples/run_p17_confidence_search_multi_gpu.sh --dry-run
bash examples/run_p17_confidence_search_multi_gpu.sh --mode smoke --devices 0,1
# After inspecting the smoke results:
bash examples/run_p17_confidence_search_multi_gpu.sh
```

The default pilot runs search seeds 0–3 for each policy: eight processes, one
per GPU, width 4, up to 32 unique scored sequences, 32 gradient calls and 320
proposals per run. These are provisional pilot budgets. Both policies keep
proposal-model seed 0, selection seed 0, eight sampling steps, edit cap 5 and
the full proposal loss defaults. Four search seeds are exploratory, not evidence
of a statistically established policy advantage.

GPU IDs default to the caller's `CUDA_VISIBLE_DEVICES`, or 0–7 if unset;
`--devices` overrides this list. With fewer GPUs, jobs run in bounded batches.
`--num-seeds N` changes the search seed count per policy. This distributes runs,
not an individual model call, and explicitly selects the JAX CUDA backend.
Patches run once before workers start. A fresh batch directory holds each run's
outputs under `runs/<policy>/seed_N/`, console logs under `logs/`,
`logs/patches.log`, `commands.sh` and `tables/status.tsv`.
The launcher waits for all runs and exits nonzero if any fail. `--dry-run` only
prints the plan; it does not patch dependencies, create outputs or load models.
The launcher has been checked without real model execution; H200 fit and
performance still require the smoke run.

## Saved structures and output layout (version 2)

Every unique candidate scored by full OpenDDE, including WT and rejected
candidates, saves one protein-heavy-atom PDB per selection seed. These are the
same predictions used for scoring; export does not rerun the model. Large
logits are excluded from the host transfer. The saved NPZ contains mean PAE,
pLDDT (0–1), CA and atom37 coordinates, atom masks, sequence, chain IDs and
residue numbering. PDB B-factors contain pLDDT on the 0–100 scale. PDB coordinates
use standard text precision; NPZ retains numeric coordinate arrays.

```text
batch/
├── README.md
├── commands.sh
├── logs/                        # patch and per-worker console logs
├── tables/status.tsv            # worker GPU, PID and exit code
└── runs/
    ├── independent/seed_0/      # also seed_1, seed_2, seed_3
    │   ├── README.md
    │   ├── config.json
    │   ├── summary.json
    │   ├── tables/
    │   │   ├── candidates.csv   # rank, sequence, aggregate score, structure folder
    │   │   └── predictions.csv  # per-seed metrics, chain identities, file paths
    │   ├── logs/               # events.jsonl and memory.jsonl
    │   ├── structures/candidate_00000/seed_0.pdb
    │   ├── confidence/candidate_00000/seed_0.npz
    │   └── best/               # winning candidate PDBs, one per selection seed
    └── population/seed_0/       # same layout for every population run
```

Candidate 0 is WT. Candidate IDs match events and both CSVs. `predictions.csv`
paths are relative to the run directory; its seed column is the structural
selection seed, distinct from the search seed in the enclosing directory.
All scored candidates get structure/confidence folders, not only candidate 0.
The `best/` copies and `summary.json` are written after successful completion.
Prediction files and their index are written incrementally; an interrupted run
can therefore have usable predictions without a final summary or candidates CSV.

This changes the former flat output paths for new runs; existing result folders
are not migrated. The search objectives and budgets are unchanged. Export adds
host transfer and disk I/O, included in scoring time, but no extra model calls.
No gradient-path structures or held-out predictions are generated.

Validation: **43 tests passed, one GPU-memory test skipped on CPU** in
`tests/test_confidence_search.py`. New checks cover PDB round trips (chain,
residue, side-chain coordinates and pLDDT), NPZ contents, JIT payload pruning,
rejection of mismatched/nonfinite predictions, and an entire toy runner with
both scoring seeds and matching prediction counts. These tests do not establish
real-model H200 memory fit or performance.

## Pose review of the completed pilot

RMSD currently guides proposals; it is not a condition for retention. The loss
aligns the predicted target CA atoms to reference target CA atoms, then measures
binder CA displacement using that same transform. Independently aligning the
binder would hide rigid-body pose drift. A low binder-internal RMSD therefore
does not establish a correct binding pose.

Exports now include mmCIF alongside PDB, and `tables/predictions.csv` reports
`binder_pose_rmsd_A`, `target_aligned_rmsd_A` and `binder_internal_rmsd_A` on the
exact scored forward prediction. These remain diagnostics; selection is unchanged.
The reference is `P17_JN1.pdb`, binder chain B and target chain T. Prediction
chain names are recorded in the CSV and may differ from reference chain names.

The downloaded September 29 pilot completed all eight runs but predates structure
export. Its sequences are preserved. To generate new structures for WT and all
eight distinct winners, sync the updated scripts to the H200 checkout and run:

```bash
cd /storage/frank/mosaic
CUDA_VISIBLE_DEVICES=0 bash examples/run_p17_rescore_winners.sh \
  --input results/p17_confidence_pilot_20260929_194857_1770469 \
  --output-dir results/p17_winner_pose_review \
  --seeds 0 1 2
```

For both validation and repeatability in one command, distribute the sequences
across all eight allocated GPUs:

```bash
cd /storage/frank/mosaic
bash examples/run_p17_pose_validation.sh --devices 0,1,2,3,4,5,6,7
```

The launcher defaults to `CUDA_VISIBLE_DEVICES`, or GPUs 0–7 if unset. Use
`--devices` to specify the allocation explicitly; `--device 0` still supports
single-GPU execution. `--input` accepts a batch directory or archive;
`--output-dir` overrides the fresh timestamped output root. `--dry-run` previews
all assignments without patching dependencies or running models.

Global candidate IDs are partitioned by ID modulo GPU count. For the nine
sequences in this pilot, GPU 0 handles WT plus one winner and GPUs 1–7 each
handle one winner. Stage 1 predicts seeds 0, 1, 2 concurrently across GPUs.
After every worker succeeds, stage 2 repeats seed 0 in fresh processes with the
same candidate-to-GPU assignment. Total work remains 27 + 9 = 36 predictions,
not 36 per GPU. Each prediction must fit on its assigned GPU.

Patches run once. Each stage has `shard_N/` folders for structures, confidence
arrays, metadata and local tables. Stage-level `tables/candidates.csv`,
`tables/source_runs.csv` and `tables/predictions.csv` combine the workers; artifact
paths in the combined prediction CSV are relative to that stage directory.
`logs/` contains separate worker logs, and `status.tsv` records stage, shard,
GPU, PID and exit code. A worker failure prevents the repeat stage from starting.
The launcher waits for all active workers before reporting a stage failure.

Validation: 48 tests passed, one GPU-memory test skipped on CPU. The actual
archive dry-run confirms the 27 + 9 split. Disposable fake-worker checks verify
parallel execution without GPU overlap, a barrier between stages, fresh repeat
processes, valid merged artifact paths, and failure propagation. No cluster
predictions were launched by these checks.

These stages are the next diagnostic pass, not proof of policy superiority or
binding. Review their results before sizing additional structural seeds or a
larger policy comparison. Two executions of seed 0 can expose a repeatability
problem but cannot establish its full distribution or cause.

For the standalone `run_p17_rescore_winners.sh` command above, add `--dry-run`
to preview without patches, model loading or output writes. `--input` also accepts the downloaded `.tar.gz` archive directly. Default sampling
steps and cutoffs come from the saved run configuration. This batch is 27
forward predictions on one GPU; it does not run gradients, optimization or
AbLang2 inference. Seed 0 revisits the original selection seed; 1 and 2 were not
used for selection in this pilot. This does not test repeated seed 0 across
fresh processes; use a second fresh output directory for that check.

Outputs include `reference.pdb`, config and summary JSON,
`tables/candidates.csv`, `tables/source_runs.csv`, `tables/predictions.csv`,
`structures/` (CIF/PDB), `confidence/` (NPZ), and `logs/memory.jsonl`.
Candidate IDs are local to this new batch; `source_runs.csv` maps them back to
original runs and candidate IDs. Predictions are saved incrementally. They are
new predictions, not recovery of the original unsaved poses, and may differ.

Validation: 46 confidence-search tests passed, one GPU-memory test skipped on CPU.
Checks include rigid-body alignment invariance, preserved binder displacement,
CIF round trips, deduplicated winner loading from directories/archives, and a toy
end-to-end rescoring run. No real-model rescoring has been launched locally.

## Deferred decisions and reference map

Hold the full proposal objective and its weights fixed across the first policy
comparison. Check memory, finite gradients and each logged loss component before
scaling the pilot. A separate full-versus-distogram ablation can follow; do not
change the proposal objective between policy arms. Germinal-inspired gradient
balancing, an explicit pose-retention condition, alternative ranking metrics,
surrogate/BO evaluation allocation, batching, and policy 3 remain follow-ups.
Automatic resume remains unimplemented. Winner rescoring now supports additional
structural seeds, and raw mean-PAE matrices are saved with each prediction.

The [full handoff](p17_jn1_redesign.md) contains the **26-paper index plus a pinned
BindCraft2 source reference** in §8.13; policy rationale in §10; Claude's response
in §11; initial implementation in §12; memory instrumentation in §13; and the
full-gradient integration and logging fixes in §14; structure export in §15.
The [literature survey](protein_search_policy_review.md) records broader context
and limits of the paper review. The main influences remain EvoProtGrad/PPDE,
AdaLead, ME-GIDE, PEX, LaMBO-2, BADASS and Germinal; their ideas are adaptations
here, not evidence that the current P17 implementation works.
