# OpenDDE: current status and next steps

Project handoff updated **2026-10-03**. The
[numerical investigation](numerical_precision_validation_summary.md) records the
reviewed October 2 synthetic results. The [detailed project record](P17_JN1.md)
preserves historical experiments and implementation decisions. Older plans there
are historical, not additional pending launch instructions.

The [October 3 pose-constraint literature handoff for Claude](P17_JN1.md#18-pose-constraint-literature-handoff-for-claude)
reviews Task Space Regions, TrajOpt, robust losses, and SVD rotation gradients.
It records conceptual distinctions and open synthetic-geometry checks; it adds
no implementation change or model-validation result.

## Objective and baseline comparison

The two baseline complexes contain the **same original P17 binder**. The target
changes from Alpha RBD to JN.1 RBD. In the current validator, “WT” means
**unmodified P17**, not an ancestral viral target. The project objective is to
recover binding to JN.1; the computational objectives discussed here are higher
predicted interface confidence and lower target-aligned binder pose RMSD to the
intended arrangement, with valid protein geometry. These metrics alone do not
establish binding or affinity.

The Alpha complex is the positive reference. The local project record identifies
`P17_Alpha.pdb` as experimental PDB 8GZ5. `P17_JN1.pdb` is a **modeled reference**:
the Alpha binding arrangement transferred onto JN.1 and relaxed. It is not an
experimentally determined P17–JN.1 bound structure. Therefore, its pose RMSD
measures agreement with an assumed arrangement, not error against a known JN.1
experimental structure.

The historical native OpenDDE comparison used the original P17 in both cases,
three seeds, and no MSA or structural template:

| Recorded metric, range across seeds | P17 + Alpha | P17 + JN.1 |
|---|---:|---:|
| ipTM | 0.9006–0.9017 | 0.2225–0.5794 |
| ipSAE minimum, PAE cutoff 12 Å | 0.7949–0.7954 | 0–0.1629 |
| Target-aligned binder pose RMSD | 1.98–2.96 Å | 22.37–57.56 Å |
| Target Cα RMSD after alignment | 0.66–0.74 Å | 1.84–1.87 Å |

Sources: [confidence results](../results/p17_alpha_vs_jn1_native_opendde/comparison.csv)
and [pose/ipSAE results](../results/p17_alpha_vs_jn1_native_opendde/rmsd_ipsae.csv).
These historical results motivate the question; they do not validate the latest
implementation. Higher confidence can accompany a different pose, and a lower
RMSD to a modeled reference does not by itself imply better binding.

**Pose RMSD uses a target-derived alignment:** align the predicted target to the
reference target, apply that same transform to the predicted binder, then compare
binder Cα coordinates. Independently aligning the binder measures its internal
shape and removes the placement error of interest. Unaligned maximum coordinate
displacement is a different measurement and must not be called pose RMSD.

## Why the tests were needed

The original question was the pose/RMSD discrepancy. Subsequent audits revealed
that some predictions also had invalid within-chain geometry. A misplaced but
well-formed binder and a malformed protein are different failures. Large pose
RMSD alone does not diagnose a numerical bug or identify its cause.

| Test layer | Question it addresses | What it does not establish |
|---|---|---|
| Historical Alpha/JN.1 comparison | How do confidence and predicted placement differ for the same binder against the two targets? | The cause of the discrepancy or current implementation correctness. |
| Structure and mapping audits | Are chains geometrically valid, and do raw coordinates agree with exported/mapped coordinates? | Correct docking pose or binding. |
| Synthetic numerical checks | Do generic attention calculations agree with an independent reference, and how do precision, compilation and process restarts affect them? | Full-model geometry, pose recovery, gradients or memory requirements. |
| Current fixed-input forward controls | Do the updated model paths produce valid, consistently mapped structures, and how repeatable are their outputs? | Improved confidence, reduced pose RMSD or successful binder redesign. |

The corrected H200 synthetic archive completed **360/360 workers** and passed
**14,760/14,760 independently recomputed FP32 numerical checks**. All within-process
comparisons were identical, but small differences persisted across fresh
processes, including on the same GPU. See the [independent review](../results/synthetic_end_to_end_review_20261002_220213/REVIEW.md).
This supports the tested synthetic calculations; it neither proves nor disproves
that numerical effects caused the earlier full-model structure failures.

Thus the tests so far establish prerequisites for interpreting confidence and
RMSD. They have **not demonstrated confidence increasing and pose RMSD decreasing**
for the current workflow. Full-model structural validity and the biological
objective remain separate, unresolved questions.

## Current finding

**The fixed-input forward controls after the numerical fixes completed and
passed.** All 10 workers completed, all 20 outputs passed independently
recomputed backbone checks, and all 16 JAX outputs passed raw-to-mapped
coordinate checks. Stable aggregation made both within-worker repeats bitwise
identical in every tested JAX worker; original aggregation did so in none. See
the [independent review](../results/forward_p17_review_20261003/REVIEW.md) of
archive `forward_p17.tar.gz` (run `p17_forward_validation_20261002_234527_1459862`).

| Control group | Workers | Outputs | Geometry | Mapping | Exact repeats |
|---|---:|---:|---:|---:|---:|
| JAX, original aggregation | 4 | 8 | 8/8 | 8/8 | 0/4 |
| JAX, stable aggregation | 4 | 8 | 8/8 | 8/8 | 4/4 |
| Native OpenDDE / Torch | 2 | 4 | 4/4 | not applicable | 2/2 |

Two consequences for the next experiment. **Stable aggregation is the path for
subsequent controlled work**, and the earlier geometry failures did not
reproduce at 64 sampling steps: median adjacent-Cα and N–Cα distances were
3.78 Å and 1.46 Å, against the 14–20 Å medians in the audited pose archive.
Target fit after alignment was 1.82–1.87 Å, inside the diagnostic gate's
provisional 3 Å limit that the earlier run missed at roughly 16 Å. Recorded JAX
process-lifetime peak allocation was 13.66–14.43 GiB; a warm 64-step forward
took about 2.4 s, against 50 s for the first compiled call.

This validates the stated forward controls only. Reference-aligned pose RMSD was
**17.51–52.85 Å** across all 20 outputs, so this batch does not demonstrate
agreement with the assumed reference pose. Stable Mosaic and direct JAX outputs
were not mutually identical: their raw arrays differ by construction (3215×3
versus 2417×3) and their shared atom37 coordinates, PAE and pLDDT differed for
both seeds. Fresh-process reproducibility, backward correctness, fold accuracy
and the paired Alpha control remain open.

The earlier WT control batch, from the archived code **before** these fixes, had
12 successful workers and 24 predictions:

| Forward path | Backbone checks at 8 steps | Backbone checks at 64 steps |
|---|---:|---:|
| Mosaic adapter | 0/4 passed | 0/4 passed |
| Direct JAX | 0/4 passed | 4/4 passed |
| Native OpenDDE / Torch | 0/4 passed | 4/4 passed |

The 64-step Mosaic failures were one or two short peptide C–N bonds per
prediction (0.587–0.988 Å), rather than the gross distortion seen at eight steps.
Raw and mapped JAX backbone coordinates agreed. Native same-seed repeats were
bitwise identical; both JAX paths varied. Those Mosaic C–N failures and the JAX
repeat variability did not recur in the post-fix batch above. **Eight-step
predictions failed on every path and have not been retested since the fixes**,
so 64 steps is the only validated sampling budget. Backbone checks alone do not
establish fold accuracy.

Earlier search and pose-validation batches completed, but their confidence gains
cannot establish a search-policy advantage or biological improvement: later
structure audits exposed invalid geometry. The diagnostic gate correctly stopped
its dependent stages after target-fit and repeat-adjusted influence checks failed.

## Implemented fixes and verification

- Corrected structural padding indices and representative/frame metadata.
- Masked padded atoms and empty structural tokens in attention; excluded padding
  from coordinate centering, diffusion noise and updates.
- Added a stable averaging kernel, with the original kernel retained as a control.
- Updated atom-template cache schema to 2 and prepared it once on CPU before
  launching GPU workers.
- Wired the two new dependency patches into the validation and main pose/search
  launchers, alongside the three existing OpenDDE patches.

The recorded local verification is **73 focused CPU tests, 14 small GPU tests,
and one separate real CPU featurizer comparison**, plus lint, shell syntax and
launcher dry-run checks. The GPU tests used an RTX 4090 and small kernels;
these were not full-model predictions. The featurizer comparison loaded no
model checkpoint. See [the detailed numerical-fix record](P17_JN1.md#1713-numerical-fixes-and-the-next-forward-only-cluster-control--2026-10-01).

A repeatability defect was reproduced in the original averaging kernel and
removed in the tested small kernels. Its contribution to whole-model variation
and the structure errors remains unmeasured.

## Next cluster run: the gated pose experiment

The forward controls that previously gated this work have now completed and
passed, so the next run is the real pose experiment. On the eight-GPU H200
node, from the updated checkout:

```bash
cd /storage/frank/mosaic
bash examples/run_p17_pose_experiment_cluster.sh --dry-run
bash examples/run_p17_pose_experiment_cluster.sh
```

That single script does everything: the environment work a fresh machine needs,
then the gated workflow in `p17_pose_experiment.py` at the settings the forward
review supports. It pins two settings that differ from that script's own
defaults — **64 sampling steps** rather than 8, because every path failed the
backbone checks at 8 steps and the gate now requires backbone plausibility; and
**stable aggregation** through `MOSAIC_OPENDDE_AGGREGATION`, read by the
dependency patch at JAX trace time and recorded in each worker's `config.json`.
Before any GPU worker starts it performs:

1. **Preflight.** Confirms JAX sees every requested device, reports each device
   kind and memory limit, records jax/jaxlib/numpy/jopendde/jablang/torch
   versions, and checks the ABAG checkpoint, the CCD components file and the
   reference PDB. It refuses to launch onto requested GPUs that already hold
   another process, because this workflow preallocates 90% of each device. The
   occupancy check is scoped to the requested devices and excludes the
   preflight's own CUDA context, so neither an unrelated job on an unrequested
   GPU nor the check itself blocks a valid launch. If `nvidia-smi` is missing
   or fails, occupancy is unknown and that is a problem rather than a warning,
   so a passing preflight does not imply a check that never ran;
   `--allow-busy-gpus` downgrades it. Devices default to an inherited
   `CUDA_VISIBLE_DEVICES` when one is set, and a `--devices` value that
   disagrees with an inherited allocation is refused rather than overwritten.
   Evidence is written to `results/p17_pose_prep_*/preflight.json`.
2. **Dependency patches.** Applies all five OpenDDE patches before the caches
   are built. `p17_pose_experiment.py` applies the same set again when it
   starts, so they run twice per launch; they are idempotent and verify their
   expected source before writing, and the second pass reports
   "already patched".
3. **Shared caches, once, on CPU with seeded host RNG.** The schema-2
   atom-template cache and the AbLang2 paired checkpoint are both first-use
   artifacts. Stages run sequentially with at most eight workers at once — two
   diagnostic, then eight search, then eight held-out shards — so the workers
   within a stage would otherwise race to create or download them. The
   template-cache filename embeds the JOpenDDE build id,
   so a different environment rebuilds it rather than reusing this checkout's
   copy; that build can take a few minutes.

A dry run reports preflight problems without aborting, so the plan can be
previewed on any machine; a real launch stops on them. `--allow-busy-gpus`,
`--skip-prep` and `--skip-forward-check` relax those checks individually, and
`--devices`, `--search-seeds`, `--weight-pose`, `--pose-margin`, the three call
ceilings, `--aggregation` and `--output-dir` expose the run parameters.

The gate is what makes this ordering safe: it requires `proposal_influence` to
be positively `True` for both diagnostic seeds, alongside
`repeat_noise_resolvable`, `interpretable_target_fit`,
`predicted_backbone_plausible`, `same_coordinate_reporting` and
`paired_pose_consistent`. If pose influence is inconclusive or merely not
demonstrated, the search stage never launches. Arms B and D therefore run only
when the pose gradient has been shown to move proposals above the repeat
threshold.

This wraps the existing `examples/p17_pose_experiment.py` workflow and pins the
configuration the forward review supports. It changes two of that script's own
defaults, deliberately:

- **64 sampling steps instead of 8.** Eight-step predictions failed the backbone
  checks on every path and have not been retested since the fixes. The schema-2
  diagnostic gate requires predicted-backbone plausibility, so eight steps would
  risk failing the gate for reasons unrelated to pose.
- **Stable aggregation pinned** through `MOSAIC_OPENDDE_AGGREGATION=stable`,
  which the dependency patch reads at JAX trace time. The variable is now also
  recorded in each worker's `config.json` environment block, so the kernel used
  is auditable per run.

Stages, budgets and gates are otherwise unchanged: `diagnostic/` (two workers,
proposal-model seeds 0/1), then `search/` (arms A neither, B guidance,
C retention, D both; two search seeds each, eight workers), then `heldout/`
(WT plus archived winners on structural seeds 101/102/103, sharded). Held-out
rescoring inherits each search run's archived sampling steps and precision.
Both diagnostic reports must pass every required check before the search stage
launches, and every search worker must succeed before held-out rescoring. The
3 Å target-fit limit, the proposal-influence rule and the geometry thresholds
are not relaxed. The launcher refuses to start without completed forward-control
evidence in the checkout unless `--skip-forward-check` is given.

**Why the gate may now be passable.** It previously failed on two checks. Target
fit was roughly 16 Å against the 3 Å limit; the post-fix forward batch measured
1.82–1.87 Å. The influence rule required proposal total variation above three
times the repeat total variation, which was 0.38–0.56 under the original kernel
and therefore demanded an unreachable value above 1. The diagnostic repeat is a
repeated pose-on **gradient** evaluation in one process; stable aggregation
removed the analogous forward repeat variation, but gradient-path repeatability
was not established by the forward batch. Whether the repeat term collapses is
exactly what this diagnostic reports. Neither threshold was changed to make this
more likely.

**Cost.** A warm 64-step forward prediction took about 2.4 s in the forward
batch, so each search worker's 32 scored sequences across two selection seeds is
roughly 64 predictions. Full-path gradient calls dominate instead: the audited
pose archive recorded 52.69/52.65 GiB JAX peaks with a 125.85 GiB preallocated
pool, against 13.66–14.43 GiB for these forward workers. That archived diagnostic
worker completed three gradient calls and eight forward predictions in 169 s at
**eight** sampling steps.

**Hypothesis, not a measurement:** raising sampling steps is expected to cost
runtime rather than memory. The mechanism is real — `jopendde`'s diffusion
sampler wraps its scan body in `jax.checkpoint` (`jopendde/diffusion.py:369`),
so reverse mode retains only the per-step carry, coordinates of shape atoms×3 or
about 39 KiB per step, and rematerializes each step's activations during the
backward pass. That accounts for a few MiB of additional carry at 64 steps and
gives a reason to expect the step-independent trunk activations to keep
dominating the peak. It does **not** measure the peak: checkpointing bounds the
carry, not the rematerialization working set, the allocator's pool growth or
fragmentation — and fragmentation, not a tensor size, is what §17.9 identified
behind the earlier OOMs. Gradient-stage runtime and memory at 64 steps are
unmeasured; this run is what would establish them.

Pose weight and retention margin are exposed as `--weight-pose` and
`--pose-margin`. Raising the weight does not guarantee proportionally stronger
guidance: the composite gradient is clipped and the proposal temperature is
recalibrated to a target entropy, so a larger coefficient can be partly
absorbed, and retention still ranks on raw mean ipSAE with pose entering arms C
and D as a feasibility ordering only. Read the diagnostic's measured influence
rather than assuming the weight transferred.

A passing gate establishes interpretable measurement and detectable proposal
influence. It does not establish improved pose, binding or affinity. The
reference remains modeled, so pose RMSD measures agreement with an assumed
arrangement. Fresh-process reproducibility, full backward correctness and the
paired Alpha control remain open after this run.

### Repeating the forward control

```bash
bash examples/run_p17_forward_validation.sh
```

The default wrapper uses the existing `P17_JN1.pdb`, containing unmodified P17
and the JN.1 target. Its reference provides sequences and geometry checks; the
reference coordinates do not constrain the predicted pose or act as a structural
template. The wrapper runs **10 workers and 20 forward predictions**, with at
most eight workers at once. It compares Mosaic and direct JAX with original and
stable aggregation, plus native Torch. Each uses two seeds and two repeats;
the preset uses BF16, 64 sampling steps and four recycles. No search follows
automatically. Use `--dry-run` to inspect the plan.

Both JAX aggregation arms include the padding and metadata fixes. The
original-kernel arm changes only aggregation; it is not a full replay of the
archived implementation. Each aggregation control uses a fresh process. The two
identical-input repeats are within each worker; different aggregation arms are
not matching fresh-process replicates.

On a first launch, CPU template-cache preparation can take a few minutes before
GPU workers start; its log is `logs/template_cache.log`. Paths follow the checkout,
including `/storage/frank/mosaic` on the cluster.

Review `results/p17_forward_validation_*/tables/geometry.csv` together with
worker reports, raw/mapped CIFs, NPZ arrays, metadata, source hashes and memory
logs. Maximum raw coordinate displacement is unaligned and is not RMSD.
Allocator peak memory is a process-lifetime high-water mark, not an isolated
per-call measurement. This JN.1-only wrapper does not include a paired Alpha
control or reproduce the complete historical confidence/pose comparison above.

## Model and policy context

The retained workflow uses frozen OpenDDE and AbLang2. Gradient-based proposals
and separate forward scoring are implemented. Independent trajectories and local
population competition are experimental policies; current results do not establish
which is better. Confidence, pose error and affinity are distinct quantities.
The earlier empirical record also found that lower composite loss could accompany
worse experimental agreement; see [OpenDDE validation history](opendde_validation_history.md).

## Documentation and implementation map

| Item | Role |
|---|---|
| [Setup](../SETUP.md) | Environment checks and OpenDDE documentation entry point |
| [Detailed project record](P17_JN1.md) | Historical results, implementation decisions and references |
| [Literature review](protein_search_policy_review.md) | Search-policy papers and limits of the review |
| [Earlier empirical evidence](opendde_validation_history.md) | OpenDDE findings retained from the mixed experiment log |
| [Cluster launcher](../examples/run_p17_pose_experiment_cluster.sh) | Single entry point: preflight, patches, caches, then the gated run at the validated settings |
| [Pose geometry tests](../tests/test_pose_rmsd_geometry.py) | CPU orientation/shape separation and Kabsch alignment-derivative checks |
| [Pose experiment](../examples/p17_pose_experiment.py) | Stage barriers, bounded GPU workers and comparison summaries |
| [Forward review](../results/forward_p17_review_20261003/REVIEW.md) | Independent verification of the post-fix forward controls |
| [Default forward launcher](../examples/run_p17_forward_validation.sh) | Fixed-input JN.1 validation with geometry/mapping exit checks |
| [Numerical launcher](../examples/run_p17_numerical_validation.sh) | Underlying forward-only validation preset |
| [WT validator](../examples/p17_wt_validation.py) | Forward-path controls, artifact export and repeat comparisons |
| [Structure audit](../examples/p17_structure_audit.py) | Named-backbone mapping and geometry checks |
| [Numerical helpers](../src/mosaic/opendde_numerics.py) / [padding helpers](../src/mosaic/opendde_padding.py) | Averaging and masking implementation |
| [Test notes](../tests/README.md) | Checkpoint-free regression checks |
| [Presentation figure](figures/p17_optimization_slide.pdf) | Conceptual model/search flow, not evidence of validation |
