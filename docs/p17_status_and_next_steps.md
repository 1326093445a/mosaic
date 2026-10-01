# OpenDDE: current status and next steps

Updated **2026-10-01**. This is the current handoff. The
[detailed project record](p17_jn1_redesign.md) preserves historical experiments,
implementation decisions and paper references. Older plans there are historical,
not additional pending launch instructions.

## Current finding

**Local numerical fixes are implemented and tested; full-model cluster
validation after those fixes is still pending.** Successful execution alone
has not established correct structures, repeatability, pose recovery or affinity.

The completed WT control batch had 12 successful workers and 24 predictions:

| Forward path | Backbone checks at 8 steps | Backbone checks at 64 steps |
|---|---:|---:|
| Mosaic adapter | 0/4 passed | 0/4 passed |
| Direct JAX | 0/4 passed | 4/4 passed |
| Native OpenDDE / Torch | 0/4 passed | 4/4 passed |

The 64-step Mosaic failures were one or two short peptide C–N bonds per
prediction (0.587–0.988 Å), rather than the gross distortion seen at eight steps.
Raw and mapped JAX backbone coordinates agreed. Native same-seed repeats were
bitwise identical; both JAX paths varied. These results describe the archived
code before the latest fixes. Backbone checks alone do not establish fold accuracy.

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
model checkpoint. See [the detailed numerical-fix record](p17_jn1_redesign.md#1713-numerical-fixes-and-the-next-forward-only-cluster-control--2026-10-01).

A repeatability defect was reproduced in the original averaging kernel and
removed in the tested small kernels. Its contribution to whole-model variation
and the structure errors remains unmeasured.

## Next cluster validation

From the updated repository checkout:

```bash
bash examples/run_p17_numerical_validation.sh --devices 0,1,2,3,4,5,6,7
```

The existing launcher runs **10 workers and 20 forward predictions**, with at
most eight workers at once. It compares Mosaic and direct JAX with original and
stable aggregation, plus native Torch. Each uses two seeds and two repeats;
the preset uses BF16, 64 sampling steps and four recycles. No search follows
automatically. Use `--dry-run` to inspect the plan.

Both JAX aggregation arms include the new padding and metadata fixes. The
original-kernel arm changes only aggregation; it is not a full replay of the
archived implementation. Each aggregation control uses a fresh process.

On a first launch, CPU template-cache preparation can take a few minutes before
GPU workers start; its log is `logs/template_cache.log`. Paths follow the checkout,
including `/storage/frank/mosaic` on the cluster.

Review the resulting `results/p17_wt_validation_*/tables/geometry.csv` together
with worker reports, raw/mapped CIFs, NPZ arrays, metadata, source hashes and
memory logs. Check geometry, raw-to-mapped agreement and same-seed repeats for
each path. Maximum raw coordinate displacement is unaligned and is not RMSD.
Allocator peak memory is a process-lifetime high-water mark, not an isolated
per-call measurement.

Whole-model repeatability, full backward behavior and H200 runtime/memory after
these fixes remain unverified. Keep subsequent pose diagnostics and search
comparisons pending until the forward results have been reviewed. Existing
geometry thresholds and experiment gates have not been relaxed.

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
| [Detailed project record](p17_jn1_redesign.md) | Historical results, implementation decisions and references |
| [Literature review](protein_search_policy_review.md) | Search-policy papers and limits of the review |
| [Earlier empirical evidence](opendde_validation_history.md) | OpenDDE findings retained from the mixed experiment log |
| [Numerical launcher](../examples/run_p17_numerical_validation.sh) | Current forward-only validation preset |
| [WT validator](../examples/p17_wt_validation.py) | Forward-path controls, artifact export and repeat comparisons |
| [Structure audit](../examples/p17_structure_audit.py) | Named-backbone mapping and geometry checks |
| [Numerical helpers](../src/mosaic/opendde_numerics.py) / [padding helpers](../src/mosaic/opendde_padding.py) | Averaging and masking implementation |
| [Test notes](../tests/README.md) | Checkpoint-free regression checks |
| [Presentation figure](figures/p17_optimization_slide.pdf) | Conceptual model/search flow, not evidence of validation |
