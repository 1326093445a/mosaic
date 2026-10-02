# Numerical precision investigation: evidence, readiness, and next checks

Updated: **2026-10-02**. This document consolidates the numerical investigation through the reviewed H200 synthetic batch `synthetic_attention_cluster_20261002_094433_601222`.

## 1. How close are we to real inference?

**We have a useful synthetic numerical baseline. We have not established reliable current end-to-end model inference or validated full-model gradients from these tests.** It would be misleading to assign a completion percentage or promise that one more toy test will finish validation.

“Real inference” and “gradient-guided optimization” are different milestones:

| Question | Evidence available | What is still unresolved |
|---|---|---|
| Can inference code execute? | Historical project records contain completed forward predictions. Fixed toy model calls also executed. | Execution alone does not establish correct outputs or validate the latest implementation. |
| Can the numerical comparison infrastructure run on H200? | All 48 workers in the latest synthetic cluster batch completed. | This is a small numerical workload, not a realistic full-model workload. |
| Does generic attention autodiff agree with an independent reference? | FP64/FP32 CPU controls passed; all strict-precision H200 workers passed their existing synthetic controls. | This does not cover every operation, shape, integration path, or full-model derivative. |
| Is BF16 numerically equivalent across execution modes? | Its behavior was measured in several synthetic configurations. Differences depend on precision policy, matrix size, and compilation. | There is no universal BF16 correctness verdict. |
| Is current full-model forward inference validated? | Historical runs exist, but the latest synthetic archive does not contain model predictions. | Current integration, output quality, and representative resource evidence are separate from the synthetic results. |
| Are full-model gradients and production memory validated? | Fixed toy model runs provided limited execution and numerical observations. | Full-model gradient correctness, realistic backward memory, and integration remain unvalidated. |

Ordinary forward inference does not require a validated input-gradient path. Conversely, an apparently reasonable forward result does not certify its derivatives. These milestones should be evaluated separately.

The [October 1 project handoff](p17_status_and_next_steps.md) contains historical forward findings and explicitly notes that execution and backbone checks do not establish overall structure accuracy. Those older results have not been re-reviewed or promoted to a current validation claim here. This document records numerical evidence; it does not provide a pathogen-targeted optimization procedure.

## 2. Why we investigated precision

BF16 was introduced in response to memory pressure in earlier work. That creates two independent questions:

1. Does a configuration fit the available memory?
2. What numerical behavior does that configuration produce?

Finishing a run answers neither gradient correctness nor output quality by itself. A finite gradient, repeatable gradient, and accurate gradient are also different claims.

The code is already Python. JAX versus PyTorch is a choice of numerical/autodiff framework, not Python versus another language. The evidence gathered so far does not justify a blanket claim that JAX is broken or that a PyTorch rewrite would remove the observed numerical effects.

An earlier code inspection also found that a wrapper-level FP32 setting could preserve separately configured BF16 attention calculations. Therefore a top-level precision label is not a complete description of intermediate arithmetic. The synthetic experiments record their precision boundaries explicitly; they do not establish the effective precision of every full-model operation.

## 3. Earlier fixed-toy audit: what was learned

These were tightly bounded numerical fixtures, not realistic structure-quality or production-memory benchmarks.

- The first archived toy backward batch completed its calls and produced finite, nonzero gradients. Its original numerical acceptance checks did not pass consistently. See the [first review](../results/opendde_toy_review/review.json).
- The second audit separated numerical probes and collected **304 model calls**. Forward scalar repetitions were stable, while gradient repetitions and finite-difference comparisons still raised questions. See the [second review](../results/opendde_toy2_review/review.json).
- Setup arrays differed across fresh CPU processes despite a fixed evaluation key. Controlling host-side setup randomness restored matching setup hashes, gradient arrays, observations, and finite-difference tables in the tested CPU runs. See the [fresh-process CPU comparison](../results/opendde_toy_cpu_repro_20261002_073822_787839/comparison.json).
- Restoring repeatability did **not** resolve the derivative-correctness question.

The older finite-difference results must be read with the reporting limitations below. A lack of observed-range overlap is not, by itself, a failed derivative-correctness test.

## 4. Corrections to audit interpretation and reporting

### 4.1 A hardcoded false was not a failed test

The standalone toy audit originally wrote `gradient_validated: false` unconditionally. It meant that the collection procedure did not certify gradient correctness, including when execution succeeded.

The updated schema reports:

- `gradient_validation_status: "not_assessed"`;
- execution completion separately;
- observed repeatability separately;
- descriptive derivative gaps separately.

The reporting change did not fix a derivative or change acceptance thresholds. The audit still has no independently justified full-model correctness criterion. See [the audit implementation](../examples/opendde_toy_numerical_audit.py) and [its regression tests](../tests/test_toy_numerical_audit.py).

### 4.2 Observed-range overlap is not a correctness criterion

When repeated values are identical, the observed ranges collapse to points. A tiny floating-point difference then produces “no overlap” even for a known correct derivative.

In an FP32 identity-function check, autodiff returned `1` and finite differences returned approximately `1.00000024`; the overlap statistic was false. The statistic describes sampled values and does not incorporate finite-difference truncation error or an appropriate numerical tolerance.

### 4.3 A final-scalar resolution check misses internal rounding

The audit's resolution heuristic examines the final FP32 scalar. It cannot rule out quantization inside the computation. Converting an already rounded intermediate back to FP32 does not recover information lost earlier.

For a scalar BF16 round-trip at `x = 1` with perturbation `±0.001`, both JAX and PyTorch gave:

| Quantity | Result |
|---|---:|
| Autodiff derivative | 1 |
| Central finite difference | 0 |

Both perturbed values rounded to the same BF16 value. This demonstrates why the disagreement alone cannot diagnose a framework-specific backward defect. See [the scalar evidence](../results/generic_precision_check_20261002.json) and [the smooth-function comparison](../results/generic_audit_chain_check_20261002.json).

## 5. Independent synthetic attention experiment on CPU

The standalone fixture uses seeded numerical arrays, no sequence input, and no checkpoint. Its smooth objective is:

```text
Q = X Wq; K = X Wk; V = X Wv
A = softmax(Q K^T / sqrt(head_width))
L = mean((A V)^2)
```

An explicit NumPy FP64 chain-rule derivative provides an independent reference. Separate componentwise finite differences check that reference. JAX eager/JIT and PyTorch are compared against it.

The corrected CPU experiment passed **19 reference/finite-gradient checks** and **three analytic-reference tests**. Approximate relative input-gradient differences against PyTorch were:

| Configuration | Relative L2 difference |
|---|---:|
| FP64 | 1.3e-16 |
| FP32 | 1.4e-7 |
| Mixed BF16, eager JAX | 4.8e-8 |
| Mixed BF16, compiled JAX | 3.09e-3, or 0.309% |

An initial version of this new test accidentally promoted FP32 queries to FP64 through a NumPy scaling constant. That test bug was corrected with an explicitly typed constant and a dtype guard. The preliminary artifact is marked superseded; use the [corrected CPU results](../results/synthetic_attention_20261002_081413_799197/summary.json).

The mixed-BF16 protocol uses FP32 projections/scaling, selected BF16 boundaries/matrix multiplications, an explicitly FP32 softmax, and an FP32 final loss. It is not an “everything BF16” test or a reproduction of every model kernel.

## 6. CPU stage investigation and instrumentation effects

Six configurations were examined:

1. FP32 throughout the tested calculation.
2. Q/K/V rounding only.
3. Attention-score matrix multiplication in BF16.
4. Softmax-probability rounding only.
5. Attention-output matrix multiplication in BF16.
6. The original mixed-BF16 combination.

The matrix-multiplication variants necessarily change operand and result precision together; they do not isolate hardware accumulation precision alone.

On the original CPU fixture, relative input-gradient differences between compiled and eager JAX were approximately:

| Configuration | Difference |
|---|---:|
| Q/K/V rounding only | Negligible at the scale of the other differences |
| Probability rounding only | Negligible at the scale of the other differences |
| Score multiplication only | 0.051% |
| Output multiplication only | 0.015% |
| Original mixed BF16 | 0.309% |

A separate isolated matrix multiplication demonstrated a specific CPU mechanism: eager execution materialized BF16 product rounding, while the compiled BF16-matmul-to-FP32 expression returned the unrounded FP32 product in that example. The maximum output difference was `0.0004386753`. Saved compiler output and [the isolated result](../results/synthetic_attention_stages_20261002_082813_802222/isolated_matmul_rounding.json) support that observation.

Exposing intermediate outputs and adding zero-valued probes changed some compiled results. The diagnostic therefore retains the original output-only calculation and measures the instrumentation effect. An apparent “first divergent stage” in an instrumented graph cannot automatically be assigned to the original graph.

This CPU mechanism is not a universal statement about GPU execution or a proof of incorrect autodiff. See [the CPU stage report](../results/synthetic_attention_stages_20261002_082813_802222/summary.json).

## 7. H200 synthetic cluster batch: verified evidence

Reviewed archive: `results/synthetic_attention_cluster_20261002_094433_601222.tar.gz`.

### Run design

- Eight NVIDIA H200 GPUs; JAX ran on CUDA.
- PyTorch **2.7.1+cpu** supplied the CPU reference.
- Four seeds, three matrix sizes, two precision policies, and two fresh-process rounds: **48 workers**.
- Six configurations and two JAX execution modes per worker: **576 variant/mode cases**.
- Three evaluations per original calculation: one baseline plus two repeats.
- Runtime recorded by the runner: **156.3 seconds**.

| Size | Rows | Input width | Query/key width | Value width |
|---|---:|---:|---:|---:|
| Small | 4 | 5 | 3 | 2 |
| Medium | 64 | 32 | 16 | 16 |
| Large | 256 | 64 | 32 | 32 |

The first attempted cluster launch stopped at preflight because it required CUDA PyTorch. The repository selects a CPU PyTorch package. An explicit `--torch-backend cpu` option was added; there is no silent fallback. JAX remained on the selected GPUs, and the reports identify comparisons as JAX CUDA versus PyTorch CPU.

### Actual precision policies and versions

| Setting | Strict | Native |
|---|---|---|
| JAX matmul precision | `highest` | Default/unset (`None`) |
| PyTorch matmul precision | `highest` | `highest` |
| PyTorch TF32 flag | False | False |
| PyTorch execution | CPU | CPU |

Both policies used JAX/JAXLIB **0.11.0** and NumPy **2.4.1**. Earlier local CPU tests used JAX **0.10.0**. Thus CPU-to-H200 comparisons change both hardware and JAX version. The repository currently declares `jax>=0.8.1,<0.11`; the cluster version is outside that range. This is a compatibility/comparison limitation, not demonstrated causation.

The saved strict-mode GPU compiler output includes `operand_precision={highest,highest}`. Native output lacks that annotation. This establishes a compiler-level precision-policy difference; the saved evidence does not identify a particular native hardware instruction, so it should not be described as proof of a specific TF32 kernel.

### Completion and failed controls

- All **48 workers completed**; no worker crashed or timed out.
- All **24 strict-precision workers passed** their existing synthetic controls.
- **16 of 24 native-precision workers** flagged failed controls.
- All failures were medium/large FP32 output comparisons against the NumPy reference: **16 eager-output checks and 16 JIT-output checks**.
- No backward-gradient control was flagged as failed in this batch.

The synthetic controls use elementwise tolerances, while the comparison tables also report relative L2 errors. Passing an elementwise tolerance is not a claim of exact agreement or a universal gradient guarantee.

### Maximum relative input-gradient differences

These maxima compare **compiled JAX on H200 against PyTorch CPU**, over the sampled seeds, sizes, and rounds. They are relative L2 differences, not maximum componentwise errors and not general error bounds.

| Configuration | Strict | Native |
|---|---:|---:|
| FP32 | 2.74294e-7 = 0.0000274% | 7.49939e-4 = 0.0750% |
| Original mixed BF16 | 2.33106e-5 = 0.00233% | 2.32285e-3 = 0.232% |

The failures therefore cannot be explained as “BF16 alone broke the test.” The precision policy materially changes discrepancies in the nominal FP32 case as well.

BF16 variants are not interchangeable. The worst observed compiled-versus-eager difference was **0.4383%** for Q/K/V-only rounding on a small fixture. Probability-only rounding reached **0.3865%**. In contrast, the original mixed-BF16 fixture had identical eager/JIT input gradients for the tested small strict cases and much smaller differences for larger strict cases. The CPU pattern did not transfer uniformly to H200.

### Repeatability and integrity

- **1,152 within-process gradient-repeat comparisons** were reported identical.
- **70 of 432 fresh-process gradient-array comparisons** were not bitwise identical.
- The largest fresh-process relative L2 difference was **6.66313e-5 = 0.00666%**.
- Three differing array pairs used the same assigned GPU; other differing pairs used different GPUs. These results do not isolate process restart from GPU assignment in every pair.
- Matched fixtures were identical across rounds.
- All **938 manifest checksums** verified.
- Independent recomputation from saved original input-gradient arrays matched the reported backend, compilation, and fresh-process comparisons.
- All **2,880 logged events** had successful status; no OOM or error alert was found in the scanned worker logs.

The raw saved intermediate outputs come from the instrumented calculation. They cannot be substituted for original uninstrumented outputs when independently reassessing an output control.

Recorded JAX allocation high-water marks were approximately **128–130 MiB per worker**. These are cumulative JAX allocation counters for small synthetic calculations, not total device usage, isolated backward memory, or a full-model memory estimate.

## 8. Conclusions supported by the evidence

1. The synthetic infrastructure executes on all eight H200 GPUs and preserves useful diagnostic artifacts.
2. Strict-precision FP32 provides a useful reference for this synthetic study.
3. Precision policy, matrix shape, compilation, and where rounding occurs all matter. “FP32” or “BF16” alone is an incomplete numerical specification.
4. Finite-difference disagreement can arise from rounding and does not automatically identify a backward bug.
5. Within-process repeatability was strong in this batch; exact fresh-process reproducibility was not universal.
6. Cross-framework comparisons include a hardware difference because PyTorch ran on CPU.
7. None of these findings certify full-model gradients, structure quality, affinity, or realistic memory requirements.

The evidence does **not** establish that all-BF16 execution is the right solution, that JAX must be replaced, or that a precision setting which passes the synthetic checks is sufficient for an end-to-end model workflow.

## 9. Fixed-device synthetic validation — implemented and reviewed on H200

The new launcher implements a **same-GPU, same-version fresh-process repeatability study**, with separate numerical and synthetic pipeline audits:

- Keep the software environment fixed for the first comparison and record exact versions.
- Keep each matching fixture assigned to the same GPU across fresh processes.
- Use five fresh processes with five repeated evaluations each.
- Compare strict FP32, the original mixed-BF16 configuration, and Q/K/V-only rounding.
- Retain fixture hashes, gradients, precision metadata, compiled IR, and numerical differences.
- Let all eight GPUs run independent matched repeats, while separating same-GPU and cross-GPU analysis.

A software-version comparison is a separate experiment; changing version and GPU assignment together would complicate attribution again. Keeping the current environment fixed for a diagnostic comparison does not turn an out-of-range dependency version into a supported configuration.

**The original queue remains available. The new `--fixed-devices` option replicates fixtures onto each requested device and pins every fresh-process round to that device.** The new [end-to-end synthetic launcher](../examples/run_synthetic_end_to_end.sh) enables this mode and an independent artifact audit:

```bash
bash examples/run_synthetic_end_to_end.sh --devices 0,1,2,3,4,5,6,7
```

The default is **360 workers**: eight GPUs × three sizes × three fixture seeds (0, 1, 2) × five fresh processes, using strict precision. Each worker evaluates FP32, original mixed BF16, and Q/K/V-only rounding in eager and JIT modes, with five evaluations per configuration. This is 2,160 JAX variant/mode configurations and 10,800 original-objective forward/backward evaluations, plus 2,160 separately executed forward-only evaluations, instrumented evaluations, compilation exports, and PyTorch references. At most one worker runs per requested GPU. PyTorch is explicitly a CPU reference; JAX uses the selected GPU. There are no package installations or model/checkpoint loads.

Use `--dry-run` to inspect assignments. Override `--seeds 0` to reproduce the smaller 120-worker plan; it has less input coverage. To test the complete pipeline locally:

```bash
bash examples/run_synthetic_end_to_end.sh --cpu --sizes small --rounds 2
```

`validation.json` keeps the questions separate:

| Axis | Evidence and interpretation |
|---|---|
| Strict FP32 numerical controls | Checks loss, original output, and input gradient for the baseline **and every FP32 repeat** against the independent FP64 NumPy analytic reference evaluated at the actual FP32 inputs. Separately checks forward-only loss/output against the reference and the corresponding gradient-enabled forward result. Uses the existing FP32 tolerances. BF16 gaps remain descriptive. |
| Repeatability | Recomputes comparisons from saved raw repeats, then separates same-device fresh-process comparisons from cross-device first-round comparisons. Reports exact equality and numerical differences; variation alone is not an autodiff-bug verdict. |
| Synthetic pipeline integrity | Checks requested device/config/backend, fixed versions and device kind, seeded fixture contents/hash, expected observations and timing events, compiled-IR presence, exact shapes, declared FP32 storage dtype, and finiteness for original, repeated, and forward-only arrays, plus reported-loss consistency. Broadcasting is forbidden in paired comparisons. Missing or inconsistent evidence prevents an assessed numerical verdict. |
| Full-model correctness | **Not assessed.** This suite does not evaluate model integration, biological structures, model gradients, or realistic model memory. Synthetic pipeline integrity does not answer that question. |

The raw uninstrumented outputs and losses are now saved separately from instrumented stage outputs. Additional tables are `independent_numerical_checks.csv`, `independent_within_process.csv`, `fixed_device_repeats.csv`, and `forward_path_comparisons.csv`. Numerical rows identify the evaluation (`baseline`, `repeatN`, `forward_only`, or `forward_vs_backward`). Configurations include effective compilation-cache settings; the test does not force a particular compiler/autotuning choice or diagnose it as the cause of variation.

The suite returns zero only when its workers complete, the independent strict FP32 numerical controls and synthetic integrity checks pass, and repeatability is assessed. `variation_observed` is a separate repeatability result and can coexist with exit zero **only if every independently checked FP32 evaluation still passes its numerical tolerance**. A large erroneous repeat cannot pass merely by being labeled variable. `full_model_correctness` always remains `not_assessed`. Read the separate axes rather than interpreting exit zero as universal validation.

Outputs use `results/synthetic_end_to_end_<timestamp>_<pid>/`, with the downloadable archive beside that folder. The automatic archive is reopened and every file is checked against the checksum manifest, with exact file coverage. Archive verification failure returns nonzero. With `--no-archive`, that archive check is intentionally not performed. The immutable archived summary records the preceding numerical/pipeline checks and labels archive verification as pending. A sibling `<result-name>.completion.json` records the final exit code and archive verification outcome after packaging. If packaging or verification fails, this completion record retains the error and exit code; the original result directory remains available. Download the archive and completion record together. The shell launcher preserves the Python exit status.

Audit review exposed two defects in the initial audit (schema 1): a saved repeat gradient multiplied by 100 did not fail numerical controls, and a one-row repeat gradient could broadcast against the baseline without failing integrity. Both were reproduced in failing regression tests before the fixes. The earlier 28-test/six-worker CPU pass did **not** cover these cases and must not be cited as evidence that repeats were numerically validated.

The corrected audit (schema 2) validates repeats independently and rejects wrong shapes, wrong dtypes, NaNs, and infinities before comparisons. Tests also cover missing/incorrect forward-only outputs. Real child-process failure tests verify nonzero runner results and preserved records for bad repeat gradients, worker timeouts, and corrupted archives; a shell test verifies exit-code forwarding. New worker outputs are required for schema 2 because older captures omit the forward-only arrays and use a different Torch scalar storage dtype.

Local validation after these corrections: **50 focused tests passed**, plus lint and shell syntax checks. A real **18-worker CPU run** covered all three seeds and sizes, two fresh processes per seed/size, all three variants, and five gradient-enabled evaluations per configuration. All **738 numerical checks** passed; the audit recorded **216 forward-only versus gradient-path comparisons**. All **1,296 within-process** and **243 same-device fresh-process** comparisons were identical. All **260 archived files**, including the manifest, verified; the final completion record reports exit 0. The 360-worker eight-GPU dry run was checked for complete fixed-device/seed/size groups. The later H200 result is reviewed below; these CPU results remain a separate validation record.

Fresh compilation versus reused compilation remains a separate future experiment, not an automatic part of this run. Recorded compilation-cache settings aid interpretation without attributing variation to a specific compiler choice.

This check can characterize numerical variation. It cannot, by itself, finish full-model validation. Further claims about integration, forward-output quality, or backward resource usage require separate evidence at that level. More generic toy repetitions alone do not close that gap.

### 9.1 Corrected H200 suite: verified result

Archive: `synthetic_end_to_end_20261002_220213_1118482.tar.gz`. The independent [review](../results/synthetic_end_to_end_review_20261002_220213/REVIEW.md) and [machine-readable evidence](../results/synthetic_end_to_end_review_20261002_220213/review.json) record these findings:

- **360/360 workers completed** in approximately 15.3 minutes on eight H200 GPUs.
- All **4,713 manifest-listed hashes** matched, with exact file coverage. All six archived synthetic source files match the local scripts. No archived code was executed.
- Independent recomputation confirmed **14,760/14,760 FP32 numerical checks passed**, including every repeat and forward-only output. All recomputed comparison rows matched the archived audit.
- **25,920/25,920 within-process comparisons were identical.**
- **535/7,776 same-GPU fresh-process comparisons differed.** The maximum relative L2 difference was **0.006663%**, for the eager Q/K/V-rounding gradient. Strict FP32 input-gradient restart differences were at most **0.00000955%**.
- **182/1,701 cross-device first-round comparisons differed**; the maximum relative L2 difference was also approximately 0.006663%.
- Of 4,320 separate forward-only versus gradient-path comparisons, 280 were not bitwise identical. Every FP32 forward-path control passed; the largest FP32 relative difference was approximately **0.0000108%**.
- A separate eager/JIT comparison reached **0.195083%** for the Q/K/V-rounding input gradient. This is a descriptive BF16 comparison, not a failed strict FP32 control, and is distinct from restart variation.

Thus the corrected synthetic numerical and pipeline checks passed, while fresh-process bitwise reproducibility remains incomplete. The fixed-device results show that GPU reassignment is not necessary for variation; they do not identify its cause. These results do not close the full-model correctness or realistic memory gap.

Recorded software remained JAX/JAXLIB 0.11.0 and PyTorch 2.7.1+cpu; the prior dependency-range caveat and CPU/GPU comparison caveat still apply. The cluster completion sidecar was not supplied, but archive integrity was independently verified here.

## 10. Existing scripts and review artifacts

| Item | Purpose |
|---|---|
| [Synthetic attention numerics](../examples/synthetic_attention_numerics.py) | Independent analytic reference, CPU precision comparisons, finite-difference curves |
| [Stage diagnostic](../examples/synthetic_attention_stages.py) | Stage precision variants, original/instrumented comparison, CPU/CUDA selection |
| [Cluster shell launcher](../examples/run_synthetic_attention_cluster.sh) | Starts the bounded synthetic sweep |
| [Cluster scheduler](../examples/synthetic_attention_cluster.py) | Preflight, eight-device scheduling, logs, tables, archive creation |
| [Verified H200 review](../results/synthetic_attention_cluster_review_20261002_094433/review.json) | Integrity checks, raw-array recomputation, exact findings and limitations |
| [H200 aggregate comparisons](../results/synthetic_attention_cluster_review_20261002_094433/comparison_summary.csv) | Maxima by precision policy and configuration, with worker identifiers |
| [Independent reference comparison](../results/synthetic_attention_cluster_review_20261002_094433/independent_reference.csv) | Recomputed gradients and explicitly labeled instrumented outputs |
| [CPU attention tests](../tests/test_synthetic_attention_numerics.py) | Componentwise check of the independent derivative |
| [Runner tests](../tests/test_synthetic_attention_cluster.py) | Plans, backend policy, cleanup, and failure archives |

The earlier CPU-reference backend update passed **16 focused tests**, lint, and a local CPU end-to-end check. See §9 for the later audit corrections and current validation. Those software checks are distinct from the 48-worker H200 experiment and its control outcomes.

The existing synthetic sweep can be reproduced with:

```bash
bash examples/run_synthetic_attention_cluster.sh \
  --devices 0,1,2,3,4,5,6,7 \
  --torch-backend cpu
```

That command reproduces the original queued sweep. Use the new launcher in §9 for fixed-device validation. Neither loads a protein model or checkpoint.

Each result bundle contains `summary.json`, `status.json`, `runner.log`, worker logs, timing/memory events, arrays, compiled IR, source snapshots, environment metadata, GPU telemetry, comparison CSVs, and `checksums.sha256`. The launcher automatically produces a sibling `.tar.gz` archive unless disabled. A finite sweep stops when finished; it does not deliberately occupy the GPUs until morning.

Links under `results/` refer to local evidence artifacts, which may not be present in another checkout unless copied there. Historical reports retain their original field names and judgments; the reporting changes described above do not rewrite old evidence.
