# Search policies, gradient guidance, and pose: a literature review

Reviewed: **2026-10-05**.

This document collects the literature comparison discussed alongside
[P17_JN1.md](P17_JN1.md). It emphasizes optimization mechanisms, the meaning of
structural measurements, computational-budget comparisons, and the limits of
the published evidence. It is a methodological review, not a specification for
changing the project's sequence-optimization workflow.

The local workflow is described from the project MD, rather than a new source
code audit. Relevant methods, results, and author-maintained resources were
checked; none of the papers' experiments was reproduced. Paper findings and
interpretations for the comparison are distinguished below.

**Central finding:** a limited search budget weakens conclusions about failure,
but the current workflow is not simply Germinal with fewer steps. The methods
also differ in search representation, conditioning, objective composition,
candidate selection, and validation. Their published success rates and RMSD
values therefore do not form a directly comparable ranking.

## 1. Separate the model from the search policy

A structure predictor, a proposal mechanism, a population policy, and a final
evaluation metric are different components. Sharing one does not make two
pipelines equivalent.

The [project description](P17_JN1.md#122-implemented-behavior) separates a
gradient callback from a confidence-score callback. In the subsequently
documented full-gradient workflow:

- Frozen OpenDDE and AbLang2 contribute to a composite proposal objective.
- Gradient information informs discrete sequence proposals.
- Separate structure predictions supply the confidence used for retention.
- Multiple active candidates compete, while evaluated candidates are archived.
- Structural seeds used for final reporting are distinct from selection seeds.

This is a **gradient-informed discrete population search**. The presence of
diffusion inside the predictor does not make the outer search a diffusion
sampler. The use of stochastic acceptance does not, by itself, establish
Metropolis–Hastings sampling. Nor is a frozen-model loop automatically Bayesian
optimization: the MD does not describe an online posterior and acquisition
function governing the current loop.

A first-order prediction of the composite proposal loss is also not a
first-order prediction of a different retention score. Disagreement between
those quantities cannot, by itself, diagnose incorrect autodifferentiation.

## 2. Germinal: shared premise, different optimization procedure

The published Germinal method combines AlphaFold-Multimer with IgLM, uses
continuous sequence representations followed by discrete refinement, and merges
normalized guidance gradients. Subsequent redesign and filtering form part of
the pipeline. Its experimental results evaluate that complete system.
[Germinal, Nature Biotechnology, 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13366713/)

| Aspect | Current workflow as described in the MD | Published Germinal |
|---|---|---|
| Guidance models | OpenDDE and AbLang2 | AlphaFold-Multimer and IgLM |
| Representation | Discrete candidates | Continuous relaxation followed by discrete refinement |
| Objective composition | Composite loss with clipping | Normalized gradients; weighted or conflict-aware composition |
| Candidate history | Competition among active parents | Staged trajectories and downstream filtering |
| Sequence setting | Restricted neighborhood of a starting sequence | De novo CDR generation with framework bias |
| Evidence | Predictor confidence and reference-pose measurements | Computational assessments plus experimental validation |

Germinal changes language-model influence across optimization phases. This is
not evidence for a controller that adjusts a pose weight in response to RMSD.
Its paper also reports representative experimental structural validation; it
does not establish arbitrary reference-pose recovery for every trajectory.
[Published methods and results](https://pmc.ncbi.nlm.nih.gov/articles/PMC13366713/)

**Version caveat:** the current authors' README advertises AbLang as the default,
whereas the publication describes IgLM. The repository is evolving. A statement
about the publication should not silently become a statement about the current
default installation. The README was reviewed on the date above; this review
does not constitute an audit of its implementation.
[Authors' repository](https://github.com/SantiagoMille/germinal)

## 3. Other methods overlap with different components

The similarities below are a methodological classification, not results from a
head-to-head benchmark against the local workflow.

| Method and primary source | Relevant mechanism | Main distinction or evidence limit |
|---|---|---|
| [EvoProtGrad / PPDE: Plug & Play Directed Evolution](https://arxiv.org/abs/2212.09925) | Gradients approximate promising changes in a discrete space. | Formulated as discrete MCMC with Metropolis–Hastings correction. This differs from heuristic population competition. Its results do not establish recovery of a specified complex pose. |
| [LaMBO-2 / NOS, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/29591f355702c3f4436991335784b503-Paper-Conference.pdf) | Guided discrete diffusion, saliency-based edit-location choice, and Bayesian acquisition. | Uses learned property models and an iterative data-acquisition setting. It is more than a proposal rule operating on a fixed structure predictor. |
| [AdaLead, 2020 preprint](https://arxiv.org/abs/2010.02141) | Adaptive evolutionary exploration using fitness feedback. | Does not require sequence gradients. Its benchmark evidence concerns exploration of fitness landscapes, not reference-pose recovery. |
| [PEX, ICML 2022](https://proceedings.mlr.press/v162/ren22a.html) | Exploration balances predicted fitness and distance from a reference sequence. | Its proximal frontier is a selection/exploration concept; a hard mutation cap alone does not reproduce the method. |
| [MosPro, iScience 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11952807/) | Differentiable property predictors inform discrete, multiobjective sampling. | Studies composition of property gradients and objective trade-offs. Evidence on property benchmarks is not evidence of spatial placement control. |
| [BindCraft, Nature 2025](https://www.nature.com/articles/s41586-025-09429-6) | Structure-predictor backpropagation, staged sequence optimization, redesign, and filtering. | Results concern the complete pipeline. They cannot be assigned solely to the initial gradient stage or equated with a discrete population policy. |
| [RFdiffusion, Nature 2023](https://www.nature.com/articles/s41586-023-06415-8) | A trained denoising model generates backbone geometry, followed by sequence design and assessment. | Geometry generation differs from changing a discrete sequence and observing a structure predictor's output. |
| [BoltzGen, 2025 manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC12697729/) | All-atom generative modeling with sequence/structure generation and downstream evaluation. | Refolding consistency and design filtering are distinct from recovering a preselected reference arrangement. |

These comparisons separate three questions that the word *policy* can obscure:
how a candidate is generated, how alternatives are retained, and how the final
outcome is judged. Success in one component does not establish superiority of
the complete combination.

## 4. Position selection and replacement choice are distinct capabilities

LaMBO-2 includes an ablation separating saliency-based position selection from
gradient guidance during generation. In its tested limited-edit setting,
position selection had the larger effect. This supports distinguishing those
two capabilities rather than treating all uses of a gradient as one operation.
[LaMBO-2, §5.3](https://proceedings.neurips.cc/paper_files/paper/2023/file/29591f355702c3f4436991335784b503-Paper-Conference.pdf)

**Interpretation limit:** that finding does not establish that frequently
modified positions in another workflow are correctly identified by its
gradient. Final winners reflect proposal generation, sampling, acceptance, and
retention together. Likewise, a residue absent from final winners may have been
proposed or temporarily accepted earlier. Endpoint differences do not reconstruct
a trajectory.

## 5. “Pose success” has several meanings

| Measurement or claim | What it addresses | What does not follow automatically |
|---|---|---|
| Independently aligned binder RMSD | Internal shape similarity after removing rigid placement | Correct placement relative to the target |
| Target-aligned binder RMSD | Agreement with a specified relative arrangement | Experimental correctness of the reference arrangement |
| Design-to-refolding RMSD | Whether another prediction reproduces a generated design | Recovery of an independently chosen reference pose |
| Interface confidence | A predictor's assessment of an interaction | Measured affinity or agreement with a particular pose |
| Experimental complex structure | Structural evidence for the measured complex | General success across all generated candidates |

The docking literature explicitly separates reference-contact recovery,
interface RMSD, and ligand RMSD. DockQ combines those quantities because they
capture different aspects of quality. A single structural number does not
replace all three.
[DockQ, Basu and Wallner, 2016](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0161879)

BoltzGen's refolding comparisons concern agreement with a generated design.
They are not automatically equivalent to target-aligned RMSD against an existing
arrangement. Comparing published cutoffs requires the same reference, alignment,
atom selection, and treatment of flexible regions.
[BoltzGen manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC12697729/)

ipSAE was introduced as an interaction-confidence measure derived from aligned
errors. It is not a direct measurement of displacement from a user-specified
reference pose.
[Dunbrack, Rēs ipSAE loquuntur](https://www.biorxiv.org/content/10.1101/2025.02.10.637595v2)

For generic geometry, low independently aligned shape error alongside high
relative-placement error supports placement as a substantial component of the
discrepancy. It does not prove perfect local geometry. An RMSD in angstroms is
also not necessarily a translation distance: rotations and deformations can
contribute to it.

## 6. What non-protein pose papers establish

| Primary source | Contribution | Limit of transfer |
|---|---|---|
| [Task Space Regions — Berenson, Srinivasa and Kuffner, IJRR 2011](https://personalrobotics.cs.washington.edu/publications/berenson2011task.pdf) | Represents acceptable sets of rigid positions and orientations, rather than only one exact configuration. | A rigid-pose region is not equivalent to RMSD on a deformable point cloud. Planning guarantees depend on the planner and its assumptions. |
| [TrajOpt — Schulman et al., RSS 2013](https://www.roboticsproceedings.org/rss09/p31.pdf) | Combines penalties, local convex approximations, and trust-region reasoning in constrained trajectory optimization. | Its results do not establish that a coefficient change alone solves an arbitrary nonlinear model-based search. The surrounding optimization procedure matters. |
| [A General and Adaptive Robust Loss Function — Barron, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Barron_A_General_and_Adaptive_Robust_Loss_Function_CVPR_2019_paper.html) | Relates loss shape to the influence of residuals and studies adaptive robustness. | Robust treatment of outliers and satisfaction of a geometric requirement are different objectives. Smoothness alone does not establish useful guidance. |
| [An Analysis of SVD for Deep Rotation Estimation — Levinson et al., NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/fec3392b0dc073244d38eba1feb8e6b7-Abstract.html) | Analyzes rotation estimation and differentiation through SVD-based orthogonalization. | Numerical validity of alignment derivatives and effectiveness of a long optimization trajectory are separate questions. |

These papers clarify representations, mathematical behavior, and evidence
requirements. None directly validates the local sequence-to-structure workflow.

## 7. Several different mechanisms are called “dynamic weighting”

| Mechanism | What changes | Distinction |
|---|---|---|
| Predetermined phase schedule | Relative influence according to optimization stage | A schedule is not necessarily feedback from measured task failure. |
| Gradient-magnitude balancing | Contributions based on gradient scales or progress | Balancing magnitudes does not itself resolve conflicting directions. |
| Conflict-aware composition | How objective gradients are combined | This concerns direction as well as size. |
| Constraint penalties | Influence of constraint violation within a constrained solver | Evidence depends on the complete solver and its assumptions. |
| Proposal-temperature adaptation | Concentration of a sampling distribution | This differs from changing the relative objectives underlying its scores. |

GradNorm studies adaptive loss balancing during multitask learning. PCGrad
studies interference between task gradients. These address related but distinct
problems, and their training results do not automatically transfer to discrete
proposals through frozen models.
[GradNorm, ICML 2018](https://proceedings.mlr.press/v80/chen18a.html),
[PCGrad, NeurIPS 2020](https://papers.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)

Two general mathematical distinctions remain important:

1. Uniformly scaling a combined objective differs from changing one component's
   relative weight. Normalization can remove a common scale while preserving a
   change in direction. The exact clipping and normalization operations matter.
2. A hinge can have a constant derivative with respect to its scalar residual
   above a threshold, while its gradient with respect to model inputs varies
   through the intervening transformations.

These distinctions establish neither that adaptive weighting will succeed nor
that it is ineffective in the current application.

## 8. What the “too few steps” objection can support

A limited computational budget is a legitimate alternative explanation for a
failure to find a known feasible solution. Such failure demonstrates the outcome
of the tested pipeline under that budget; it does not prove impossibility or an
intrinsic failure of its gradient.

But published iteration counts are not a common currency:

| Reported unit | Why it cannot be equated with the others |
|---|---|
| Discrete proposal | May be rejected, cached, or require multiple model evaluations. |
| Gradient update | Includes differentiation, whose cost differs from forward inference. |
| Diffusion step | One part of a trained generative trajectory. |
| Full design trajectory | May include multiple representations and downstream processing. |
| Experimentally tested design | Already passed a selection process; does not count all computational attempts. |

BindCraft's benchmarking accounts for multiple computational stages. LaMBO-2's
experimental rounds also include retraining and methodological changes. Neither
supports equating its iteration count or tested-library size with another
workflow's score-call count.
[BindCraft methods](https://www.nature.com/articles/s41586-025-09429-6),
[LaMBO-2 experimental evaluation](https://proceedings.neurips.cc/paper_files/paper/2023/file/29591f355702c3f4436991335784b503-Paper-Conference.pdf)

Three further limits apply to interpretations of search histories:

- Improvements after an earlier cutoff establish that useful later outcomes
  occurred in that run. They do not establish continued improvement at its end.
- Final distance from the starting state does not measure the number of moves
  explored or the depth of the trajectory.
- Comparing variance across different tasks and budgets does not isolate the
  effect of budget. A known feasible endpoint likewise does not establish a
  unique path to it or that all intervening states will be retained.

A fair performance claim needs comparable tasks, input information, success
definitions, and computational accounting. No reviewed paper establishes a
universal sufficient budget for the workflow described in the project MD.

## 9. Conclusions supported by this review

- The local proposal mechanism belongs broadly to gradient-informed discrete
  search; its population component adds a separate selection/history mechanism.
- Germinal and BindCraft share the broader use of structure-predictor guidance,
  while differing substantially in representation and pipeline organization.
- LaMBO-2 distinguishes position selection from generation, but its evidence
  does not establish either capability in another system.
- Pose agreement, refolding consistency, predictor confidence, and experimental
  structural agreement are separate outcomes.
- Limited compute, mismatched objectives, numerical behavior, and candidate
  selection remain distinct explanations. The literature does not identify one
  as the cause of the project's remaining pose discrepancy.
- Reported metric improvements do not alone establish the contribution of
  gradients, population memory, or adaptive weighting.

Claims such as “the gradient lacks residue information,” “population search
caused the improvement,” or “more time would solve placement” remain stronger
than the documented evidence. Published success in another pipeline does not
resolve those attributions.

## 10. Provenance and reading limits

Primary sources are linked at the relevant claims above. The comparison draws
on the relevant methods and results in the papers, plus the current Germinal
README for the explicitly labeled version caveat. It does not claim exhaustive
review of every supplement or a reproducibility audit of each repository.

For the local results, [P17_JN1.md](P17_JN1.md) is the source. This literature
review does not independently verify its raw archives, structures, timings, or
implementation. No new model runs, source changes, or performance results are
introduced here.
