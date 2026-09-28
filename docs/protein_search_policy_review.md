# AI-guided protein search: literature review for the P17 project

**Follow-up, 2026-09-22:** The latest user constraints, three-policy shortlist,
Germinal discussion, evolutionary-search follow-up, and paper index
are now in [the project handoff](p17_jn1_redesign.md#10-search-policy-handoff-for-claude--2026-09-22)
and [its references](p17_jn1_redesign.md#813-sources). That handoff supersedes the
initial priority ranking below. The current P17 MCMC implementation is an
unfinished prototype, not a validated baseline. The first proposed comparison
retains frozen OpenDDE + AbLang2 guidance, fixed framework/CDRs, and requires no
additional model training.

The subsequent index includes 26 papers plus a pinned BindCraft2 implementation
reference. [§12 of the handoff](p17_jn1_redesign.md#12-first-implementation-and-response-to-claude--2026-09-22)
records the first coded harness, review corrections, and CPU validation; real-model
GPU validation remains pending.

Reviewed 2026-09-21. This is a broad, targeted literature survey, not an exhaustive review of every protein-design paper. Relevant methods, results, and limitations were inspected in primary papers; entries explicitly marked abstract-level received a lighter review. No experiments were reproduced. Project observations come from `p17_jn1_redesign.md` and the earlier code review, not independent validation of its experimental outputs.

The most relevant problem formulation is **constrained optimization of an existing antibody, with expensive and imperfect evaluation**. This overlaps several literatures that the original notes underrepresent: local sequence optimization, antibody lead optimization, surrogate-assisted evolution, and discrete gradient sampling.

## What the broader literature changes

The choice is larger than continuous hallucination versus beam search. It helps to separate five decisions:

1. **Representation:** complete discrete sequences, relaxed sequence probabilities, learned latent states, or partially denoised structures.
2. **Proposal:** random edits, language-model suggestions, gradient-informed edits, recombination, or generative sampling.
3. **Search memory:** one current sequence, multiple independent chains, a population, a beam, or a learned proposal policy.
4. **Evaluation allocation:** which proposals receive cheap scoring, expensive prediction, or repeated evaluation.
5. **Feedback:** whether new evaluations update only an archive, the parent population, a surrogate, or a generative policy.

These choices are composable. A population search can use gradient proposals and a Bayesian acquisition rule. A surrogate can predict an expensive computational score rather than an experimental measurement, but then its learned target remains computational confidence, not binding affinity.

My revised interpretation is that search and objective quality should be studied together. Existing results can reveal proxy failures, while controlled search comparisons reveal whether the current policy explores too narrowly. Neither question must be completely resolved before investigating the other.

## Closest methodological comparisons

| Work | Search mechanism and evidence | Relevance and limitation for this project |
|---|---|---|
| [LaMBO-2 / NOS, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/29591f355702c3f4436991335784b503-Paper-Conference.pdf) | Guided discrete diffusion with Bayesian optimization, edit constraints, and saliency-based position selection; includes experimental antibody optimization. | One of the closest problem matches. Its low-edit ablation finds position selection particularly consequential. It uses learned property models and experimental feedback, so its performance cannot be assumed with OpenDDE scores substituted. |
| [PEX, ICML 2022](https://proceedings.mlr.press/v162/ren22a/ren22a.pdf) | Proximal exploration trades predicted fitness against distance from the starting sequence and explores near the resulting frontier. Benchmarked on measured protein landscapes. | Directly addresses improvement with few mutations. Its learned fitness models differ from this project's structural oracle. A hard mutation cap alone does not reproduce its exploration policy. |
| [Plug & Play Directed Evolution / EvoProtGrad, 2023](https://arxiv.org/pdf/2212.09925) | Gradient-informed discrete MCMC composes functional predictors and sequence priors as a product of experts; evaluates multistep proposals with forward/reverse probabilities. | Closest algorithmic relative of the existing discrete search. The paper discusses, but does not demonstrate, additional hard edit and region constraints. Adapting constrained proposals still requires care. |
| [AdaLead, 2020](https://arxiv.org/pdf/2010.02141) | Adaptive evolutionary search retains promising parents and uses mutation/recombination rollouts. Its FLEXS comparisons show simple search can be a strong baseline. | A useful population baseline before attributing gains to elaborate search machinery. It is not equivalent to retaining the top sequences in a beam. |
| [BO-EVO, 2023](https://academic.oup.com/bib/article/24/1/bbac570/6958505) | Gaussian-process uncertainty and an acquisition function prioritize evolutionary proposals; includes landscape benchmarks and robotic experimental work. | Relevant to spending expensive evaluations efficiently. Its published MCMC comparison is not a benchmark of mosaic's particular gradient-guided, constrained implementation. |
| [BoGA, 2026 preprint](https://arxiv.org/html/2603.02753v1) | Uses evolutionary proposals inside an online surrogate/acquisition loop, including computational structure objectives. | Closer to the expensive-computational-oracle setting than methods requiring an assay every round. Benefits depend on surrogate accuracy and the relative costs of fitting, proposing, and evaluating. |
| [MosPro, 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11952807/) | Combines discrete gradient sampling with multi-objective gradient balancing; evaluated on experimental fitness landscapes. | Relevant when objectives conflict. Keeping one best sequence per edit count is not the same as maintaining a frontier over interface quality, plausibility, and structural agreement. |
| [Improving Protein Optimization with Smoothed Fitness Landscapes / GGS, 2024 revision](https://arxiv.org/html/2307.00494v3) | Smooths learned fitness landscapes and combines them with gradient-informed discrete sampling. | Shows that search difficulty can depend on the learned landscape, not just the sampler. It does not establish that smoothing OpenDDE outputs preserves useful structural information. |

These papers suggest three distinct questions for P17: whether good combinations are reachable under the allowed edits; whether the proposal and population policy finds them; and whether the evaluator recognizes them. Their published results do not resolve those questions for this system.

## Broader policy families

| Work | What it contributes | Transfer boundary |
|---|---|---|
| [LaMBO, ICML 2022](https://proceedings.mlr.press/v162/stanton22a.html) | Multi-objective Bayesian optimization using a denoising autoencoder and a Gaussian-process head. | Relevant surrogate/representation framework; not a replacement requiring no training data. Abstract and method overview reviewed. |
| [CbAS, ICML 2019](https://proceedings.mlr.press/v97/brookes19a/brookes19a.pdf) | Adapts a generative distribution toward desired properties while accounting for a prior distribution. | Addresses exploitation of unreliable predictor regions; remaining near a prior is not a guarantee of binding. |
| [Biological Sequence Design with GFlowNets, ICML 2022](https://proceedings.mlr.press/v162/jain22a/jain22a.pdf) | Learns to generate diverse, rewarding candidates within an active-learning loop. | Relevant to batch diversity and amortized proposal generation; training overhead needs justification for a single small campaign. |
| [Fast SeqProp, 2020 manuscript](https://arxiv.org/abs/2005.11275) | Differentiable optimization through discrete samples addresses limitations of continuous sequence relaxation. | Relevant to the soft-to-discrete transition; the abstract alone does not establish superiority to APGM here. Abstract-level review. |
| [Language-model-guided antibody evolution, 2023](https://www.nature.com/articles/s41587-023-01763-2) | Experimentally tests proposals based on evolutionary plausibility, without target-specific input to the language model. | Supports language-model proposals as a meaningful baseline, not a guarantee of target-specific improvement. |
| [EVOLVEpro, Science 2025](https://doi.org/10.1126/science.adr6006) | Combines protein-language-model representations with few-shot active learning and experimental feedback. | Relevant when measured functional data are available; labels derived from structural confidence would define a different task. Publisher abstract-level review; full text was not accessible in this pass. |
| [BindCraft, Nature 2025](https://www.nature.com/articles/s41586-025-09429-6) | Structure-predictor backpropagation plus sequence refinement and filtering; experimentally validated de novo binders. | Useful continuous/discrete refinement precedent, but its complete pipeline and published success rates do not transfer to an optimizer function with a fixed scaffold and strict edit cap. |
| [EasyNano, June 2026 preprint](https://arxiv.org/html/2606.12772v1) | CDR-restricted distogram optimization with epitope and structural-pose objectives. | Closest structural-objective match; explicitly reports proxy/full-model divergence, dependence on initial pose, and absence of experimental validation. |
| [Proteina-Complexa, ICLR 2026](https://arxiv.org/html/2603.27950v1) | Compares best-of-N, beam search, Feynman–Kac steering, MCTS, and generative-plus-hallucination refinement under compute budgets. | Its structured search operates on generative denoising trajectories. It supports evaluating search policies, not assuming mutation-space beam search inherits its advantage. |
| [RosettaSearch, 2026 preprint](https://arxiv.org/abs/2604.17175) | LLM-assisted multi-objective inference-time search for backbone-conditioned sequence design, evaluated computationally. | Shows another search representation; not evidence for constrained antibody binding recovery. Abstract-level review. |
| [Why risk matters for protein binder design, 2025 workshop paper](https://arxiv.org/html/2504.00146v1) | Compares BO model configurations using campaign performance, cost, and downside-risk metrics. | Concerns variation in optimization campaigns, not repeated structure-prediction seeds for one candidate. |

RL-style approaches and GFlowNets learn proposal policies; BO learns an objective surrogate and chooses evaluations. These are different ways of using previous evaluations. Neither removes the need for a useful reward or trustworthy labels. For this project, their training costs and data requirements matter as much as their expressivity.

## Corrections and qualifications to the existing notes

**The BO-EVO numbers exist, but their scope is narrower than stated.** The paper reports an 11% and 21% increase in round-five success ratio against its MCMC and AdaLead baselines, respectively. This is a specific benchmark result, not an expected improvement over mosaic's current policy. The paper's AdaLead baseline should be named rather than described only as generic pure evolution. [BO-EVO](https://academic.oup.com/bib/article/24/1/bbac570/6958505)

**The risk paper does not validate the proposed seed-variance ranking.** It examines optimization-campaign risk and reports no added benefit from risk-aware model ranking in its tested setting because optimization stochasticity obscures it. Its findings can motivate careful repeated benchmarking, but cannot be cited as proof that low variance across structural seeds identifies better binders. [Risk paper](https://arxiv.org/html/2504.00146v1)

**EasyNano is methodological precedent, not biological validation.** It explicitly acknowledges that distogram improvement may diverge from full-model confidence, especially for poor initial poses, and that designed sequences have not been experimentally validated. The inspected version also contains a placeholder repository URL and an unresolved archive DOI. That limits reproducibility from the paper's links. [EasyNano, Discussion and Code availability](https://arxiv.org/html/2606.12772v1)

**Proteina-Complexa's advantage is conditional on its model and search space.** The paper measures computational success and attributes gains to combining a generative prior with search. Its easy/hard target categories do not establish that P17 belongs to the same algorithmic regime merely because its starting confidence is low. [Proteina-Complexa](https://arxiv.org/html/2603.27950v1)

**Structural confidence remains a surrogate for the biological endpoint.** BindCraft itself distinguishes useful binding classification from affinity prediction and discusses prediction limitations. Full OpenDDE evaluation can be more informative than the cheap objective without becoming experimental ground truth. [BindCraft, Conclusions](https://www.nature.com/articles/s41586-025-09429-6)

Two additional points follow from the optimization setup rather than a particular paper: enlarging a feasible set does not guarantee a finite stochastic search recovers the smaller-budget run's results; and a correct Metropolis–Hastings implementation is not automatically the most effective optimizer under a fixed compute budget.

## Implications for assessing the current implementation

The existing APGM-plus-discrete-search pipeline is a recognizable member of the differentiable-design and gradient-informed-evolution families. It is not an obviously wrong starting point. The missing comparisons are local exploration with explicit distance management, population search, and policies that learn from expensive evaluation.

The most informative shortlist is therefore **PEX, EvoProtGrad, AdaLead, LaMBO-2, and BO-EVO/BoGA**, with different roles: local constraint handling, gradient proposals, population exploration, constrained antibody generation, and expensive-query allocation. This is a relevance ranking, not a predicted performance ranking.

For future comparisons, the main evidence to retain is:

- The same allowed sequence space and evaluation criteria across policies.
- Total compute including proposals, backward passes, full predictions, repeat seeds, and surrogate fitting.
- Unique feasible candidates and their diversity, not just the best cheap loss.
- Both objective agreement and cases where cheap and expensive scores disagree.
- Separation of structural sampling variation, uncertainty in a learned surrogate, and biological uncertainty.
- Repeated search runs, with selection distinguished from subsequent evaluation to reduce selection bias.

My current hypothesis is that a population or archive that incorporates expensive evaluation feedback is a more useful organizing direction than selecting one isolated replacement optimizer. That hypothesis is compatible with gradient proposals, evolutionary proposals, or a surrogate; it does not yet justify choosing beam search, changing the mutation budget, or training a new policy as the winning implementation.
