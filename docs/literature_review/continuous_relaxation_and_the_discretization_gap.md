# Continuous relaxation of a discrete sequence, and the cost of rounding it back

## Current reading map and BindCraft2 code comparison — 2026-10-22

**Continuous and discrete describe the search representation; gradient guidance
can be used with either.** The hybrid idea has several precedents. A staged
handoff, straight-through optimization, gradient-informed discrete proposals
and guided diffusion are different ways to combine them, not interchangeable
implementations of one algorithm.

This update supersedes the categorical recommendations in the October 7 review.
The comparison below distinguishes published papers, a preprint, current
software documentation and observations from this repository.

| Source | How continuous/discrete variables and gradients interact | Relevance and evidence limit |
|---|---|---|
| [BindCraft 1, Nature 2025](https://www.nature.com/articles/s41586-025-09429-6); [official algorithm description](https://github.com/martinpacesa/BindCraft#implemented-design-algorithms) | Relaxed gradient optimization transitions to hard sequences and semigreedy refinement. | Direct precedent for staged hybrid search. The complete pipeline's experimental results do not isolate the contribution of each optimizer stage. |
| [BindCraft2 design guide](https://github.com/PacesaLab/BindCraft2/blob/b6fb3ae15ab6a702f1156cffc11bcc3323c2fd06/docs/source/design-guide.md) | Relaxed stages, gradient-based hardening, mutation polishing, then ProteinMPNN redesign and validation. | Code-level comparator, not a reproduction of its performance. The [official site](https://www.pacesalab.com/bindcraft) labels its preprint “coming soon” at this review date. |
| [Fast SeqProp: Fast activation maximization for molecular sequence design, 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8527647/) | Evaluates sampled discrete sequences while updating normalized continuous logits with a straight-through estimator. | A within-iteration bridge between the two representations. Straight-through gradients are surrogate gradients, not exact derivatives of discrete sampling. |
| [RSO: Scalable protein design using optimization in a relaxed sequence space, Science 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11734486/) | Relaxed sequence optimization produces a backbone; ProteinMPNN generates discrete sequences for that backbone. | Evidence for relaxed search plus a separate decoder. It is not evidence that simple rounding preserves the result or that a parent-relative edit cap is satisfied. |
| [PPDE / EvoProtGrad, 2023](https://arxiv.org/abs/2212.09925) | Uses gradients to propose discrete sequence changes in a product-of-experts MCMC framework. | Relevant to gradient-guided discrete search; it does not require a continuous seeding phase. The local confidence-retention harness is not automatically the same sampler. |
| [Gibbs With Gradients: Oops I Took A Gradient, ICML 2021](https://proceedings.mlr.press/v139/grathwohl21a.html) | Uses local gradient information to construct discrete Metropolis–Hastings proposals. | Foundational explanation of how gradients guide discrete moves. Its sampling guarantees do not transfer to an arbitrary acceptance rule. |
| [LaMBO-2 / NOS: Protein Design with Guided Discrete Diffusion, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/29591f355702c3f4436991335784b503-Abstract-Conference.html) | Guides discrete diffusion through gradients in denoiser hidden states; saliency supports edit-constrained design. | Particularly relevant to the distinction between choosing positions and generating replacements. It requires a generative model and is not a drop-in logit optimizer. |
| [RSS: Relaxed Sequence Sampling for Diverse Protein Design, 2025 preprint](https://arxiv.org/abs/2510.23786) | Combines gradient-informed moves and language-model-informed jumps in continuous logit space. | Relevant to combining local guidance and broader exploration. Its reported gains concern computational designability and diversity; jumps do not make the maintained state a hard sequence. |
| [GGS: Improving Protein Optimization with Smoothed Fitness Landscapes](https://arxiv.org/abs/2307.00494) | Learns a smoothed fitness landscape and applies discrete MCMC. | Separates the quality of the modeled landscape from the search procedure; it adds modeling work beyond a frozen-predictor optimizer. |
| [CoSiNE: Conditionally Site-Independent Neural Evolution of Antibody Sequences, ICML 2026](https://arxiv.org/abs/2602.18982) | A learned continuous-time Markov chain over discrete sequences, with guided sampling. | Continuous **time** is not a continuous sequence relaxation. Useful adjacent work, but a different model family. |

### What the supplied BindCraft2 checkout actually does

Read-only inspection of `/storage/frank/BindCraft2`, clean at commit
`b6fb3ae15ab6a702f1156cffc11bcc3323c2fd06` (2026-10-20):

- [`build_stage_plan`](https://github.com/PacesaLab/BindCraft2/blob/b6fb3ae15ab6a702f1156cffc11bcc3323c2fd06/bindcraft/trajectory.py#L207)
  constructs `screen`, `refine`, `anneal`, then `harden`. The first stages use
  continuous parameters; hardening still uses sequence gradients.
- [`straight_through_one_hot`](https://github.com/PacesaLab/BindCraft2/blob/b6fb3ae15ab6a702f1156cffc11bcc3323c2fd06/bindcraft/sequence_optimization.py#L38)
  presents a hard sequence in the forward computation while retaining a soft
  gradient path. Hard model input therefore does not imply gradient-free search.
- [`run_sequence_mutation_stage`](https://github.com/PacesaLab/BindCraft2/blob/b6fb3ae15ab6a702f1156cffc11bcc3323c2fd06/bindcraft/trajectory.py#L152)
  uses forward predictions and weighted design loss. Its mutation sampler uses
  confidence-weighted position choices and stored sequence values; this stage
  does not call `sequence_gradients` for fresh gradient-ranked proposals.
- [`select_best_sequence`](https://github.com/PacesaLab/BindCraft2/blob/b6fb3ae15ab6a702f1156cffc11bcc3323c2fd06/bindcraft/sequence_optimization.py#L212)
  retains the lower design loss. Final campaign acceptance and ranking occur
  later, with redesign, validation and filters. Design loss and final ranking
  should not be conflated.

### Relationship to the local pipeline

Our `bindcraft` branch is Mosaic's implementation of a BindCraft-style staged
optimizer, operating with OpenDDE and local losses. It already includes a
straight-through hard stage. It returns probabilities, which the seeder
projects to a budget-feasible sequence. The discrete harness then starts from
that sequence; it does not receive the full continuous optimizer state.

Our discrete proposals use composite-loss gradients, while default retention uses
ipSAE. The optional Pareto mode now retains confidence–pose tradeoffs (§35).
BindCraft2's mutation polishing uses design-loss selection and forward
predictions. These are material differences, not evidence that one must win.
Both are sequential hybrids; the local experiment has not tested every possible
way of coupling continuous and discrete search.

**Interpretation for the project:** the literature supports studying hybrid
methods. The local results in [P17_JN1.md](../P17_JN1.md), §§30–34, establish
some gradient attribution but no demonstrated advantage from the tested
continuous-seeding handoff. That comparison cannot reject hybrid methods in
general. The relevant comparison remains the quality of final feasible discrete
outputs under comparable total compute, with selection and evaluation metrics
reported separately.

### Verification scope

The update checked the supplied BindCraft2 stage, optimizer and mutation code,
its official documentation, published RSO and Fast SeqProp full-text passages,
RSS full text, and primary publication/abstract pages for the remaining papers.
BindCraft 1's publication record and official algorithm description were
available; full-text access was blocked on this pass. Paper experiments and
BindCraft2 model runs were not reproduced. The older detailed note below retains
its own access limits; its absence-of-literature claims are not exhaustive.

### Follow-up: enforcing a small edit budget during search

The following primary sources were checked specifically for the five-edit
question. They support different components, not a complete validated local method.

| Paper | Relevant mechanism | Limit of the evidence |
|---|---|---|
| [LaMBO-2 / NOS, NeurIPS 2023, §4.2 and Fig. 6](https://papers.neurips.cc/paper_files/paper/2023/file/29591f355702c3f4436991335784b503-Paper-Conference.pdf) | Samples a bounded set of edit positions using gradient saliency, then generates replacements while preserving unselected positions. Position selection was influential in its limited-edit ablation. | A learned diffusion/value-model system; not evidence that a changing top-five gate inside OpenDDE works. A small edit count also does not guarantee pose preservation. |
| [Fast SeqProp, 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8527647/) | Evaluates discrete samples while updating continuous sampling parameters through a straight-through approximation. | Supports the hard-forward/continuous-update bridge, not an exact parent-relative five-edit guarantee. |
| [PPDE, §3–4](https://arxiv.org/html/2212.09925) | Builds a path of gradient-informed discrete edits and accepts or rejects its combined endpoint, avoiding exhaustive enumeration of multi-edit neighborhoods. | Path length is a proposal-size limit, not a cumulative distance cap from the original sequence. First-order proposals still approximate interactions. |

**Local inference:** test budget-aware position choice and joint endpoint
evaluation before committing to a continuously learned position gate. The
original reference must remain the budget anchor across feedback rounds; a
fresh allowance per round would solve a different problem. An expected-edit
penalty alone is not an exact bound on every sampled sequence.

The proposed feedback loop is agreed in principle. Its mechanism, objective
consistency, and recovery performance require separate tests. The concrete
consistency checks and their current status are recorded in
[P17_JN1.md §35.4](../P17_JN1.md#354-objective-consistency-what-is-checked-and-what-is-pending).

---

## October 7 review, with corrections


Scope note: this covers (a) whether optimizing a *relaxed* (continuous, simplex- or
logit-valued) sequence beats taking discrete moves, and on what evidence; (b) what
is known about the loss incurred when the relaxed result is rounded back to a
discrete sequence; (c) how published methods initialize the relaxed variable, and
whether any of them anchors it to a starting sequence under a hard edit budget;
and (d) whether anyone separates *which position to change* from *what to change
it to*. Reviewed 2026-10-07.

Framing, to keep this on the proof-of-concept path: the object of study is a
**discrete optimization problem under a hard distance constraint from a given
start** — pick at most *k* substitutions at a fixed set of designable positions so
as to improve a frozen predictor's own scalar readouts. Continuous relaxation is
one reparameterization of that problem. Everything below is about the
reparameterization's cost and benefit, not about molecules.

This note was written to answer a design question raised in chat: if a relaxed
stage runs *before* the discrete search, does it throw away the information in
the starting sequence? See §22.5, §23.4, §24.6 and §27 of
[P17_JN1.md](../P17_JN1.md) for the measurements it bears on.

---

## Q1. Does optimizing a relaxed sequence actually beat discrete moves, and on what evidence?

### Takeaway

**RSO reports an advantage in its benchmark, but this does not establish a
universal ordering of continuous and discrete methods.** The relaxed-sequence-space (RSO) paper is the
only source located that runs relaxed, rounded-discrete and MCMC variants against
each other inside one framework. No source compares a relaxed stage against a
discrete search *under an edit budget measured from a parent sequence*, which is
this project's regime.

### Cited Findings

- RSO benchmarked three approaches: **"gradient-descent with relaxed sequence (GD
  relaxed); GD with argmax() and one-hot encoded sequences (GD hard); and
  Markov-Chain-Monte-Carlo search (MCMC)."** GD-relaxed **converged in ~20
  iterations**, while **GD-hard and MCMC failed to converge within the tested
  iteration count** — [RSO, bioRxiv 2023.02.24.529906](https://www.biorxiv.org/content/10.1101/2023.02.24.529906v1.full)
- RSO's setting is **unconditional hallucination**: no starting sequence, no
  mutation budget, no distance penalty. The relaxed representation is used
  precisely to allow **unconstrained** exploration — [ibid.](https://www.biorxiv.org/content/10.1101/2023.02.24.529906v1.full)
- Relaxed Sequence Sampling (RSS, 2025) characterizes the prior art it improves on
  as relying on **"single-path gradient descent"** that **"ignore[s] sequence-space
  constraints, limiting diversity and designability"** — i.e. the known weakness of
  a plain relaxed descent is mode collapse, not slow convergence — [RSS, arXiv 2510.23786](https://arxiv.org/abs/2510.23786)
- RSS reports, against RSO at matched compute over ten complexes at a 3.5 Å
  self-consistency threshold: **~5.3× more designable structures** (one sampled
  sequence per structure), **~5×** (eight sequences per structure), and **53.2 vs
  22.6 distinct structural clusters, a 2.4× increase** — [RSS full text](https://arxiv.org/html/2510.23786v1)
- The gradient-based discrete-MCMC line (PPDE / EvoProtGrad) is explicitly
  positioned as working with pretrained models **without using continuous
  relaxations**, and controls its move size by drawing the number of substitutions
  from a Poisson distribution — [PPDE, Mach. Learn.: Sci. Technol.](https://iopscience.iop.org/article/10.1088/2632-2153/accacd) *(mechanism from search summary plus abstract, not full text)*

### Inferences

- RSO's result is the strongest available argument for putting a relaxed phase in
  front of a discrete search, and it is *not* a small effect — "converged in ~20
  iterations" versus "did not converge" is a qualitative difference. But three
  things bound its transfer: it is a within-framework comparison, the task has no
  distance constraint, and "GD hard" there means rounding **at every step**, which
  is a weaker discrete baseline than a gradient-informed discrete search with an
  acceptance rule.
- RSS's criticism of single-path relaxed descent (diversity collapse) is the
  continuous-side analogue of the concern §21.6 investigated on the discrete side
  (whether a population preserves distinct edit sets). Both literatures arrive at
  "one trajectory is not enough" independently.
- Nothing here establishes that a relaxed phase helps when the feasible set is a
  small ball around a parent. The relaxed representation's advertised advantage is
  freedom of exploration, and a hard edit budget is exactly a restriction on that
  freedom.

### Gaps

- No head-to-head of relaxed versus discrete search under a hard WT-distance cap.
- All reported wins are measured by **designability / self-consistency** (does an
  independent model reproduce the design) and **diversity** (cluster counts), never
  by recovery of a *specified* arrangement. The companion notes already record that
  no surveyed pipeline publishes a site-recovery rate.

---

## Q2. Does the relaxed optimum survive rounding back to a discrete sequence?

### Takeaway

**Rounding can lose the benefit of relaxation.** The published RSO paper
reports structural disagreement after direct argmax conversion and uses
ProteinMPNN for sequence generation. This is evidence for its pipeline, not a
universal claim that rounding always fails or every method needs a separate decoder.

### Cited Findings

- **Correction from the published RSO text (2026-10-22):** direct argmax
  conversion caused structural disagreement; sequences obtained after annealing
  had poor experimental success. The earlier insolubility wording conflated
  distinct outcomes. [Published RSO](https://pmc.ncbi.nlm.nih.gov/articles/PMC11734486/)
- RSS avoids the problem by **never binarizing during search**: logits stay
  continuous, their softmax gives per-position distributions, and a discrete model
  is applied only at final evaluation. No intermediate rounding step exists —
  [RSS full text](https://arxiv.org/html/2510.23786v1)
- BindCraft resolves it by **annealing the parameterization to one-hot inside the
  optimization**, in four stages: a soft stage whose representation is
  `(1 − λ)·logits + λ·softmax(logits/T)` with `λ = (step+1)/iterations` and
  `T = 1.0`; then **45 iterations** of softmax with temperature lowered each step;
  then a hard one-hot stage; then greedy refinement — [BindCraft, bioRxiv 2024.09.30.615802](https://www.biorxiv.org/content/10.1101/2024.09.30.615802v1.full)
- Fast SeqProp's mechanism for the same problem is a **straight-through estimator**:
  the loss is evaluated on the **hard, discrete** sequence while gradients propagate
  through the soft relaxation, combined with normalization across the parameters of
  the input sequence distribution. Reported **up to 100-fold faster convergence**
  and improved optima, motivated explicitly by **"input parameters becoming skewed
  during optimization"** — [Fast SeqProp, arXiv 2005.11275](https://arxiv.org/pdf/2005.11275) *(see access limits — abstract and indexed summaries of the published version only)*

### Inferences

- Staged hardening, straight-through sampling and inverse-folding decoding
  are different approaches to the continuous/discrete boundary. They are not
  mutually exclusive or an exhaustive taxonomy. Their existence does not
  establish which handoff is best for this project's constraints.
- Fast SeqProp's framing is the most relevant diagnosis for a ranking built on
  margins in probability space. Once the relaxed variable saturates near a simplex
  vertex, differences between the preferred and the parent entry compress toward
  their bound, so a top-*k* ordering taken from those differences degrades toward
  arbitrary exactly in the regime where the optimizer is most confident. Their fix
  is structural (normalize the parameters, score the hard sequence), not a
  post-hoc correction to the ranking.
- RSO's "~20 iterations" and RSS's "5×" therefore come with a hidden cost that is
  never charged to the relaxed stage: a second, discrete model is doing repair work
  afterwards. A pipeline that omits that stage is not getting the same deal.

### Gaps

- Nobody quantifies the rounding loss. RSO's evidence includes structural disagreement and experimental outcomes, not a measured drop in the optimized objective between
  the relaxed point and its rounding.
- No source evaluates rounding **under a budget truncation** — i.e. keeping only
  the *k* largest changes and reverting the rest. That operation has no published
  validation at all.

---

## Q3. How is the relaxed variable initialized, and does anyone anchor it to a starting sequence?

### Takeaway

**Every method located initializes from noise, and none anchors to a parent
sequence or imposes a hard edit budget — because none of them has a parent.** All
are solving de novo design. The anchored, budgeted regime this project works in is
*outside* the surveyed literature, so there is no published initialization to copy
and no published result that the noise initialization is appropriate here.

### Cited Findings

- RSO initializes logits from the **"Softmax of the Gumbel distribution"**. No
  alternative initializations (parent-anchored, profile-based) are discussed or
  compared — [RSO](https://www.biorxiv.org/content/10.1101/2023.02.24.529906v1.full)
- BindCraft initializes with a **random sequence** for the designed chain,
  predicted in single-sequence mode, with a structural template supplied for the
  target only — [BindCraft](https://www.biorxiv.org/content/10.1101/2024.09.30.615802v1.full) · [project wiki](https://github.com/martinpacesa/BindCraft/wiki/De-novo-binder-design-with-BindCraft)
- RSS's Algorithm 1 specifies only **"Initialize logits ℓ₀"**; no initialization
  scheme is stated — [RSS full text](https://arxiv.org/html/2510.23786v1)
- RSS imposes **no mutation budget and no sequence-distance penalty**. Its
  sequence-model term encourages agreement with learned sequence statistics but
  does not constrain Hamming distance from any reference — [ibid.](https://arxiv.org/html/2510.23786v1)
- RSO likewise constrains neither a starting sequence nor an edit count — [RSO](https://www.biorxiv.org/content/10.1101/2023.02.24.529906v1.full)
- The nearest external task match found is CoSiNE, which evaluates **constrained
  local optimization under a strict budget of five substitutions restricted to the
  designable loop positions**, on a SARS-CoV-1 case. Its baselines are a genetic
  algorithm and a Product-of-Experts sampler steering two pretrained sequence
  models. **No continuous-relaxation method and no structure-predictor-gradient
  method is among its baselines**, and its objective is a learned fitness/liability
  oracle rather than a structure predictor's own confidence and placement
  readouts — [CoSiNE, arXiv 2602.18982](https://arxiv.org/pdf/2602.18982) *(see access limits)*

### Inferences

- A noise initialization is correct for the regime these papers occupy and is a
  category error in this one. It is not a quirk to be inherited: with nothing to
  anchor to, random logits are the only sensible choice, and the moment a parent
  exists the choice reopens.
- The concrete cost here is quantifiable from the project's own objective rather
  than from the literature. `EditBudget` is a hinge `relu(E(s) − budget)` anchored
  to the parent one-hot at weight 5.0, and its own docstring notes it is **linear
  in `s`** and "pulls toward `s_ref` only when the budget is exceeded". At a
  near-uniform start over 29 designable positions, `E(s) ≈ 29 × (1 − 1/20) ≈ 27.6`,
  so the weighted hinge is `5.0 × relu(27.6 − 5) ≈ 113` against roughly 34 for
  every other term combined (§27.5's per-term table). Because the term is linear,
  its gradient is a *constant* pull toward the parent that does not decay until the
  budget is nearly satisfied. The early trajectory is therefore dominated by
  returning to the start it was initialized away from. **This is inference from the
  project's own loss definition, not a finding from any paper.**
- CoSiNE establishes that a five-substitution, position-restricted refinement task
  is a recognized benchmark setting, which is useful context. It does not bear on
  the relaxed-versus-discrete question, since neither family is represented in its
  baselines.

### Gaps

- No published guidance on initializing a relaxed variable when a parent sequence
  exists and must be respected.
- No published method combines a relaxed representation with a **hard** distance
  cap. Soft hinges are the only mechanism in evidence, and they are not caps.

---

## Q4. Does any published method separate position selection from identity selection?

### Takeaway

**Yes — RSS does exactly this, and it is the closest published mechanism to the
asymmetry §22.5 measured here.** Its jump kernel chooses *which* positions to
change from gradient norms and *what* to change them to from a pretrained sequence
model's conditional distribution, with a Metropolis–Hastings correction that keeps
the composite move reversible.

### Cited Findings

- RSS runs a **mixture kernel in logit space** with two move types — [RSS full text](https://arxiv.org/html/2510.23786v1):
  - **Walks (MALA):** proposes `ℓ′ = ℓ − ηg + √(2η/β)·ξ` where `g` is the gradient
    of the composite energy and `ξ` is Gaussian noise, with a
    Metropolis-adjusted-Langevin acceptance that corrects the proposal density to
    maintain detailed balance against a Boltzmann target.
  - **Jumps (sequence-model-guided):** samples a mask set `S` of positions **based
    on gradient norms**, then at each masked position draws a forward token from
    the pretrained sequence model's **conditional distribution** and a reference
    token uniformly, applying an additive swap `ℓ′ᵢ = ℓᵢ + γ(e_{y⁺} − e_{y⁻})`. The
    acceptance ratio carries **three** factors — the target energy ratio, the
    mask-proposal ratio, and a sequence-model conditional ratio — to ensure
    reversibility.
- RSS validates its differentiable sequence-model surrogate against the discrete
  model it approximates: **KL divergence 0.034**, **gradient correlation 0.987**,
  and rank correlation **1.0** on mean scores — [ibid.](https://arxiv.org/html/2510.23786v1)
- RSS's stated limitations: dependence on learned surrogates introduces model bias;
  **"over-regularization from large [sequence-model] weight λ"** can over-constrain
  exploration; evaluation is restricted to **fixed-length** designed chains; and the
  jump moves' sequence-model forward passes add overhead that is **not fully
  analyzed** — [ibid.](https://arxiv.org/html/2510.23786v1)
- LaMBO-2's own ablation separates saliency-based position selection from gradient
  guidance during generation, and in its limited-edit setting **position selection
  had the larger effect** — recorded in
  [search_policy_and_pose_literature_review.md](../search_policy_and_pose_literature_review.md) §4

### Inferences

- The project measured the two halves coming apart (§22.5: position 108 identified
  by three of four search seeds, the correct residue chosen by none; §20.9: a
  position drawing 31 proposals across nine different accepted identities). RSS is
  that observation as an algorithm — trust the gradient for *where*, delegate *what*
  to a model of sequence statistics. That is a mechanism match, not evidence that
  it works on this objective.
- The project already loads a pretrained sequence model, used only as a naturalness
  penalty at weight 0.10. RSS's jump kernel is the argument for promoting it from a
  penalty term to the **identity proposal distribution**, with positions still
  selected by gradient magnitude. The smallest version of this is the jump kernel
  without the MALA half, which needs no relaxed representation at all and so sits
  inside the existing discrete harness.
- RSS also demonstrates a worked Metropolis–Hastings correction for a *composite*
  proposal, including the awkward part — a reversible masked-position proposal. §5's
  known issue 1 and §8.2 record that this project computes a proposal log-probability
  and then discards it, so its acceptance rule does not establish sampling from any
  target. RSS is a template for fixing that, if correctness of sampling is ever
  wanted; §8.2's caution still applies, that correct sampling and effective
  optimization under fixed compute are different objectives.

### Gaps

- RSS's gains are reported as designability and diversity in a de novo setting. No
  result in it speaks to recovering a specified arrangement, and it carries no
  distance constraint, so its effect sizes do not transfer.
- Whether a sequence-statistics model supplies *useful* identity proposals for an
  interface objective is untested anywhere located. A model of what sequences look
  like is not a model of what forms a given interface, and §6's proxy caveat applies
  to it as much as to anything else.

---

## What this implies for the project

**Updated 2026-10-22.** The former list treated proposed mechanisms as proved
requirements. Anchored initialization is a choice, not a theorem; probability
margins alone cannot establish sequence quality; and a literature search cannot
prove that a handoff has no precedent. The local launcher now measures the
combined edit budget against the reference, so the old double-budget warning
does not describe that implemented path. A comparison must verify its actual
budget anchor rather than infer one from the names of the stages.

The current reading map above supplies the applicable distinctions. None of
these papers establishes the cause of this project's inconclusive seeding
comparison or proves that changing representation will recover a reference pose.

## Historical access limits and provenance

- **Fast SeqProp (October 7 access; published PMC text checked October 22):** the arXiv PDF would not parse and the publisher version is
  behind an authentication redirect. Its mechanism and headline numbers here come
  from the abstract and indexed summaries of the published version, **not** verified
  full text. The initialization-sensitivity question put to it is unanswered.
- **CoSiNE:** the full text is ~143k characters and only the first ~100k were read;
  quantitative results for the five-substitution experiment are deferred to a
  supplementary table that was not retrieved. A search summary additionally claimed
  an unbounded greedy hill-climbing baseline as a reference upper bound; **this was
  not confirmed in the body text and should be treated as unverified.**
- **RSS:** read from the full HTML version. Initialization is genuinely unspecified
  in the text, not merely unretrieved.
- **RSO:** the original note used the preprint. The published-text correction
  above replaces the insolubility attribution; the older iteration comparison
  was not independently rechecked or reproduced in this update.
- **PPDE:** mechanism from the abstract and a search summary only.
- No paper's experiments were reproduced, and no claim in this note was tested
  against this project's models.

## Sources

- [Efficient and scalable de novo protein design using a relaxed sequence space (RSO)](https://www.biorxiv.org/content/10.1101/2023.02.24.529906v1.full)
- [Relaxed Sequence Sampling for Diverse Protein Design (RSS)](https://arxiv.org/abs/2510.23786) · [full text](https://arxiv.org/html/2510.23786v1)
- [Fast differentiable DNA and protein sequence optimization for molecular design (Fast SeqProp)](https://arxiv.org/pdf/2005.11275) · [published version](https://link.springer.com/article/10.1186/s12859-021-04437-5)
- [BindCraft: one-shot design of functional protein binders](https://www.biorxiv.org/content/10.1101/2024.09.30.615802v1.full) · [project wiki](https://github.com/martinpacesa/BindCraft/wiki/De-novo-binder-design-with-BindCraft)
- [Conditionally Site-Independent Neural Evolution of Antibody Sequences (CoSiNE)](https://arxiv.org/pdf/2602.18982)
- [Plug & play directed evolution of proteins with gradient-based discrete MCMC (PPDE)](https://iopscience.iop.org/article/10.1088/2632-2153/accacd)
- [Improving protein optimization with smoothed fitness landscapes (GGS)](https://arxiv.org/html/2307.00494v3)
