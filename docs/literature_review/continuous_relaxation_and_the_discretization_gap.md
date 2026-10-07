# Continuous relaxation of a discrete sequence, and the cost of rounding it back

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

**Yes, decisively, on the one direct benchmark I found — but in the unconstrained
de novo regime, and scored by self-consistency rather than by agreement with a
specified reference arrangement.** The relaxed-sequence-space (RSO) paper is the
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

**No, and this is measured rather than argued.** RSO reports that taking its
relaxed result to one-hot directly produced unusable designs, which is *why* a
separate discrete refinement model is a required pipeline stage rather than a
convenience. Every relaxed method located ends with a discrete model cleaning up
after it. **No source consumes `argmax` of its own relaxed optimum as the answer.**

### Cited Findings

- RSO: the discrete conversion is post-hoc, **"saved as pdb files with a
  placeholder sequence generated by taking the argmax of the sequence logits"** —
  and directly one-hot-encoding the relaxed sequences **"resulted in largely
  insoluble proteins"**, which the authors give as the reason a separate
  inverse-folding refinement stage is necessary despite successful optimization.
  No quantified size for the rounding loss is given — [RSO](https://www.biorxiv.org/content/10.1101/2023.02.24.529906v1.full)
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

- The field has exactly three answers to the rounding problem, and they are
  mutually exclusive in practice: **anneal to discrete inside the optimization**
  (BindCraft), **never discretize and correct the search instead** (RSS), or
  **round and then repair with a separate discrete model** (RSO). A fourth option —
  round once, by truncation, and treat the result as a finished starting point — is
  not used by any source located, and RSO's insolubility finding is direct evidence
  against it.
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

- Nobody quantifies the rounding loss. RSO's evidence is a qualitative outcome
  ("largely insoluble"), not a measured drop in the optimized objective between
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

Stated as consequences, not recommendations, with the relevant sections named.

1. **Anchoring the relaxed initialization is forced rather than preferred.** No
   published method anchors, because none has a parent; the noise initialization is
   inherited from a regime with no starting point. The hinge arithmetic in Q3
   explains what the current initialization costs.
2. **Taking `argmax` of a budget-truncated relaxed result is the one option the
   literature does not use.** The three published resolutions are anneal-to-discrete
   inside the optimization (BindCraft), never discretize (RSS), or round-then-repair
   with a separate model (RSO). Round-and-consume has no precedent and RSO's
   insolubility result argues against it.
3. **If the round-then-repair shape is chosen, the budget accounting is the hazard.**
   §22.5 established that the edit cap is measured from the search's own start, so a
   seed at *k* edits followed by a fresh *k*-edit budget permits `2k` from the true
   parent — and §27.3 measured that more edits produced more of the inverted
   placement mode. A pose regression under that arrangement would be attributable to
   the doubled budget, not to the relaxed stage.
4. **Ranking kept substitutions by a margin in probability space is the known failure
   mode, not an implementation detail.** Fast SeqProp's normalization-plus-hard-loss
   is the published alternative. §8.7 already lists that paper as relevant to this
   handoff.
5. **The cheapest item with literature behind it needs no relaxed stage at all:**
   positions from gradient magnitude, identities from the pretrained sequence model
   (RSS's jump kernel, minus MALA). It targets §22.5's measured gap directly and is
   a change to the proposal distribution inside the existing harness.
6. **§24.6 is unaffected.** It rejected a relaxed representation as a *placement*
   fix, and nothing above connects the reparameterization to placement. The case in
   this note is about identity selection and budget traversal (§23.2), which is a
   different claim.

## Access limits and provenance

- **Fast SeqProp:** the arXiv PDF would not parse and the publisher version is
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
- **RSO:** read from the full text. The insolubility and iteration-count findings are
  from its own reporting; neither was reproduced.
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
