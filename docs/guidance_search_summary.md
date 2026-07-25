# Guidance and Search Summary

This note summarizes the current conclusion about how Mosaic should perform
guided antibody optimization, with emphasis on:

- what soft vs hard selection means in practice
- why mutation-level greedy search is not enough
- why epistasis pushes us toward proposal-level search
- where Feynman-Kac steering ideas help
- what a practical hybrid algorithm should look like

This is a companion to `docs/guidance_design_notes.md`. That note focuses on
the coordinate-gradient controller. This note focuses on the search object and
selection policy.

---

## 1. Notation and scope

This note uses the same structure notation as
`docs/guidance_design_notes.md`:

- `x_t`: noisy structure state at step `t`
- `x0_hat`: current denoised clean estimate inside one reverse step
- `x_final`: final structural sample after the reverse trajectory finishes
- `design proposal`: the joint sequence-structure candidate derived from
  `x_final`, optionally followed by decode, polish, and refold

This matters because proposal-level selection should act on the finished
candidate or a candidate derived from `x_final`, not directly on `x0_hat`.

---

## 2. Core problem

The current pipeline asks a single guided diffusion trajectory to do too much.

It must simultaneously:

1. move the structure toward a better interface
2. keep the sequence plausible
3. stay within an edit budget or locality prior
4. preserve pose enough to survive refolding

That is a difficult search problem, especially for antibody affinity
optimization where the useful signal is often epistatic and context-dependent.

The main consequence is:

> the search object should not be an isolated mutation.

It should be a **joint CDR hypothesis**: a coupled sequence-structure proposal.

---

## 3. Why isolated mutation search is the wrong default

A single mutation can look bad on its own but become good when paired with:

- a compensating neighboring substitution
- a loop-shape change
- a different rotamer assignment
- a shifted hydrogen-bond network
- a slightly changed interface pose

This is exactly the kind of epistasis that matters in antibody optimization.

If the search is built around residue-by-residue greedy selection, it will
prune away many of the combinations we actually care about.

So the following should **not** be the primary search strategy:

- rank mutations one-by-one
- accept or reject substitutions independently
- prune early at the residue level before context is known

That approach loses the coupled signal.

---

## 4. What should be selected instead

The correct unit of search is a **trajectory-level proposal**.

That means:

- one partial-diffusion trajectory proposes a coordinated CDR solution
- the proposal may contain multiple coupled edits
- the proposal is scored jointly, not residue-by-residue
- selection happens across proposals, not across independent mutations

This preserves epistasis.

In simple terms:

```text
bad:   mutation -> score -> keep/drop
good:  trajectory/proposal -> score jointly -> keep/drop
```

---

## 5. Soft selection vs hard selection

This distinction is about how the next proposal is chosen.

### 4.1 Hard selection

Hard selection is greedy.

```text
generate K proposals
score all K
keep the best one (or top M only)
```

This strongly exploits the current score.

Advantages:

- simple
- compute-efficient after scoring
- useful late in refinement

Disadvantages:

- brittle
- easy to collapse diversity
- easy to kill near-miss epistatic solutions too early

### 4.2 Soft selection

Soft selection is score-weighted exploration.

```text
generate K proposals
score all K
convert scores to weights
sample or resample using those weights
```

For example:

```text
w_i ∝ exp(score_i / T)
```

Advantages:

- preserves diversity
- better at keeping multiple competing local solutions alive
- better suited to uncertain or noisy objectives

Disadvantages:

- can spend compute on mediocre proposals
- needs a temperature or weighting schedule

### 4.3 What we should use

For our use case:

- **early search** should be closer to soft selection
- **late refinement** can move toward hard selection

So the right answer is probably not “hard or soft”, but a staged hybrid:

- soft earlier
- harder later

---

## 6. Why Feynman-Kac steering matters

Feynman-Kac (FK) steering and Feynman-Kac Correctors (FKC) provide a principled
way to do proposal-level selection over multiple stochastic trajectories.

The basic idea is:

1. run multiple particles / trajectories
2. evaluate each particle using a reward or potential
3. upweight promising particles
4. downweight or kill poor particles
5. resample and continue

This is directly relevant because it addresses a weakness of raw single-path
gradient guidance:

> a single guided trajectory is a poor container for multimodal search.

FK/FKC spreads the search across multiple trajectories instead of forcing one
trajectory to carry all uncertainty.

### 5.1 Why it is attractive for us

It helps with:

- trajectory-level diversity
- black-box or weakly differentiable rewards
- multi-objective search without crude raw gradient summation
- preserving epistatic multi-edit hypotheses

### 5.2 What it does not solve for free

It does **not** automatically make the objective good.

If the scoring function is noisy or misaligned, FK/FKC will simply resample
toward the wrong region more efficiently.

It is also more expensive than a single trajectory.

---

## 7. Where gradient guidance still fits

FK/FKC is not a replacement for the entire guidance controller.

It solves a different problem.

- **gradient guidance** is good for making a local continuous move within a
  single trajectory
- **FK/FKC-style selection** is good for deciding which trajectories survive

So these methods are complementary.

One important caveat from the controller side still applies here:

- even a well-designed proposal-selection policy does **not** guarantee that
  the in-trajectory guidance remains compatible with the pretrained BoltzGen
  prior

That unresolved issue is documented explicitly in
`docs/guidance_design_notes.md` under **Prior compatibility with the BoltzGen
model**.

The practical interpretation is:

> use gradients inside a particle, and use particle selection across particles.

That is a better decomposition than asking one raw coordinate gradient to do
all exploration and all selection.

---

## 8. Recommended hybrid for Mosaic

The best practical direction is a hybrid:

### Inside each trajectory

Use a **mild, controlled coordinate guidance**:

- BoltzGen denoiser predicts `x0_hat`
- BoltzGen IF gives a differentiable soft sequence bridge
- evaluate smooth in-loop objectives
- apply masked, normalized, projected, trust-region-clipped guidance

This is the controller described in `guidance_design_notes.md`.

### Across trajectories

Use **proposal-level selection**:

- start several guided diffusion trajectories from the same parent
- decode each into a joint sequence-structure proposal
- score proposals jointly
- soft-resample or top-k keep proposals
- continue only surviving proposals

This is the search-level role of FK/FKC ideas.

---

## 9. Two different schedules

This note uses the word "early" and "late" along the **search axis**, not the
within-trajectory diffusion-noise axis.

That means:

- **early search** = earlier proposal rounds / earlier outer iterations
- **late search** = later proposal rounds / later outer iterations

This is different from:

- **high sigma / low sigma**

inside one trajectory.

The clean split is:

- the **noise-axis schedule** belongs to the gradient controller
- the **search-axis schedule** belongs to proposal survival and soft/hard
  selection

---

## 10. What should be scored in each stage

### 8.1 In-trajectory guidance

Use smooth differentiable surrogates:

- ipTM-style confidence
- interface PAE / iPAE penalty
- optional pTM-energy-like stabilizer
- weak naturalness prior
- weak locality / edit restraint

These should shape the local move.

### 8.2 Proposal-level selection

Use somewhat stronger but still affordable criteria:

- in-loop confidence surrogate
- decoded-sequence plausibility
- edit count or locality
- optional shallow structural proxy

This is where soft vs hard selection acts.

### 8.3 Final downstream ranking

Use stronger validation metrics:

- ipSAE
- refolded ipTM
- RMSD / pose filter
- interface contacts / H-bonds / salt bridges

This is the final acceptance stage.

---

## 11. Why ipSAE should not lead diffusion-time guidance

ipSAE is useful, but it should act as a downstream rank/filter, not the
diffusion-time guidance signal.

The reason is simple:

- its hard PAE-cutoff mask and best-row/max-style reduction are not suitable
  as a smooth differentiable per-step objective
- ipTM and interface PAE/iPAE are the intended differentiable binding guidance
  terms
- it is better suited to comparing completed candidates
- it is most meaningful after the structure has been decoded/refolded

So a sensible split is:

- **during diffusion**: ipTM + interface PAE/iPAE
- **after generation/refold**: ipSAE for ranking

---

## 12. How a mutation is retained or rejected

This needs to be explicit.

A mutation should not be kept just because it looks intuitively favorable in
isolation. It should be retained only if the **joint proposal** it belongs to
survives the multi-stage selection process.

In practice a mutation survives when:

1. the proposal improves the main binding/confidence objective enough
2. the proposal remains sequence-plausible enough
3. the proposal does not violate locality/edit constraints too badly
4. the proposal survives downstream pose/confidence validation

So mutation retention is proposal-level and context-dependent.

This is important because many “good” mutations are only good conditionally,
after neighboring changes or loop movement.

---

## 13. Recommended staged search loop

The practical loop should be:

```text
for each parent complex:
    initialize K trajectories / particles

    for each trajectory:
        run guided partial diffusion with mild local gradient control
        decode with BoltzGen IF

    score K joint proposals with an intermediate reward

    if early stage:
        soft-resample proposals
    else:
        keep top proposals more aggressively

    optionally continue survivors for another round

    refold only the final survivors
    rank by ipSAE + ipTM + RMSD/pose filter + interface metrics
```

This preserves epistasis, keeps search diversity alive, and avoids spending the
most expensive refolding compute on every proposal.

---

## 14. Relation to `edit_budgeted_greedy_descent`

The existing `edit_budgeted_greedy_descent` should not be the main search
engine in this design.

Why:

- it is local
- it is residue-level
- it is greedy

Those are exactly the properties that make it weak for strongly epistatic joint
CDR optimization.

So the intended role is:

- **not** the central search policy
- **optional shallow late-stage polish only**

The preferred architecture is:

```text
guided diffusion proposals
-> optional shallow polish
-> refold / rank
```

not:

```text
guided diffusion
-> heavy greedy residue-level descent as the main optimizer
```

This should be treated as an architectural choice, not an implementation
detail.

---

## 15. FK/FKC caveats that must be checked empirically

### 15.1 Proxy correlation

If proposal survival uses a cheap intermediate proxy, that proxy must be shown
to correlate with the expensive downstream metric we actually care about.

At minimum, this should be audited on a modest sample by comparing:

- intermediate proposal proxy
- final refolded ipSAE
- final refolded ipTM
- RMSD / pose pass

If the proxy is poorly aligned, early resampling can kill proposals that would
have succeeded after refolding.

### 15.2 Particle degeneracy

FK/FKC-style resampling can lose diversity if too many surviving proposals
trace back to the same ancestor.

So even a light first pass should log at least:

- ancestor identity
- number of unique surviving ancestors
- optionally effective sample size (ESS)

For the first implementation pass, a simple mitigation is:

- do at most one resampling round
- keep a no-resample baseline arm
- optionally cap the number of children per parent

These checks are important because the value of FK/FKC here is diversity
preservation; if diversity collapses immediately, the method loses its main
advantage.

---

## 16. Recommended first implementation pass

Do **not** jump directly to a full particle-SMC implementation at every single
diffusion step.

Start with a lighter version:

1. run `K` independent guided partial-diffusion trajectories from the same
   parent
2. decode each as a joint CDR proposal
3. score them jointly
4. keep top `M`, or soft-resample them
5. refold only survivors

This captures the main value of FK/FKC without forcing a full rewrite of the
sampling loop.

---

## 17. Bottom line

The main conclusion is:

> do not make isolated mutations the search object.

Instead:

- keep the search object as a **joint sequence-structure proposal**
- use **mild gradient guidance inside a trajectory**
- use **soft/hard proposal selection across trajectories**
- preserve epistasis by scoring coupled CDR hypotheses jointly
- use **ipSAE mainly after refolding**, not as the raw in-loop gradient target

This is the most coherent path for Mosaic if the goal is low-edit antibody
optimization rather than unrestricted de novo generation.

---

## 18. Relevant references

- **Gradient Guidance for Diffusion Models: An Optimization Perspective**  
  https://arxiv.org/abs/2404.14743

- **Universal Guidance for Diffusion Models**  
  https://proceedings.iclr.cc/paper_files/paper/2024/hash/dfbb3d1807b21dadee735eb75069ada4-Abstract-Conference.html

- **PCGrad: Gradient Surgery for Multi-Task Learning**  
  https://arxiv.org/abs/2001.06782

- **Feynman-Kac Correctors in Diffusion: Annealing, Guidance, and Product of Experts**  
  https://arxiv.org/abs/2503.02819

- **A General Framework for Inference-time Scaling and Steering of Diffusion Models**  
  https://arxiv.org/html/2501.06848v1

- **Controllable protein design through Feynman-Kac steering**  
  https://arxiv.org/html/2511.09216v1

- **Protein Design with Guided Discrete Diffusion**  
  https://papers.nips.cc/paper_files/paper/2023/hash/29591f355702c3f4436991335784b503-Abstract-Conference.html
