# Guidance Design Notes

This note explains how the current guided VHH workflow works, why the present
gradient update is too aggressive, and what a cleaner guidance controller
should look like for the next iteration.

The intended audience is someone reading the code and trying to decide how the
guidance term should be redesigned for low-edit antibody optimization.

---

## 1. Notation

The symbols below are used throughout this note.

- `x_t`
  - the current noisy structure state at diffusion step `t`
- `x0_hat`
  - the BoltzGen denoiser's current clean-structure estimate from `x_t`
  - this is **not** the final ranked candidate and **not** "the structure with
    the best ipSAE"
- `x0_guided`
  - the guided clean estimate after applying the coordinate update
- `x_final`
  - the final structural sample after the full reverse-diffusion trajectory
    finishes
- `s`
  - the soft sequence representation produced by the differentiable inverse-fold
    bridge from the current structure estimate
- `design proposal`
  - the joint sequence-structure candidate derived from `x_final`, optionally
    followed by decode, polish, and refold

The important distinction is:

```text
x_t -> x0_hat -> x0_guided -> ... -> x_final -> decoded/refolded candidate
```

Guidance acts on `x0_hat` inside the reverse process. Final ranking acts on
the finished candidate derived from `x_final`.

---

## 2. Current mechanism

At a high level, the current guided partial diffusion loop does this:

1. BoltzGen denoises the current noisy coordinates `x_t` and predicts a clean
   structure estimate `x0_hat`.
2. A differentiable inverse-folding bridge maps `x0_hat` to soft sequence
   probabilities.
3. Sequence / confidence losses are evaluated on that soft representation.
4. The gradient of the scalar loss with respect to `x0_hat` is subtracted from
   `x0_hat`.
5. The guided `x0_hat` is used in the EDM reverse update for the next step.

In compact form:

```text
x0_hat = D_theta(x_t, sigma)
s      = IF_phi(x0_hat)
L      = sum_k w_k L_k(s)
g      = dL / d(x0_hat)
x0_guided = x0_hat - lambda(sigma) * g
```

Where:

- `D_theta` is the BoltzGen denoiser.
- `IF_phi` is the inverse-folding bridge.
- `L_k` are task losses such as naturalness or structure-confidence surrogates.

This is the right overall shape, but the controller around `g` is still too
weak.

---

## 3. Why the current guidance is too brutal

The main problem is not that guidance exists. The problem is that the current
update is too close to a raw gradient subtraction in coordinate space.

The failure modes are:

1. **No explicit per-objective normalization**
   - different losses produce gradients on very different scales
   - one spiky objective can dominate a step

2. **No explicit trust region**
   - late in diffusion, even a moderate coordinate gradient can make a large
     semantic change
   - this is especially dangerous when `sigma` is already low

3. **No conflict handling between objectives**
   - binding/confidence gradients and naturalness gradients can push in
     different directions
   - simply summing them can produce unstable motion or over-regularization

4. **Late-step guidance remains too strong**
   - near the end of diffusion, the model prior is already strong
   - the auxiliary gradient should become conservative, not keep pulling hard

5. **The update acts directly in coordinate space**
   - this is necessary for our use case, but it means the controller must be
     more careful than sequence-only optimization

So the problem is not “we need more losses.” The problem is that the guidance
controller is underdesigned.

---

## 4. Design goal

For our use case, the guidance system should satisfy four constraints:

1. Improve interface quality or binding-confidence proxies.
2. Keep designs antibody-like enough to avoid sequence collapse.
3. Limit unnecessary edits, especially for affinity maturation tasks.
4. Avoid destabilizing the BoltzGen denoising trajectory.

That means the guidance should be **hierarchical**, not flat:

- **Primary objective**: binding / interface-confidence proxy
- **Secondary objectives**: naturalness, locality, edit restraint

The primary objective should lead. The secondary objectives should regularize.

---

## 5. Recommended architecture

### 4.1 Separate the gradient into components

Instead of one merged raw loss, compute separate coordinate gradients:

```text
g_bind = dL_bind / d(x0_hat)
g_nat  = dL_nat  / d(x0_hat)
g_edit = dL_edit / d(x0_hat)
```

Where:

- `L_bind` is a differentiable binding/confidence surrogate
- `L_nat` is an antibody naturalness prior
- `L_edit` is a locality or edit-restraint prior

### 4.2 Apply a strict design mask

All gradients should be masked to the designable region:

```text
g_k <- M_design * g_k
```

This ensures frozen framework / target atoms do not receive optimization
pressure.

### 4.3 Remove rigid drift

Before combining gradients, remove translation-like motion from the design
region:

```text
g_k <- g_k - mean_design_atoms(g_k)
```

This keeps the controller focused on local reshaping instead of drifting the
whole binder.

### 4.4 Normalize each gradient

Normalize by RMS magnitude over design atoms:

```text
g_k_norm = g_k / (rms_design(g_k) + eps)
```

This makes the merge depend on direction, not arbitrary scale.

### 4.5 Merge gradients asymmetrically

Use binding as the anchor direction. Auxiliary terms should not override it.

A practical merge is:

```text
g_total =
    g_bind_norm
  + alpha(sigma) * proj_compatible(g_nat_norm,  g_bind_norm)
  + beta(sigma)  * proj_compatible(g_edit_norm, g_bind_norm)
```

Where `proj_compatible` means:

- if the auxiliary gradient agrees with `g_bind`, keep it
- if it strongly conflicts, project away the conflicting component

This is the same general idea as PCGrad-style gradient surgery, but used here
as a controller for coordinate guidance rather than multi-task training.

### 4.6 Enforce a per-step trust radius

After merging, clip the final coordinate step:

```text
step = lambda(sigma) * g_total
step = clip_rms(step, tau(sigma))
x0_guided = x0_hat - step
```

`tau(sigma)` should shrink as `sigma` decreases.

This is the key guardrail missing from the current implementation.

---

## 6. Prior compatibility with the BoltzGen model

The controller must do more than keep the guidance step small. It must also
avoid pushing the sampler away from the structural patterns learned by the
pretrained BoltzGen denoiser.

This point is important enough to state explicitly:

> a bounded external gradient is still dangerous if it points consistently away
> from the model's learned structural prior.

So "trust radius" and "guidance decay" are necessary but not sufficient.
They control **magnitude**. They do not guarantee **prior compatibility of the
direction**.

### 6.1 What this means in practice

The current notes already propose:

- masking
- de-meaning
- per-objective normalization
- conflict-aware merge
- trust-region clipping
- sigma-dependent decay

Those controls are still useful, but they only answer:

- how much should the external objective move the sample?

They do **not** fully answer:

- is the direction of that move still consistent with what BoltzGen itself has
  learned as a plausible denoising trajectory?

### 6.2 Why this matters

If the guidance term is too disconnected from the learned BoltzGen prior, then
the system stops behaving like a guided generative model and starts behaving
like an unstable coordinate-space optimizer.

That creates exactly the failure mode we want to avoid:

- the model is no longer refining plausible samples
- it is being dragged toward an external objective that may exploit local
  artifacts of the bridge or the proxy loss
- the resulting sample may score well locally but decode or refold badly

### 6.3 Design principle

The controller should therefore be viewed as three coupled parts:

```text
external objective
+ prior-compatibility guard
+ step-size control
```

not just:

```text
external objective
+ step-size control
```

### 6.4 What is still unresolved

This document does **not** yet specify a final prior-compatibility mechanism.
That is a real remaining design question.

Reasonable candidate mechanisms include:

1. **Denoiser-consistency penalty**
   - penalize updates that move too far from a denoiser-consistent direction

2. **Forward-prediction / self-consistency loss**
   - shape the external objective through a loss tied back to the pretrained
     denoiser's own prediction structure

3. **Projection against a model-prior direction**
   - allow only the component of the external guidance that is compatible with
     the denoiser's own local geometry preference

4. **Prior-residual diagnostics**
   - even before a full mechanism is implemented, log a measure of how strongly
     the external guidance is fighting the model prior

### 6.5 Current status

At present, the design should be read as:

- the proposed controller fixes scale imbalance, objective conflict, and
  over-aggressive late-step motion
- it does **not yet** fully solve the stronger problem of explicitly tying the
  guidance direction back to the learned BoltzGen prior

That omission should be treated as deliberate and unresolved, not as something
already handled by trust-radius clipping alone.

---

## 7. Which losses should be used

### 5.1 Primary guidance during diffusion

Use a **smooth differentiable interface-confidence surrogate**, not a jagged
post hoc score.

Recommended candidates:

- `ipTM`-style confidence term
- interface `PAE` / iPAE penalty
- optional pTM-energy-like stabilizing term

These should drive diffusion-time guidance because they are smooth enough to
provide useful gradients.

### 5.2 Secondary guidance during diffusion

Use weak regularizers:

- antibody LM naturalness prior
- locality / edit restraint
- optional framework tether

The antibody LM should not dominate the structural search. Its role is to keep
the search out of obviously bad sequence regions.

### 5.3 Ranking after generation / refold

Use stronger downstream metrics for filtering and ranking:

- `ipSAE`
- refolded `ipTM`
- RMSD / pose filter
- interface geometry terms

`ipSAE` is valuable, but it is not the in-loop gradient target: the hard
PAE-cutoff mask and best-row/max-style reduction make it unsuitable as a
smooth differentiable per-step objective. Use it after refolding for ranking;
use differentiable `ipTM` and interface `PAE`/iPAE during diffusion.

---

## 8. Two different time axes

This redesign uses two different schedules. They should not be conflated.

### 7.1 Noise-axis schedule: inside one trajectory

This is the diffusion-time axis:

- current denoising step `t`
- current noise level `sigma_t`

This axis controls:

- `lambda(sigma)` for guidance strength
- `tau(sigma)` for trust-radius clipping
- the relative weight of structure, naturalness, and locality terms **within**
  a single trajectory

### 7.2 Search-axis schedule: across trajectories / outer rounds

This is the proposal-search axis:

- outer iteration index
- proposal generation round
- resampling stage across multiple trajectories

This axis controls:

- soft vs hard proposal selection
- how aggressively to prune proposals
- how much diversity to preserve across candidate trajectories

The clean rule is:

- **noise-axis schedule** controls local guidance inside a trajectory
- **search-axis schedule** controls which trajectories survive

---

## 9. Noise-dependent schedule

The guidance should change over the diffusion trajectory.

### Early diffusion: high noise

Goal: exploration.

- stronger binding/confidence guidance
- weak naturalness
- very weak edit/locality restraint

### Mid diffusion

Goal: shape the interface while beginning to regularize.

- binding still dominant
- naturalness starts to matter
- edit/locality penalty turns on

### Late diffusion: low noise

Goal: refinement, not re-planning.

- sharply reduce coordinate guidance magnitude
- increase conservatism
- naturalness and locality prevent last-step collapse

This is where a trust radius is most important.

---

## 10. Proposed algorithm

```text
Input:
  x_t                noisy coordinates at step t
  sigma_t            current noise level
  M_design           atom mask for designable region
  D_theta            BoltzGen denoiser
  IF_phi             differentiable BoltzGen inverse-fold bridge

Step 1: denoise
  x0_hat = D_theta(x_t, sigma_t)

Step 2: decode soft sequence representation
  p_seq = IF_phi(x0_hat)

Step 3: evaluate objectives
  L_bind = w_iptm * L_iptm(p_seq, x0_hat)
         + w_ipae * L_ipae(p_seq, x0_hat)
         + optional w_ptm * L_ptm_energy(p_seq, x0_hat)

  L_nat  = w_nat  * L_antibody_lm(p_seq)
  L_edit = w_edit * L_locality(x0_hat, x_parent)

Step 4: differentiate each objective separately
  g_bind = dL_bind / d(x0_hat)
  g_nat  = dL_nat  / d(x0_hat)
  g_edit = dL_edit / d(x0_hat)

Step 5: mask and center
  for g in {g_bind, g_nat, g_edit}:
      g <- M_design * g
      g <- g - mean_design_atoms(g)

Step 6: normalize
  for g in {g_bind, g_nat, g_edit}:
      g <- g / (rms_design(g) + eps)

Step 7: conflict-aware merge
  g_total = g_bind
          + alpha(sigma_t) * compat(g_nat,  g_bind)
          + beta(sigma_t)  * compat(g_edit, g_bind)

Step 8: trust-region clip
  delta = lambda(sigma_t) * g_total
  delta = clip_rms(delta, tau(sigma_t))

Step 9: guided clean estimate
  x0_guided = x0_hat - delta

Step 10: use x0_guided in the EDM reverse update
  x_{t-1} <- EDM_update(x_t, x0_guided, sigma_t)
```

---

## 11. Practical recommendation for Mosaic

For the next implementation pass, the most useful changes are:

1. **Keep BoltzGen IF as the only differentiable decoder bridge**
   - do not rely on AbMPNN as the gradient bridge

2. **Split gradients by objective**
   - no raw sum over all losses before differentiation

3. **Normalize each gradient separately**
   - RMS or norm-based normalization over design atoms

4. **Use binding as the anchor**
   - naturalness and edit terms are regularizers, not co-equal leaders

5. **Add trust-region clipping**
   - especially important at low sigma

6. **Decay guidance harder near the end**
   - late steps should refine, not strongly redirect

7. **Keep ipSAE out of the gradient loop**
   - ipSAE is non-smooth / effectively non-differentiable for this purpose
   - use differentiable `ipTM` + iPAE/interface-PAE inside diffusion
   - use ipSAE after refolding for final rank/filter

This is the smallest redesign that fixes the main conceptual weakness without
changing the whole pipeline.

---

## 12. Implementation scope

This is a real rewrite of the guidance injection block, not a small
hyperparameter patch.

The current code path differentiates a single scalar guidance objective. The
proposed controller requires at least:

- separate objective evaluations
- separate gradients for `g_bind`, `g_nat`, `g_edit`
- masking, de-meaning, normalization, and compatibility projection per
  objective
- merged trust-region clipping before the reverse update

So this should be sized as a real implementation phase inside
`guided_partial_diffusion`, not treated as a light configuration change.

---

## 13. Relationship to prior work

The proposed controller is closest in spirit to:

- **Gradient Guidance for Diffusion Models: An Optimization Perspective**
  - frames guidance as a controlled optimization step on top of a generative
    prior
  - arXiv: https://arxiv.org/abs/2404.14743

- **Universal Guidance for Diffusion Models**
  - inference-time steering with arbitrary differentiable guidance functions
  - ICLR 2024: https://proceedings.iclr.cc/paper_files/paper/2024/hash/dfbb3d1807b21dadee735eb75069ada4-Abstract-Conference.html

- **PCGrad: Gradient Surgery for Multi-Task Learning**
  - useful reference for conflict-aware gradient merging
  - arXiv: https://arxiv.org/abs/2001.06782

- **Protein Design with Guided Discrete Diffusion**
  - protein-design example of guided generative sampling under edit and fitness
    constraints
  - NeurIPS 2023: https://papers.nips.cc/paper_files/paper/2023/hash/29591f355702c3f4436991335784b503-Abstract-Conference.html

These are not identical to our setting, but they support the core design
principle: a guidance term should be treated as a controlled optimization signal
inside a generative prior, not as an unrestricted raw gradient overwrite.
