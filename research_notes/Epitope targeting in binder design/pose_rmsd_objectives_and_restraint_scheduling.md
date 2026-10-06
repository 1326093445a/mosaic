# Reference-pose RMSD as a differentiable objective, and restraint-weight scheduling

Scope note: this covers (a) prior art for backpropagating a pose/placement objective through a frozen structure predictor to optimise sequence, and (b) evidence about whether escalating/scheduling a geometric restraint coefficient achieves constraint satisfaction. Current as of 2026-10.

---

## Q1. Does any published work backprop an RMSD-to-reference-structure objective through AF2/RF/a diffusion predictor to optimise SEQUENCE?

### Takeaway

Yes — but only for **intra-chain** objectives (fixed-backbone and motif recapitulation), where the atoms used for alignment and the atoms being scored belong to the same structural unit. I found **no published work that uses an inter-chain, align-on-target / measure-on-binder reference-pose RMSD** as a differentiable loss through a structure predictor to optimise sequence. Every epitope-/pose-targeting binder method I located specifies placement through **distogram-space contact losses on named residue pairs**, not through coordinate RMSD.

### Cited Findings

**ColabDesign / AfDesign — the canonical implementation, and it does contain the exact machinery in question**

- `_get_rmsd_loss(true, pred, weights=None, L=None, include_L=True, copies=1)` performs weighted Kabsch (SVD) alignment via `_np_kabsch()`, centring on weighted means, and returns `rmsd = sqrt(mean_squared_deviation + 1e-8)` together with the alignment function — [ColabDesign af/loss.py](https://github.com/sokrypton/ColabDesign/blob/main/colabdesign/af/loss.py)
- Critically, that function **"aligns on first L positions, optionally computes error on remaining positions when `include_L=False`"**, and for `copies>1` performs all-vs-all alignment of remaining chains taking the minimum. This is precisely the align-on-subset / score-the-complement pattern the project implements; it exists in ColabDesign and is wired into the **`partial` (partial hallucination) protocol**, not into the binder protocol — [ColabDesign af/loss.py](https://github.com/sokrypton/ColabDesign/blob/main/colabdesign/af/loss.py)
- The `partial` protocol takes `opt["pos"]` (target residue indices), subsamples inputs/outputs with `jax.tree_util.tree_map(lambda y: jnp.take(y, pos, axis), x)`, and computes **RMSD, distogram CE and FAPE only on the selected residues**; optional sidechain terms if `use_sidechains=True` — [ColabDesign af/loss.py](https://github.com/sokrypton/ColabDesign/blob/main/colabdesign/af/loss.py)
- ColabDesign supports multichain design/hallucination for `fixbb`, `hallucination` and `partial` protocols, with partial-hallucination-specific losses including sidechain FAPE — [ColabDesign af/README](https://github.com/sokrypton/ColabDesign/blob/main/af/README.md)
- The ColabDesign **binder** protocol's losses are confidence- and contact-based (pLDDT, PAE/i-PAE, i-con, pTM/ipTM, experimentally-resolved), with interface masking `mask_2d = mask_2d * mask_1d[:,None] * mask_1b[None,:]` and `pae / 31.0` normalisation. **No coordinate RMSD term appears in the binder loss set** — [ColabDesign af/loss.py](https://github.com/sokrypton/ColabDesign/blob/main/colabdesign/af/loss.py)
- A dedicated "AfDesign — Partial Hallucination with sidechain constraints" notebook is archived, confirming RMSD/FAPE-to-reference-motif is an intended, supported mode — [Zenodo 6803187](https://zenodo.org/records/6803187)
- AfDesign needs an extra "contact" term in its loss; **"without this term, most AfDesign runs tended to converge to extended alpha helices"** — [ESMFold hallucination paper](https://www.biorxiv.org/content/10.1101/2023.05.23.541774v1.full)

**ESMFold hallucination (Jeliazkov et al. 2023) — the only paper I found with quantitative convergence data on an RMSD loss through a predictor**

- Implements **"a differentiable Kabsch alignment algorithm"** to make RMSD usable as a loss through ESMFold — [Jeliazkov et al., bioRxiv 2023.05.23.541774](https://www.biorxiv.org/content/10.1101/2023.05.23.541774v1.full)
- **"We did not scale RMSD in the loss function, which caused RMSD loss to contribute 10-fold more than pLDDT."** — i.e. the RMSD term was deliberately left ~10× dominant — [ibid.](https://www.biorxiv.org/content/10.1101/2023.05.23.541774v1.full)
- Despite that 10× dominance: **"It took five times more optimization steps to reach these RMSD values as it did to reach pLDDT values exceeding 90."** For the fibronectin domain (1TEN) this was ~1,500 steps vs ~300 for pLDDT — [ibid.](https://www.biorxiv.org/content/10.1101/2023.05.23.541774v1.full)
- Outcome: **"while convergence was slower than for pLDDT loss alone, designs eventually achieved alpha-carbon RMSD values below 2 Angstroms... low RMSD values were not guaranteed, indicating that further investigation of the overall design strategy was necessary."** Ubiquitin (76 res) reached median RMSD 0.08 Å — [ibid.](https://www.biorxiv.org/content/10.1101/2023.05.23.541774v1.full)
- Summary framing from the same source: **"Including RMSD in the loss function constrained hallucination. However, designing with RMSD loss is more challenging than designing for high pLDDT."** — [ibid.](https://www.biorxiv.org/content/10.1101/2023.05.23.541774v1.full)

**Constrained hallucination / motif scaffolding (Wang et al.) — coordinate-space losses through RoseTTAFold**

- Constrained hallucination optimises sequence "to fold to a structure containing the desired functional site using a composite loss function that combines the hallucination loss with a motif reconstruction loss over the functional motif"; the composite loss has three parts — hallucination loss, motif reconstruction loss, and a problem-specific loss — [Wang et al., bioRxiv 2021.11.10.468128](https://www.biorxiv.org/content/10.1101/2021.11.10.468128v2.full); [Science abn2100](https://www.science.org/doi/full/10.1126/science.abn2100)
- **RoseTTAFold outperformed trRosetta** for site-constrained hallucination, "likely reflecting the better overall modeling of protein sequence-structure relationships." Explicit rationale for coordinates over 6D: "because 3D coordinates are explicitly modeled, site recapitulation can be assessed at the coordinate level and **additional problem-specific loss terms can be implemented in coordinate space that assess interactions with a target**" — [Wang et al.](https://www.biorxiv.org/content/10.1101/2021.11.10.468128v2.full)
- For their binder/epitope-scaffolding case they add **"an additional repulsive term assessed on the complex 3D coordinates in the composite loss function to penalize interactions with the antibody beyond those present in the epitope"** — note this is a *repulsive* coordinate term, not an attractive pose-RMSD term — [ibid.](https://www.biorxiv.org/content/10.1101/2021.11.10.468128v2.full)
- Cost caveat: "the constrained hallucination approach is compute-intensive, as a forward and backward pass through the network is required for each gradient descent step" — [ibid.](https://www.biorxiv.org/content/10.1101/2021.11.10.468128v2.full)

**Germinal (epitope-targeted antibodies/nanobodies, AF-Multimer backprop + IgLM) — no coordinate RMSD at all**

- Germinal combines gradients from AF-Multimer and IgLM by "gradient merging" across three design phases (logits → softmax → semi-greedy) — [Germinal, bioRxiv 2025.09.19.677421](https://www.biorxiv.org/content/10.1101/2025.09.19.677421v1.full); published as [Nat Biotechnol s41587-026-03187-0](https://www.nature.com/articles/s41587-026-03187-0)
- It reuses ColabDesign losses (pLDDT, ipLDDT, PAE, iPAE, pTM, ipTM, inter-protein contacts, intra-protein contacts) plus three custom terms; **no coordinate RMSD-to-reference-pose term is used** — [ibid.](https://www.biorxiv.org/content/10.1101/2025.09.19.677421v1.full)
- Epitope specification is via a **paratope loss of the form `L_paratope ∝ L_CDR − λ · L_framework`**, where `L_CDR` is a binary cross-entropy loss between CDR residues and target epitope hotspots and `L_framework` is BCE between framework residues and all target residues; λ is a user-defined weight ("offset" in the code). Purpose: "binding occurs primarily through CDRs rather than framework residues" — [ibid.](https://www.biorxiv.org/content/10.1101/2025.09.19.677421v1.full)
- Secondary-structure terms are also distogram-BCE: α-helix loss penalises `{i, i+3}` pairs with predicted distances in **2–6.2 Å**; β-strand loss uses **9.75–11.5 Å**, averaging the top three most likely pairs — [ibid.](https://www.biorxiv.org/content/10.1101/2025.09.19.677421v1.full)

**TorchCraft (2026) — inverting an all-atom AF3-class predictor; still no reference-pose RMSD**

- Inverts **AlphaFold3** (TorchFold implementation), predictor frozen; binder sequence is a trainable logit matrix `L`, converted to `X` by softmax/straight-through, gradients taken w.r.t. `L` only — [TorchCraft, arXiv 2609.19770](https://arxiv.org/html/2609.19770)
- All geometric control is distogram- or eigenvalue-based, with **no RMSD-to-reference term**:
  - radius-of-gyration proxy: `Rg² = (1/2N²) Σᵢ Σⱼ E[dᵢⱼ²]`, `ℒ_rog = ReLU(Rg − Rg,target)`, `Rg,target = αN^β + γ`
  - isotropy: `ℒ_isotro = [(λ₁−λ₂)² + (λ₂−λ₃)² + (λ₁−λ₃)²] / [(λ₁+λ₂+λ₃)² + ε]` from centred Gram-matrix eigenvalues
  - α-helix: `ℒ_α = −(1/|H|) Σ₍ᵢ,ⱼ₎∈H log(pᵢⱼ^α + ε)`, `H = {(i,j) | |i−j| = 3}`
  - β: strand distance interval [9.75, 11.5) Å, sheet [4.4, 6.0) Å, with top-K strand pair selection
  - paratope (VHH): `ℒ_paratope = (ℒ_CDR-Hotspot · ℒ_CDR-Interface) / [ReLU(ℒ_framework − λ) + ε]`
  — [TorchCraft, arXiv 2609.19770](https://arxiv.org/html/2609.19770)
- Note the **`ReLU(Rg − Rg,target)` hinge**: the same functional form as the project's `relu(rmsd − tolerance)`, but applied to a distogram-derived expectation rather than to a Kabsch-aligned coordinate scalar — [ibid.](https://arxiv.org/html/2609.19770)

**Fold-conditioned AF2 binder design (2025) — a project that explicitly wanted a pose/fold condition and chose contact maps over RMSD**

- Fold conditioning is imposed by a **contact-map (cmap) similarity loss**, not coordinate RMSD or FAPE: a template cmap `Z_cond` is built from a reference structure; AF2-Multimer's predicted distogram is converted to `Z_pred` using **14 Å intra-chain and 21.7 Å inter-chain** thresholds; the loss is the **RMSE over a masked subset of residue–residue contact pairs** between `Z_pred` and `Z_cond` under a binary mask `M` — [Fold-Conditioned De Novo Binder Design via AlphaFold2, bioRxiv 2025.07.02.662497](https://www.biorxiv.org/content/10.1101/2025.07.02.662497v1.full)
- Three-stage optimisation: 100 steps logits, 100 steps softmax, 20 steps one-hot. **RMSD is used only for post-hoc evaluation, never as the optimisation objective** — [ibid.](https://www.biorxiv.org/content/10.1101/2025.07.02.662497v1.full)
- Effectiveness: **8–45% of designs met RMSD < 3.5 Å** across six target folds; cmap loss correlated with AF2 confidence metrics (pLDDT, PAE, intra-chain contacts) — [ibid.](https://www.biorxiv.org/content/10.1101/2025.07.02.662497v1.full)
- Failure modes: recurring **"side-docking interactions"** in VHH designs (attributed to training-data bias), and the method "did not generalize to scFv antibody topologies," attributed to "lower accuracy of AF2-Multimer in docking complex antibody folds" — [ibid.](https://www.biorxiv.org/content/10.1101/2025.07.02.662497v1.full)

**EasyNano (2026) — epitope-targeted nanobody CDR design through ESMFold2 distograms**

- Described as "rapid epitope-targeted nanobody CDR design via differentiable distogram optimization with ESMFold2," using "a dedicated CDR-to-epitope distance loss, which directs optimization toward specific target residues rather than any interface," with "structural signals backpropagated through the ESMFold2 distogram" — [EasyNano, arXiv 2606.12772](https://arxiv.org/pdf/2606.12772)

**Related: differentiable pose optimisation where pose variables (not sequence) are the free parameters**

- DeepRMSD+Vina builds "a fully differentiable docking pose optimization framework, in which sub-molecular translational and rotational movements are guided by back-propagation gradients of the scoring function." Notably, instead of differentiating a Kabsch RMSD they train an **MLP (DeepRMSD) to *predict* the RMSD of a pose w.r.t. the native pose** from vdW/electrostatic interaction features; optimisation success rate in the 0–3 Å band is **69%** — [arXiv 2206.13345](https://arxiv.org/abs/2206.13345); [Brief Bioinform 24(1) bbac520](https://academic.oup.com/bib/article/24/1/bbac520/6887112)

### Inferences

- The field's revealed preference is unambiguous: **when a specific pose or epitope must be hit, published pipelines write the restraint in distogram space over named residue pairs** (Germinal's paratope BCE, TorchCraft's paratope/hotspot terms, the fold-conditioned cmap RMSE, EasyNano's CDR-to-epitope distance loss), and they reserve RMSD for post-hoc evaluation. The project's coordinate pose-RMSD hinge is, as far as I can find, without published precedent in the inter-chain setting.
- The mechanistic reason is plausibly gradient density, not correctness. A distogram loss over an |epitope| × |CDR| pair set supplies a gradient to every one of those pairs directly from a network *output head*. A global pose RMSD supplies **one scalar per step**, and the Kabsch step deliberately removes exactly the rigid-body component that constitutes most of a 22–25 Å placement error, leaving a residual that is then averaged over all binder CA atoms (≈1/N dilution).
- The ESMFold result is the closest thing to a controlled measurement, and it cuts against the "just raise the weight" hypothesis: at ~10× dominance over the confidence term the RMSD objective still needed ~5× more steps, i.e. **coefficient magnitude bought little or no optimisation speed**. It is, however, an *intra-chain* fixbb-style objective that eventually did converge, so it does not establish that an inter-chain pose RMSD can converge at all.
- Wang et al.'s choice is informative in the opposite direction: having coordinates available, they used them for a **repulsive** term (keep contacts off non-epitope surface) rather than an attractive pose-RMSD term. A repulsive/exclusion formulation may be better matched to what coordinate gradients can deliver than an attractive global-alignment one.

### Gaps

- I could not retrieve the **exact equation** for Wang et al.'s motif reconstruction loss (is it motif CA RMSD, a 6D/distogram KL, or FAPE?). The main-text fetch returned only qualitative language; the formula lives in the Methods/SI, which was not accessible. The trRosetta-era version is described as 6D and the RoseTTAFold version as coordinate-level, but I cannot quote the RF formula.
- EasyNano's exact loss formulas, weights and quantitative epitope-hit rates could not be extracted (the arXiv PDF returned compressed/unparseable text). Treat the description above as title/abstract-level only.
- BindEnergyCraft's loss equations (arXiv 2505.21241) were not extractable from the PDF; its stated critique of ipTM/PAE-driven design could not be quoted.
- No head-to-head published comparison of **FAPE vs RMSD vs distogram** as a *design* objective was found.

---

## Q2. Known problems with RMSD as a differentiable objective; is FAPE used as a design objective and is it preferable?

### Takeaway

The Kabsch/SVD gradient pathology is real but confined to near-degenerate alignments (coincident or vanishing singular values), and the original authors report **no practical instability**; it is therefore an unlikely explanation for a *weak* (rather than explosive) pose-RMSD gradient. FAPE **is** implemented as a design objective (ColabDesign `get_fape_loss`, including sidechain FAPE) and avoids global alignment by construction — but ColabDesign's default **clamp of 10 Å** means it is fully saturated, hence zero-gradient, at a 22–25 Å placement error.

### Cited Findings

**Kabsch/Procrustes SVD gradients (Levinson et al., NeurIPS 2020)**

- Exact gradient of the SVD orthogonalisation layer: `∂L/∂M = U[(F∘(Uᵀ∂L/∂U − ∂L/∂Uᵀ U))Σ + Σ(F∘(Vᵀ∂L/∂V − ∂L/∂Vᵀ V))]Vᵀ` with `F_{i,j} = 1/(s_i² − s_j²)` for `i ≠ j` and `F_{i,j} = 0` for `i = j`; this simplifies to `Z_{ij} = −X_{ij}/(s_i + s_j)` for `i ≠ j`, 0 on the diagonal — [Levinson et al., NeurIPS 2020](https://papers.nips.cc/paper_files/paper/2020/file/fec3392b0dc073244d38eba1feb8e6b7-Paper.pdf); [arXiv 2006.14616](https://ar5iv.labs.arxiv.org/html/2006.14616)
- The 3×3 Jacobian has rank 3 with nonzero singular values **`2/(s_i + s_j)`** and condition number **`κ = (s₁ + s₂)/(s₂ + s₃)`** — [ibid.](https://ar5iv.labs.arxiv.org/html/2006.14616)
- Degeneracy statements, verbatim: **"∂L/∂M is undefined whenever two singular values are both zero and large when their sum is very near zero."** For the special-orthogonal projection `SVDO⁺`: **"∂L/∂M is undefined if the smallest singular value occurs with multiplicity greater than 1. It is large if the two smallest singular values are close."** This arises when `det(M) < 0` and the smallest singular value has multiplicity > 1, which makes the underlying optimisation problem itself degenerate — [ibid.](https://ar5iv.labs.arxiv.org/html/2006.14616)
- Practical verdict from the same authors: **"SVD-Train converges quickly (relative to all other methods) in all of our experiments, indicating no instabilities due to large gradients,"** with gradient profiles matching the 6D Gram-Schmidt representation — [ibid.](https://ar5iv.labs.arxiv.org/html/2006.14616)
- SVD orthogonalisation is "optimal as a projection in the least squares sense and in the presence of Gaussian noise (MLE)," making it the natural choice for projecting onto the rotation group — [ibid.](https://ar5iv.labs.arxiv.org/html/2006.14616)
- A later gradient analysis argues the SVD Jacobian introduces an **O(1/δ) gradient amplification that is "unnecessary"** (where δ is a singular-value gap), proposing training without orthogonalisation and applying SVD only at inference — [Training Without Orthogonalization, Inference With SVD, arXiv 2604.05414](https://arxiv.org/html/2604.05414v1). *Caveat: I did not fetch this paper in full; the claim comes from search-result text and the abstract-level HTML, and I could not independently confirm its publication venue or date.*
- A related line proposes replacing the analytic SVD backward pass altogether with a manifold-aware gradient layer for rotation regression — [Projective Manifold Gradient Layer for Deep Rotation Regression, arXiv 2110.11657](https://arxiv.org/pdf/2110.11657)

**FAPE as an implemented design objective**

- ColabDesign `get_fape_loss(inputs, outputs, copies=1, clamp=10.0, return_mtx=False)`: builds local frames from N, CA, C by Gram–Schmidt, computes inter-residue vectors in frame coordinates, and takes **`fape = clip(norm(true_vector − pred_vector), 0, clamp) / 10.0`**, i.e. default clamp 10 Å with the result normalised to [0,1]. Requires valid N, CA, C in both true and predicted structures — [ColabDesign af/loss.py](https://github.com/sokrypton/ColabDesign/blob/main/colabdesign/af/loss.py)
- Sidechain FAPE exists via `folding.sidechain_loss(batch, sc_struct, config)` using the same local-frame formalism, but carries an explicit limitation: **"ERROR: 'sc_fape' not currently supported for 'multimer' mode"** — i.e. unavailable for complex/binder work — [ColabDesign af/loss.py](https://github.com/sokrypton/ColabDesign/blob/main/colabdesign/af/loss.py)
- Sidechain RMSD (`_get_sc_rmsd_loss`) aligns non-ambiguous atoms by Kabsch then takes the rotamer-ambiguity minimum, `sd = min(||P−T||², ||P−T_alt||²)`, `rmsd = sqrt(mean(sd) + 1e-8)` — [ibid.](https://github.com/sokrypton/ColabDesign/blob/main/colabdesign/af/loss.py)
- In ColabDesign both FAPE and RMSD are applied to **position-subsetted** inputs in the `partial` protocol (via `opt["pos"]`), i.e. they are motif-recapitulation objectives — [ibid.](https://github.com/sokrypton/ColabDesign/blob/main/colabdesign/af/loss.py)

### Inferences

- Levinson's degeneracy conditions require coincident or near-zero singular values of the cross-covariance matrix. For a Kabsch fit over the CA atoms of a **folded, globular, three-dimensionally extended target domain**, the cross-covariance is well-conditioned and `κ = (s₁+s₂)/(s₂+s₃)` is modest. The predicted failure mode there is *gradient blow-up*, not *gradient vanishing* — so the SVD pathology is a poor explanation for a term that "does almost nothing." It would become relevant if the alignment set were small, collinear/planar (e.g. a short helical epitope), or if the predicted target were near-mirror-image (`det < 0`).
- **FAPE is not a drop-in fix at this error scale.** With ColabDesign's `clamp=10.0`, every per-frame error term in a 22–25 Å mis-placement is clipped to 10 and divided by 10, giving a constant 1.0 with **identically zero gradient**. Using FAPE for a gross placement objective would require raising or removing the clamp (and in AF2's own loss the clamp exists precisely to stop distant errors from dominating). Also relevant: sidechain FAPE is unavailable in multimer mode, so the all-atom variant is not an option for complexes in that codebase.
- FAPE's structural advantage over RMSD for a *pose* objective is real in principle — it scores each residue in every other residue's local frame, so it never needs a single global superposition and it does not let a rigid-body fit absorb the error. The cost is that it changes the objective from "the binder sits here" to "every pairwise frame relationship matches," which is a much stiffer, higher-dimensional requirement.
- A hinge on a global scalar, `relu(rmsd − tol)`, has a further structural weakness independent of Kabsch: its gradient magnitude w.r.t. the scalar is exactly 1 whenever active and 0 otherwise, so it carries **no information about how far off** the pose is. Scaling the weight scales a direction that is already nearly orthogonal to what the predictor's inter-chain machinery responds to.

### Gaps

- I found no paper that explicitly attributes a protein-design optimisation failure to Kabsch/SVD gradient pathology. The connection between Levinson et al. and protein design losses is one I am drawing, not one reported in the literature.
- No published ablation comparing FAPE vs coordinate RMSD as the geometric term in an AF-backprop design loop. ColabDesign exposes both but I found no study that benchmarks them against each other.

---

## Q3. Distogram / contact-map restraints as an alternative way to specify a pose

### Takeaway

This is the dominant published approach, and the functional forms cluster into three families: **binned categorical cross-entropy** against a reference distogram, **binary "contact probability mass below a cutoff"** losses, and **masked RMSE on thresholded contact maps**. Reported effectiveness for pose/fold control is modest but real (8–45% of designs under 3.5 Å RMSD in the one study that quantified it), and systematic side-docking failures persist.

### Cited Findings

**Exact functional forms (ColabDesign, the reference implementation)**

- `get_dgram_loss`: true pairwise distances from pseudo-β atoms are binned with edges **`linspace(2.3125, 21.6875, 63)`** (64-bin AF2 distogram) or **`linspace(3.25, 50.75, 39)`** (39-bin); loss is categorical cross-entropy **`−(true_onehot * log_softmax(pred_logits)).sum(-1)`** — [ColabDesign af/loss.py](https://github.com/sokrypton/ColabDesign/blob/main/colabdesign/af/loss.py)
- `_get_con_loss(dgram, dgram_bins, cutoff=None, binary=True)`: masks bins below the cutoff distance; **binary form `−log((contact_bins * softmax).sum() + 1e-8)`**; categorical form `−(masked_softmax * log_softmax(raw)).sum(-1)`. The `get_con_loss()` wrapper adds sequence-separation filtering and **top-k selection** — [ibid.](https://github.com/sokrypton/ColabDesign/blob/main/colabdesign/af/loss.py)
- Interface contact loss for binders applies `get_con_loss` with `opt["i_con"]` and `mask_1d=binder_id, mask_1b=target_id`, i.e. the contact distogram loss restricted to interface residue pairs — [ibid.](https://github.com/sokrypton/ColabDesign/blob/main/colabdesign/af/loss.py)
- Interface PAE loss: `mask_2d = mask_2d * mask_1d[:,None] * mask_1b[None,:]`, error `pae / 31.0`, symmetrised — [ibid.](https://github.com/sokrypton/ColabDesign/blob/main/colabdesign/af/loss.py)

**Flat-bottom / hinge forms on distogram-derived quantities**

- TorchCraft's `ℒ_rog = ReLU(Rg − Rg,target)` with `Rg² = (1/2N²) Σᵢ Σⱼ E[dᵢⱼ²]` is a **one-sided flat-bottom (hinge) restraint computed on a distogram expectation** rather than on coordinates — [TorchCraft, arXiv 2609.19770](https://arxiv.org/html/2609.19770)
- TorchCraft's contact loss: "soft constraints using predicted distance distributions. Probability mass below a distance threshold is treated as contact bins; cross-entropy computed over renormalized contact probabilities" — [ibid.](https://arxiv.org/html/2609.19770)
- TorchCraft's paratope loss uses a hinged denominator to suppress framework contact: `ℒ_paratope = (ℒ_CDR-Hotspot · ℒ_CDR-Interface) / [ReLU(ℒ_framework − λ) + ε]` — a multiplicative/ratio combination rather than a weighted sum — [ibid.](https://arxiv.org/html/2609.19770)

**Binned/interval BCE forms for pose and secondary structure**

- Germinal: paratope `L_CDR` = BCE between CDR residues and target epitope hotspots; `L_framework` = BCE between framework residues and all target residues; combined as `L_paratope ∝ L_CDR − λ·L_framework` — [Germinal, bioRxiv 2025.09.19.677421](https://www.biorxiv.org/content/10.1101/2025.09.19.677421v1.full)
- Germinal's secondary-structure restraints are explicit **distance-interval** BCE terms on the AF distogram: α-helix on `{i, i+3}` pairs in 2–6.2 Å; β-strand on 9.75–11.5 Å averaged over the top three most likely pairs — [ibid.](https://www.biorxiv.org/content/10.1101/2025.09.19.677421v1.full)

**Masked contact-map RMSE for fold/pose conditioning, with quantified effectiveness**

- Fold-conditioned AF2 design: cmap thresholds **14 Å intra-chain, 21.7 Å inter-chain**; loss = masked RMSE between predicted and template cmaps. Result: **8–45% of designs met RMSD < 3.5 Å** across six folds; cmap loss correlated with pLDDT/PAE/intra-chain contacts. Persistent failure: VHH **side-docking**; no generalisation to scFv — [bioRxiv 2025.07.02.662497](https://www.biorxiv.org/content/10.1101/2025.07.02.662497v1.full)

**Hotspot-based epitope specification is standard practice**

- "De novo protein binders can be designed by backpropagating through AlphaFold2 multimer toward a chosen target surface, with users specifying hotspot residues," treating AF2-Multimer "as a differentiable function to backpropagate a sequence whose predicted folded complex binds the target at exactly those hotspot residues with high confidence" — [BindCraft](https://www.nature.com/articles/s41586-025-09429-6); see also [Ranomics/BindCraft overview](https://www.ranomics.com/technology/bindcraft)
- EasyNano uses "a dedicated CDR-to-epitope distance loss, which directs optimization toward specific target residues rather than any interface" — [arXiv 2606.12772](https://arxiv.org/pdf/2606.12772)

### Inferences

- There is an important representational difference between a distogram restraint and a pose RMSD: a distogram/contact restraint on epitope×CDR pairs **specifies the pose up to the residual freedom of the contact set**, and does so in the predictor's own output space where gradients are dense and well-scaled (logits). A coordinate RMSD specifies the pose exactly but reaches the sequence only through the structure module, after a global superposition that cancels the dominant error mode.
- The functional forms in use are overwhelmingly **cross-entropy / BCE on probability mass**, not quadratic or hinge penalties on distances. When a hinge *is* used (TorchCraft's ROG), it is applied to a distogram expectation, which keeps the gradient flowing through all pairs.
- Practically, the fold-conditioned paper is the closest analogue to the project's goal (impose a reference placement through AF2) and its numbers suggest the achievable ceiling for this class of restraint is "a useful minority of designs," not reliable satisfaction — and that some pose errors (side-docking) are driven by predictor bias that no loss weight fixes.
- TorchCraft's ratio/multiplicative composition and Germinal's difference form are both ways of avoiding a plain weighted sum, in which a hard-to-move geometric term can be trivially traded away against easy-to-move confidence terms. This is directly relevant to the project's observation that the RMSD term has no effect in a summed loss.

### Gaps

- No source I found reports a **direct head-to-head** comparison of an RMSD pose restraint against a distogram pose restraint within the same pipeline, so the claimed superiority of distograms is inferred from universal adoption plus one quantified distogram result, not from a controlled experiment.
- Default numerical weights for ColabDesign's loss terms were not recovered from `loss.py` ("No explicit preference stated in code"); weights live in the protocol/config files, which I did not fetch.

---

## Q4. Restraint-weight scheduling: does escalating a coefficient achieve constraint satisfaction?

### Takeaway

In **classical optimisation** (sequential convex programming / exact-penalty methods) escalation provably and practically works, but it depends on two conditions that an AF-backprop design loop does not satisfy: each inner problem is solved to convergence before the coefficient is raised, and an exact-penalty theorem guarantees a finite threshold μ* exists. In **learned-model settings** the evidence points the other way: Rosetta's FastRelax ramps restraints **down**, and the one paper that measured guidance-weight escalation through a generative structure model reports that it **disrupts the trajectory** and leaves constraint satisfaction at 0%. Among AF-design pipelines, what is actually scheduled is the *sequence parameterisation* (logits→softmax→straight-through→one-hot) and temperature, not geometric restraint weights.

### Cited Findings

**Classical penalty continuation (TrajOpt, Schulman et al. RSS 2013) — escalation works, under conditions**

- "TrajOpt uses a sequential convex optimization procedure which penalizes collisions with a **hinge loss** and **increases the penalty coefficients in an outer loop as necessary**" — [TrajOpt, RSS 2013](https://www.roboticsproceedings.org/rss09/p31.pdf); [IJRR version](https://dl.acm.org/doi/10.1177/0278364914528132)
- Penalty forms: inequality `g_i(x) ≤ 0` → `|g_i(x)|+ = max(0, g_i(x))`; equality `h_i(x) = 0` → `|h_i(x)|`. Combined objective: **`min f̃(x) + μ Σ|g̃_i(x)|+ + μ Σ|h̃_i(x)|`** — [Understanding TrajOpt (secondary exposition)](https://shrenikm.com/posts/2024-01-06-understanding-trajopt/)
- Outer loop: after SCP convergence, check constraint satisfaction against a tolerance `ctol`; if satisfied, return; **if not, set `μ ← k·μ`** and resume SCP — [ibid.](https://shrenikm.com/posts/2024-01-06-understanding-trajopt/)
- The two stated ingredients of the algorithm are "(1) a method for constraining the step to be small, so the solution vector remains within the region where the approximations are valid; (2) a strategy for turning the infeasible constraints into penalties, but **eventually ensuring that all of the constraint violation are driven to zero**" — [TrajOpt, RSS 2013](https://www.roboticsproceedings.org/rss09/p31.pdf)
- Theoretical basis: "there exists a finite μ* above which... the penalty minimizer satisfies the constraints" (the exact-penalty property of the l1 penalty) — [Understanding TrajOpt](https://shrenikm.com/posts/2024-01-06-understanding-trajopt/)
- Per-constraint rather than global coefficients help: "rather than using a single penalty parameter, maintaining a separate penalty parameter for each constraint — this individualized penalization strategy often leads to superior convergence in practice" — [continuous-time successive convexification literature](https://arxiv.org/pdf/2502.01055)
- In successive-convexification variants, "when hard-penalized state constraints are not satisfied, the penalty weight is increased, pushing the solver to seek constraint satisfaction at the next iteration" — [arXiv 2502.01055](https://arxiv.org/pdf/2502.01055); see also [GuSTO](https://stanfordasl.github.io/wp-content/papercite-data/pdf/Bonalli.Cauligi.Bylard.Pavone.ICRA19.pdf)

**Rosetta — restraints are ramped DOWN, not up**

- The `-relax:ramp_constraints` flag: "When explicitly set to false, do not ramp down constraints (does not affect ramping in custom scripts). **When true, constraints are ramped down during each simulated annealing cycle.**" Ramping down is the default behaviour — [Rosetta Relax documentation](https://docs.rosettacommons.org/docs/latest/application_documentation/structure_prediction/relax)
- The default FastRelax script's coordinate-constraint schedule across the four repack/minimise steps of each repeat is **`coord_cst_weight 1.0 → 0.5 → 0.0 → 0.0`**, repeated over five cycles — [ibid.](https://docs.rosettacommons.org/docs/latest/application_documentation/structure_prediction/relax)
- What *is* ramped up is the repulsive term: relax "iterates four cycles of Monte Carlo rotamer optimization with all-atom minimization, **ramping the weight on van der Waals repulsion in each cycle**" — [ibid.](https://docs.rosettacommons.org/docs/latest/application_documentation/structure_prediction/relax)
- General guidance on constraint weight magnitude: "the more highly constraint weights are applied, the more Rosetta relies on constraints when building structures, but you need to find a **balance to avoid artifacts from potentially inaccurate constraints**" — [RosettaCommons forum](https://forum.rosettacommons.org/node/3747)

**AF-backprop design pipelines — what they actually schedule**

- BindCraft's four stages schedule the **sequence parameterisation**, not loss weights: Stage 1 logit-space with representation **`(1 − λ)·logits + λ·softmax(logits/T)`**; Stage 2 softmax with **temperature gradually lowered over 45 iterations**; Stage 3 straight-through estimator for **five iterations** (one-hot forward, softmax backward); Stage 4 one-hot discrete with random single mutations accepted on best loss — [BindCraft, Nature 2025](https://www.nature.com/articles/s41586-025-09429-6)
- TorchCraft's four stages are likewise parameterisation schedules: Stage 0 `X = Softmax(L)`; Stage 1 `X = (1−a)L + a·Softmax(L)` with `a` increasing linearly; Stage 2 `X = Softmax(L/τ)` with τ decreasing from τ₀ to τ_min; Stage 3 straight-through hard. Reported explicitly: **"No universal annealing schedule for individual loss weights is explicitly reported; weights appear task-configured statically."** — [TorchCraft, arXiv 2609.19770](https://arxiv.org/html/2609.19770)
- **The one escalating geometric/interface coefficient I found:** for small-molecule design TorchCraft uses **`ℒ(t) = ℒ_scaffold + w_interface(t) · ℒ_interface`** with "interface weight increases over time" — [ibid.](https://arxiv.org/html/2609.19770)
- Germinal anneals a **prior** weight, not a geometric one, and reports that it helped: IgLM weight is "linearly annealed from v₁ to v₂" during the logits phase, set to v₃ in softmax and v₄ in semi-greedy, and **"This annealing schedule empirically provided more robust performance than fixed IgLM weights."** A convergence gate requires average positional certainty above 0.1 before leaving the softmax phase — [Germinal, bioRxiv 2025.09.19.677421](https://www.biorxiv.org/content/10.1101/2025.09.19.677421v1.full)
- Fold-conditioned AF2 design reports stage lengths (100 logits / 100 softmax / 20 one-hot) but **"no explicit loss weights or scheduling information"** for the competing objectives — [bioRxiv 2025.07.02.662497](https://www.biorxiv.org/content/10.1101/2025.07.02.662497v1.full)

**Augmented Lagrangian / dual methods as the alternative to escalation**

- "Constrained Diffusion for Protein Design with Hard Structural Constraints" solves the feasibility problem with **consensus ADMM**: proximal subproblem `x̃₀ᵗ = argmin_x [1/(2ηₜ)||x − x̂₀ᵗ||² + g(x)]`; decomposition `min_{y,z} F(y) + G(z) s.t. y = z`; updates `yᵏ⁺¹ := prox_{ρᵏ,F}(yᵏ − uᵏ)`, `zᵏ⁺¹ := prox_{ρᵏ,G}(zᵏ + uᵏ)`, **dual `uᵏ⁺¹ := uᵏ + yᵏ⁺¹ − zᵏ⁺¹`** — [arXiv 2510.14989](https://arxiv.org/html/2510.14989)
- General theory: augmented Lagrangian methods "can be designed for fully nonconvex settings and natively handle nonsmooth cost functions, nonlinear constraints and nonconvex sets" — [Constrained composite optimization and augmented Lagrangian methods, Math. Program.](https://link.springer.com/article/10.1007/s10107-022-01922-4)

### Inferences

- The conditions under which penalty escalation is *guaranteed* to work are specific: a smooth-enough subproblem solved to convergence at each outer iteration, a trust region keeping steps inside the valid approximation region, and the l1 exact-penalty property. An AF-backprop design loop has **none** of these — the "inner solve" is a fixed number of Adam steps on a non-convex, discontinuity-prone predictor with a discrete end-target, and no exact-penalty theorem applies. So the classical literature's success with escalation should **not** be read as support for simply raising the pose-RMSD weight here.
- The protein-structure-refinement community's practice is the opposite of escalation: restraints are a **funnel used early and released later** (coord_cst 1.0 → 0.5 → 0.0), explicitly because strong restraints against a possibly-inaccurate reference create artifacts. If the project's reference pose is itself uncertain, the Rosetta precedent argues for decreasing, not increasing, the restraint.
- The design-pipeline literature converges on scheduling the **parameterisation** (continuous→discrete, temperature) rather than the loss weights. That is a meaningful signal: practitioners who have tuned these loops extensively did not find loss-weight schedules to be the lever, with the exception of a prior-term anneal (Germinal's IgLM) that was reported as helpful, and one interface-weight ramp (TorchCraft) reported without ablation.
- If hard pose satisfaction is genuinely required, the better-supported mechanism is a **dual variable** (augmented Lagrangian: `w ← w + η·violation`, updated on the *observed* violation rather than on a fixed schedule) or a **constrained/projection step**, not a larger fixed coefficient. This has the useful property of automatically escalating only while the constraint is violated and relaxing once it is met.

### Gaps

- I could not confirm from the **primary** TrajOpt paper the exact value of the escalation factor `k`, the initial μ, or `ctol`; the secondary exposition states the update rule `μ ← k·μ` but explicitly "does not specify explicit values for the scaling factor k or initial μ." The PDF fetch of the RSS paper returned unparseable binary.
- I found no AF-design paper that reports an **ablation of a geometric-loss-weight schedule** (e.g. "we tried ramping w_rmsd and it did/didn't help"). TorchCraft's `w_interface(t)` is stated but unablated.

---

## Q5. Any report that increasing a geometric-restraint coefficient FAILED in a structure-predictor-gradient setting, with failure analysis?

### Takeaway

I found **no paper reporting exactly this negative result** for sequence-gradient design — the project's finding appears to be unreported in the literature. The two nearest pieces of evidence are (i) the ESMFold hallucination paper, where a ~10× dominant RMSD term still took 5× more steps than the confidence term, and (ii) the constrained-diffusion paper, which states outright that increasing guidance weight **disrupts the trajectory** and reports 0% constraint satisfaction for guidance/penalty baselines across tens of thousands of samples.

### Cited Findings

**Closest sequence-gradient evidence: a 10× dominant coefficient did not buy speed**

- "We did not scale RMSD in the loss function, which caused RMSD loss to **contribute 10-fold more than pLDDT**" — and yet "it took **five times more optimization steps** to reach these RMSD values as it did to reach pLDDT values exceeding 90" — [Jeliazkov et al., bioRxiv 2023.05.23.541774](https://www.biorxiv.org/content/10.1101/2023.05.23.541774v1.full)
- Same paper: "low RMSD values were **not guaranteed**, indicating that further investigation of the overall design strategy was necessary" — [ibid.](https://www.biorxiv.org/content/10.1101/2023.05.23.541774v1.full)

**Explicit statements that weight escalation fails through a learned structure model**

- **"Increased weight on guidance terms tends to disrupt the diffusion trajectory"** (citing Dhariwal & Nichol 2021) — [Constrained Diffusion for Protein Design with Hard Structural Constraints, arXiv 2510.14989](https://arxiv.org/html/2510.14989)
- **"Soft constraints into discrete sequence prediction... fail to consistently provide constraint adherent outputs"** — [ibid.](https://arxiv.org/html/2510.14989)
- Quantified baseline failures on a PDZ-domain hydrogen-bond-geometry constraint (bond distances 2.9 ± 0.2 Å; C=O···N angles 155° ± 10°; Cα-N···O angles 120° ± 10°): standard RFdiffusion **0%** constraint satisfaction; "Recenter Mass Guidance" **0% across 31,000 samples**; SMC-based constraint-guided diffusion **0% for the angle requirements**. Baselines achieved **zero usable samples across ~100k total samples**; the ADMM method achieved **100% constraint satisfaction** (21.0% usable after realism filtering), and 100%/97.8% on a molecule-encapsulation task — [ibid.](https://arxiv.org/html/2510.14989)
- Mechanistic explanation offered: "Projecting highly noisy states onto complex, nonconvex constraints... results in solutions **trapped in local minima**," and per-step projection "introduces statistical biases... intermediate samples concentrate near constraint boundaries" — [ibid.](https://arxiv.org/html/2510.14989)
- Their Theorem 6.2 formalises *when* escalation is safe: the schedule **`λₜ = cₜ/ηₜ`** with increasing `cₜ` ensures "feasibility becomes dominant **only when the model's prediction is accurate**"; a uniform increase causes **early-step over-constraint** that disrupts the trajectory. Their remedy is to apply proximal steps to the **predicted clean state** rather than to noisy intermediates — [ibid.](https://arxiv.org/html/2510.14989)

**Related failure analyses where an added loss term was necessary but insufficient**

- Germinal: "naive application of AF-M guidance resulted in generations with **paratopes composed of framework residues** or enriched with secondary-structures," which is what motivated the paratope and secondary-structure losses — i.e. the geometric control had to be *added as a new term in distogram space*, not obtained by reweighting existing terms — [Germinal, bioRxiv 2025.09.19.677421](https://www.biorxiv.org/content/10.1101/2025.09.19.677421v1.full)
- Germinal still requires heavy post-hoc filtering because the losses alone are insufficient: AF3 re-prediction with strict thresholds, PyRosetta biophysical scores, and Borda-count ranking to cut ~1,000+ candidates to ~43–101 for experimental screening — [ibid.](https://www.biorxiv.org/content/10.1101/2025.09.19.677421v1.full)
- Germinal reports a **Pareto trade-off** between competing objectives: "the preference an antibody language model has for a given sequence (IgLM log-likelihood) and the predicted binding confidence (AF-M interface predicted aligned error; iPAE) are competing objectives, exhibiting a trade-off that yields a Pareto frontier" — [ibid.](https://www.biorxiv.org/content/10.1101/2025.09.19.677421v1.full)
- Fold-conditioned AF2 design: persistent **side-docking** in VHH designs attributed to training-data bias, and complete failure to generalise to scFv attributed to "lower accuracy of AF2-Multimer in docking complex antibody folds" — a pose failure traced to the predictor, not the loss — [bioRxiv 2025.07.02.662497](https://www.biorxiv.org/content/10.1101/2025.07.02.662497v1.full)
- TorchCraft reports "minimal negative results" and provides "no systematic ablation showing which losses fail or require much higher weights" — [arXiv 2609.19770](https://arxiv.org/html/2609.19770)

### Inferences

- Taking the ESMFold datapoint seriously: a 10× weight advantage translating into a 5× *disadvantage* in steps-to-target is consistent with the geometric term's gradient being **poorly conditioned or diluted rather than merely small**. Multiplying a badly-directed gradient by a constant does not change its direction, and under Adam (which normalises by a running second-moment estimate) **a uniform rescaling of a single loss term is largely absorbed** — Adam's update is approximately scale-invariant per-parameter, so if the pose-RMSD term's contribution is small relative to the other terms' contribution in the *same* parameter directions, increasing its coefficient changes the mixture only until the second-moment estimate re-adapts. This is a strong candidate explanation for the project's observed coefficient-independence, and it predicts that *larger weights will keep not helping*.
- The constrained-diffusion Theorem 6.2 logic transfers in spirit: a geometric restraint is only informative once the model's prediction is good enough for the restraint's gradient to point somewhere useful. Early in a design trajectory (random-ish sequence, nonsense complex) a pose-RMSD gradient is close to noise. That argues for a **late-activated** restraint (off during logits/soft, on during softmax/hard) rather than a uniformly larger one — and specifically against ramping it up from step 0.
- Germinal's history is the most actionable precedent: the fix for "the model puts the binder in the wrong place" was **not** reweighting, it was **adding a new, differently-shaped term in the predictor's output space** (hotspot BCE) plus a **repulsive** term for the surface that must *not* be contacted. That matches Wang et al.'s repulsive coordinate term as well.

### Gaps

- **No published report of the project's exact negative result** (raising a coordinate pose-RMSD coefficient through a frozen AF3-class predictor without improving pose). This appears genuinely unreported, which makes the project's weight-0 ablation (22.9 vs 25.3 Å at matched compute) a novel finding rather than a replication.
- No source analysed the specific candidate mechanisms the assignment lists (gradient clipping, optimiser normalisation, vanishing coordinate gradients) for a pose-RMSD term. My Adam-scale-invariance argument above is an inference, not a cited finding.

---

## Q6. Is placement/pose simply much harder to optimise than interface confidence? Any work reporting both and finding them decoupled?

### Takeaway

Yes, there is direct evidence of decoupling, from both the design side and the prediction-benchmarking side: AF3 assigns confident scores to interfaces it has invented on decoy targets, and a third of its structurally-failed antibody predictions nonetheless have the **correct epitope** — so confidence, RMSD and epitope-correctness vary independently. Separately, ipTM is known to be a chain-level aggregate that is not a placement metric.

### Cited Findings

**Prediction-side: confidence and placement dissociate in both directions**

- AF3 has "an **innate false positive rate of approximately 3%**," and "AF3 can **hallucinate plausible antibody–target interfaces on decoy proteins**, distributing predicted epitopes across much of a folded target's surface" while avoiding disordered regions — [A Structural Antibody Benchmark of AlphaFold3 reveals Hallucinated Epitopes and a Bias for Orderness, bioRxiv 2026.07.30.741792](https://www.biorxiv.org/content/10.64898/2026.07.30.741792v1)
- Benchmark scale: 3,401 experimentally validated SAbDab complexes plus 23,798 negatives; overall **recall 50%** at the maximum conditions tested — [ibid.](https://www.biorxiv.org/content/10.64898/2026.07.30.741792v1); scale figures also summarised in [a secondary social-media summary](https://x.com/BiologyAIDaily/status/2084295009613050051)
- Direct decoupling statistic: **"approximately 34% of false negatives retained the correct epitope location despite poor structural alignment"** (poor RMSD) — the authors suggest "conformation refinement tools could recover additional true binding predictions" from these — [ibid.](https://www.biorxiv.org/content/10.64898/2026.07.30.741792v1)
- ipTM is structurally unfit as a placement score: "if they trim full-length sequence constructs (e.g. from UniProt) to the interacting domains (or domain+peptide), their **ipTM scores change, even though the structure prediction of the interaction is unchanged**"; the cause is that ipTM "scores the interactions of whole chains," assigning "equal weight to all inter-chain pairs, including those involving low-confidence, non-interacting residues," so it can underreport interface confidence — [Rēs ipSAE loquuntur: What's wrong with AlphaFold's ipTM score and how to fix it, bioRxiv 2025.02.10.637595](https://www.biorxiv.org/content/10.1101/2025.02.10.637595v2); [PMC11844409](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11844409/)
- A secondary (commercial blog) source states the decoupling even more sharply: "two predictions with **similar ipTM can place the paratope on opposite sides of the antigen**," and that "epitope success rates in blind benchmarks hover in the 20-30% range despite high overall scores" — [Orbion blog](https://www.orbion.life/blog/when-alphafold-multimer-gets-protein-complexes-wrong-six-failure-modes-every-structural-biologist-should-know). *Flagged: this is a company blog, not a primary source; I could not locate a peer-reviewed statement of the "opposite sides at equal ipTM" claim, though it is consistent with the AF3 benchmark's hallucinated-epitope finding.*

**Design-side: structural objectives converge far more slowly than confidence objectives**

- "Designing with RMSD loss is more challenging than designing for high pLDDT": ~5× more steps for RMSD than for pLDDT > 90, despite a 10× larger coefficient — [Jeliazkov et al., bioRxiv 2023.05.23.541774](https://www.biorxiv.org/content/10.1101/2023.05.23.541774v1.full)
- Fold-conditioned AF2 design reports its structural (cmap) objective **correlating** with the confidence metrics (pLDDT, PAE, intra-chain contacts) — i.e. when the geometric restraint is written in distogram space, it is *not* decoupled from confidence — yet pose still failed systematically for VHH (side-docking) and entirely for scFv — [bioRxiv 2025.07.02.662497](https://www.biorxiv.org/content/10.1101/2025.07.02.662497v1.full)
- TorchCraft optimises ipTM/iPAE as the primary in-loop objectives and reports Rosetta interface scores only as **post-hoc** validation, not as optimisation targets; no ablation isolates the geometric contribution — [arXiv 2609.19770](https://arxiv.org/html/2609.19770)
- Germinal's reported Pareto frontier between IgLM likelihood and iPAE demonstrates that at least two of its objectives are genuinely competing rather than co-improving — [bioRxiv 2025.09.19.677421](https://www.biorxiv.org/content/10.1101/2025.09.19.677421v1.full)

### Inferences

- The AF3 benchmark's two findings together imply the decoupling is **bidirectional**: correct epitope with bad RMSD (34% of false negatives), and confident-looking interface with no real epitope at all (3% FP on decoys, epitopes spread over the whole surface). A loss built on interface confidence therefore has a large null space in placement, which is consistent with the project's observation that pose improvement is nearly identical with and without the pose term — most of the observed pose improvement is probably coming from the *confidence/contact* terms and from the predictor's own priors, not from the RMSD term.
- This also reframes the project's problem. If the predictor itself will confidently place a binder almost anywhere on the surface, then the deficiency is not a weak restraint but a **near-flat landscape in the placement direction**: the predictor assigns similar confidence to many poses, so no amount of weight on a restraint that reaches sequence only through that predictor will sharpen it. This is consistent with the fold-conditioned paper's side-docking being attributed to predictor/training bias rather than to the loss.
- Practical implication, grounded in the cited work: the leverage is in (1) writing the restraint on **named epitope×paratope residue pairs in distogram space** (Germinal/TorchCraft form) so the gradient is dense and the pose is pinned by contacts rather than by a global superposition; (2) adding an explicit **repulsive/exclusion** term for non-epitope target surface (Wang et al.'s coordinate repulsion; Germinal's `−λ·L_framework`); (3) if hard satisfaction is required, a **dual/ADMM-style update on the observed violation** rather than a bigger coefficient; and (4) **late activation** of the geometric term, after the parameterisation has annealed, rather than ramping it from step 0. The project's existing ipSAE-based validation is well-matched to this, since ipSAE was built precisely because ipTM is not an interface-placement metric.

### Gaps

- No study I found reports a **correlation coefficient** between a design-loop's interface confidence metric and the resulting pose RMSD-to-intended-epitope across a design set. The AF3 benchmark gives categorical rates (34%, 3%, 50% recall) but not a scatter/correlation; the "20-30% blind epitope success" figure comes only from the commercial blog.
- BindEnergyCraft (arXiv 2505.21241) positions itself against confidence-metric optimisation and would be the most directly relevant source for a quantified pose-vs-confidence decoupling in a *design* loop, but its PDF was not parseable and I could not extract its formulas, critique language, or numbers.
- I did not find any 2024–2026 work that backpropagates an inter-chain reference-pose RMSD through a predictor and reports its effectiveness, positively or negatively — so there is no published baseline against which the project's 22.9 vs 25.3 Å result can be compared.
