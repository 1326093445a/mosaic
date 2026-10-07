# Explicit Pairwise Distance Restraints as Model Conditioning (vs. as a Differentiable Loss)

Scope note on the central distinction, used consistently below:

- **Conditioning** = the restraint is an *input tensor* the frozen network consumes. It enters before/inside the trunk (typically as an additive bias on the pair representation `z`), and the network's weights were *trained* to interpret it. The restraint changes the model's learned prior over poses.
- **Guidance / loss** = the restraint is a *penalty on the output*, applied either (a) as a gradient term during diffusion sampling (Boltz-2 steering, experiment-guided AF3), or (b) as a differentiable objective backpropagated to inputs (the user's distogram restraint; resTrain; ColabDock). The network's weights never saw it; it fights the model's prior rather than informing it.
- A third hybrid exists and is directly relevant: **resTrain** optimizes a *pair-representation bias* (a conditioning-shaped tensor) *by gradient descent on a loss*, with weights frozen.

---

## Q1. Boltz-1 / Boltz-2: the YAML `constraints` block — exact schema, `max_distance` semantics, hard vs. soft, and the training task that taught the model to use it

### Takeaway
Boltz-2 is the cleanest published example of true restraint **conditioning**: `pocket` and `contact` constraints become a categorical pair-type matrix plus a continuous threshold matrix, embedded and **added to the initial pair representation** `z_init`, with weights trained on restraints randomly sampled from ground-truth structures. `max_distance` is an **upper bound** (default 6.0 Å), it is a **soft prior** by default, and `force: true` additionally turns it into a flat-bottom diffusion-time guidance potential. Critically, the feature set includes a learned **"UNSELECTED"** class for every non-restrained pair — an exclusionary negative signal a loss term cannot express.

### Cited Findings

**Schema (verbatim from the parser docstring)** — `src/boltz/data/parse/schema.py`, lines 968–979 — [boltz/src/boltz/data/parse/schema.py](https://github.com/jwohlwend/boltz/blob/main/src/boltz/data/parse/schema.py):
```yaml
constraints:
    - bond:
        atom1: [A, 1, CA]
        atom2: [A, 2, N]
    - pocket:
        binder: E
        contacts: [[B, 1], [B, 2]]
        max_distance: 6
    - contact:
        token1: [A, 1]
        token2: [B, 1]
        max_distance: 6
```

- Parsing block is `schema.py:1511–1594`. `pocket` requires keys `binder` and `contacts` (`schema.py:1529–1530`); `contact` requires `token1` and `token2` (`schema.py:1564–1565`). — [schema.py](https://github.com/jwohlwend/boltz/blob/main/src/boltz/data/parse/schema.py)
- `max_distance` **default = 6.0 Å** for both `pocket` (`schema.py:1539`) and `contact` (`schema.py:1574`). — [schema.py](https://github.com/jwohlwend/boltz/blob/main/src/boltz/data/parse/schema.py)
- **Boltz-1 limitations, enforced in code:** only one pocket designed chain is supported (`schema.py:1535–1536`: `"Only one pocket binders is supported in Boltz-1!"`), and `max_distance != 6.0` is rejected for Boltz-1 (`schema.py:1540–1541`). `contact` constraints are Boltz-2 only in practice. — [schema.py](https://github.com/jwohlwend/boltz/blob/main/src/boltz/data/parse/schema.py)
- Both constraint types accept an undocumented **`force` flag, default `False`** (`schema.py:1560` for pocket, `schema.py:1592` for contact). Each constraint is stored as the 4-tuple `(binder, contacts, max_distance, force)` / `(token1, token2, max_distance, force)` (`schema.py:1561`, `schema.py:1594`). — [schema.py](https://github.com/jwohlwend/boltz/blob/main/src/boltz/data/parse/schema.py)

**How the restraint becomes an input feature** — `src/boltz/data/feature/featurizerv2.py`:
- Two dense `[N_token, N_token]` matrices are built: `contact_conditioning` (categorical) and `contact_threshold` (float, Å). Initialized at `featurizerv2.py:709–714`. — [featurizerv2.py](https://github.com/jwohlwend/boltz/blob/main/src/boltz/data/feature/featurizerv2.py)
- Categorical vocabulary, `src/boltz/data/const.py:412–418`:
  ```python
  contact_conditioning_info = {
      "UNSPECIFIED": 0,
      "UNSELECTED": 1,
      "POCKET>BINDER": 2,
      "BINDER>POCKET": 3,
      "CONTACT": 4,
  }
  ```
  — [const.py](https://github.com/jwohlwend/boltz/blob/main/src/boltz/data/const.py)
- A **`pocket` constraint marks the entire designed chain chain against each named pocket token**, asymmetrically: `contact_conditioning[binder_mask, idx] = BINDER>POCKET` and `contact_conditioning[idx, binder_mask] = POCKET>BINDER`, with `contact_threshold[...] = max_distance` on both (`featurizerv2.py:716–735`). — [featurizerv2.py](https://github.com/jwohlwend/boltz/blob/main/src/boltz/data/feature/featurizerv2.py)
- A **`contact` constraint marks exactly one token pair** symmetrically: `contact_conditioning[idx1, idx2] = contact_conditioning[idx2, idx1] = CONTACT`, threshold on both (`featurizerv2.py:737–762`). — [featurizerv2.py](https://github.com/jwohlwend/boltz/blob/main/src/boltz/data/feature/featurizerv2.py)
- **The exclusionary semantics (load-bearing):** the matrix is *initialized to `UNSELECTED`*, and only if **no** restraint at all was supplied is the whole matrix rewritten to `UNSPECIFIED` (`featurizerv2.py:1018–1023`):
  ```python
  if np.all(contact_conditioning == const.contact_conditioning_info["UNSELECTED"]):
      contact_conditioning = (contact_conditioning
          - const.contact_conditioning_info["UNSELECTED"]
          + const.contact_conditioning_info["UNSPECIFIED"])
  ```
  So supplying *any* restraint flips every other pair from "no information" to "explicitly not a specified contact." One-hot at `featurizerv2.py:1024–1027`; emitted as features `"contact_conditioning"` and `"contact_threshold"` at `featurizerv2.py:1102–1103`. — [featurizerv2.py](https://github.com/jwohlwend/boltz/blob/main/src/boltz/data/feature/featurizerv2.py)

**Where it enters the network** — `src/boltz/model/modules/trunkv2.py:21–65`, class `ContactConditioning`:
- Input is `[3 one-hot channels (POCKET>BINDER, BINDER>POCKET, CONTACT)] ⊕ [normalized threshold scalar] ⊕ [Fourier embedding of normalized threshold, dim token_z]` → `nn.Linear(token_z + len(contact_conditioning_info) - 1, token_z)` (`trunkv2.py:26–28, 37–54`).
- Threshold normalization: `(contact_threshold - cutoff_min) / (cutoff_max - cutoff_min)` (`trunkv2.py:39–41`).
- `UNSPECIFIED` and `UNSELECTED` are **separate learned `nn.Parameter` vectors** (`self.encoding_unspecified`, `self.encoding_unselected`, `trunkv2.py:30–31`), mixed in at `trunkv2.py:56–64`.
- **Additive bias on the initial pair representation:** `src/boltz/model/models/boltz2.py:430` — `z_init = z_init + self.contact_conditioning(feats)`. — [trunkv2.py](https://github.com/jwohlwend/boltz/blob/main/src/boltz/model/modules/trunkv2.py), [boltz2.py](https://github.com/jwohlwend/boltz/blob/main/src/boltz/model/models/boltz2.py)
- The Boltz-2 paper describes this as pairwise features with "one-hot encoding of the contact type and an encoding of the distance," with distance constraints restricted to **4 Å ≤ d ≤ 20 Å**, encoded via normalized distance plus Fourier embedding. — [Boltz-2 preprint](https://www.biorxiv.org/content/10.1101/2025.06.14.659707v1.full)

**Training task mixture (what taught the model to use restraints)**:
- Featurizer training-time knobs and defaults, `featurizerv2.py:612–618`: `binder_pocket_conditioned_prop=0.0`, `contact_conditioned_prop=0.0`, `binder_pocket_cutoff_min=4.0`, `binder_pocket_cutoff_max=20.0`, `binder_pocket_sampling_geometric_p=0.0`, `only_ligand_binder_pocket=False`, `only_pp_contact=False`. — [featurizerv2.py](https://github.com/jwohlwend/boltz/blob/main/src/boltz/data/feature/featurizerv2.py)
- Pocket-conditioning generation (`featurizerv2.py:766–857`): `while random.random() < binder_pocket_conditioned_prop:` pick a random chain as "designed chain" (ligand preferred), **sample the cutoff from a 1/d distribution over [4, 20] Å** via `sample_d` (`featurizerv2.py:58+`), compute the true pocket within that cutoff, then **subsample it geometrically** with `select_subset_from_mask(pocket_mask, binder_pocket_sampling_geometric_p)` (`featurizerv2.py:580+`), then write `BINDER>POCKET` / `POCKET>BINDER`.
- Contact-conditioning generation (`featurizerv2.py:860–1017`): same `while random.random() < contact_conditioned_prop:` loop, sample a cutoff, pick a random chain, enumerate true cross-chain contacts, pick **one random pair**, mark it `CONTACT`.
- Boltz-1 training configs in the repo: `scripts/train/configs/structure.yaml:67–70` and `scripts/train/configs/confidence.yaml:69–72` both set `train_binder_pocket_conditioned_prop: 0.3`, `val_binder_pocket_conditioned_prop: 0.3`, `binder_pocket_cutoff: 6.0`, `binder_pocket_sampling_geometric_p: 0.3`. — [structure.yaml](https://github.com/jwohlwend/boltz/blob/main/scripts/train/configs/structure.yaml)

**Hard vs. soft, and the `force` path**:
- Independent description: "Scientists can define residue-pair distances or a set of pocket residues, and Boltz-2 will apply **soft potentials** that nudge the prediction to satisfy those geometric constraints." — [Rowan, Boltz-2 FAQ / tool page](https://www.rowansci.com/blog/boltz2-faq)
- `force: true` routes the same restraint into an inference-time potential. `src/boltz/model/potentials/potentials.py:653–668`, `class ContactPotentital(FlatBottomPotential, DistancePotential)`: `upper_bounds = feats["contact_thresholds"][0]`, `lower_bounds = None`, `k = 1`, with a `negation_mask` and a `contact_union_index` implementing the pocket's "min over a set" semantics. — [potentials.py](https://github.com/jwohlwend/boltz/blob/main/src/boltz/model/potentials/potentials.py)
- `get_potentials` (`potentials.py:755–785`) registers `ContactPotentital` whenever `fk_steering` or `contact_guidance_update` is set, with `guidance_interval=4`, `guidance_weight = PiecewiseStepFunction(thresholds=[0.25, 0.75], values=[0.0, 0.5, 1.0])`, `resampling_weight=1.0`, and `union_lambda = ExponentialInterpolation(start=8.0, end=0.0, alpha=-2.0)` (a soft-min temperature annealed toward a hard min). — [potentials.py](https://github.com/jwohlwend/boltz/blob/main/src/boltz/model/potentials/potentials.py)
- `contact_guidance_update` **defaults to `True`** in `BoltzSteeringParams` (`src/boltz/main.py:148–156`), i.e. contact/pocket guidance is on by default at inference even though `--use_potentials` (physical guidance / FK steering) defaults to `False` (`main.py:969–971, 1070, 1309–1311`). — [main.py](https://github.com/jwohlwend/boltz/blob/main/src/boltz/main.py)

### Inferences
- Boltz-2 applies a restraint through **two independent channels**: conditioning (`z_init` bias, trained) and guidance (flat-bottom gradient during diffusion, untrained). The user's distogram loss is a third channel of the same family as the second. The conditioning channel is the only one that can alter the model's *prior*.
- The `UNSELECTED` class is the mechanism most likely to suppress a rotated/flipped alternative registry: it tells the trunk "these are the contacts, and by implication the others are not." A purely additive loss on named pairs has no way to say "and no other arrangement." Adding an explicit *repulsive* term on pairs that are close in the decoy registry but far in the reference would be the loss-side analogue.
- A `pocket` constraint in Boltz is **orientation-indifferent by construction** (whole designed chain chain × named pocket tokens, with a single shared threshold). It cannot distinguish a 180° flip that keeps the same chain in the same pocket. Only the `CONTACT` class carries pairwise registry information. For the user's failure mode (right site, right face, wrong rotation), `pocket` is the wrong primitive and `contact` is the right one.
- Because every training restraint is derived from the ground truth, the model has **no training signal for rejecting a wrong restraint**. Conditioning is therefore credulous: it will distort a structure to satisfy a bad restraint rather than ignore it. This is a safety property to be aware of when the "reference arrangement" is itself a modeling assumption.

### Gaps
- **No quantitative ablation of pocket/contact conditioning is reported in the Boltz-2 paper.** Confirmed by direct read: "No quantitative ablation data on conditioning effectiveness is reported," and "No explicit ablation studies on accuracy metrics (DockQ, RMSD, success rate) are provided for these conditioning mechanisms." — [Boltz-2 preprint](https://www.biorxiv.org/content/10.1101/2025.06.14.659707v1.full). So the *effectiveness* of Boltz conditioning is undocumented; only the mechanism is.
- The exact Boltz-2 training values for `binder_pocket_conditioned_prop` / `contact_conditioned_prop` are not in the public repo — `scripts/train/configs/` contains only `structure.yaml`, `confidence.yaml`, `full.yaml` (the Boltz-1 recipe, which has no `contact_conditioned_prop` key at all). The Boltz-2 values must be inferred from the featurizer's cutoff range (4–20 Å), which matches the paper.
- `force` is undocumented in the public YAML docs; its behavior is only readable from source.

---

## Q2. Chai-1 / Chai-2: restraint support, format, and reported effect on accuracy

### Takeaway
Chai-1 is trained to consume three distinct restraint feature families as input — **pocket**, **contact**, and **docking** — with explicitly stated functional forms and training sampling distributions. The published ablation is strong and quantitative: conditioning on **four randomly sampled specified-site residues at 8 Å more than doubles the antibody–antigen DockQ success rate** across all quality cutoffs. Chai's **`docking` restraint** (one-hot pairwise distance bins between token subsets) is the closest published primitive to "pin the whole arrangement," and is the one most relevant to an orientation failure.

### Cited Findings

**Exact functional forms (from the Chai-1 technical report)** — [Chai-1 preprint](https://www.biorxiv.org/content/10.1101/2024.10.10.615955v2.full):
- **Pocket constraint:** specified by token ID *i*, chain ID *C*, and threshold *θ_P*, encoding `min_{j∈C} ‖x_i − x_j‖ ≤ θ_P`.
- **Contact constraint:** token pair *i*, *j* with threshold *θ_D*, encoding `‖x_i − x_j‖ ≤ θ_D`.
- **Docking constraint:** one-hot encoded pairwise distances between *subsets of tokens*, using four bins: **[0–4 Å, 4–8 Å, 8–16 Å, >16 Å]**.

**Training procedure** — [Chai-1 preprint](https://www.biorxiv.org/content/10.1101/2024.10.10.615955v2.full):
- Thresholds sampled randomly during training: **θ_P ∈ (6, 20) Å**; **θ_D ∈ (6, 30) Å**.
- Satisfying constraints are derived from ground-truth structures.
- **Chain-wise and token-wise dropout** on constraint features; each feature included independently with **10% probability**; when omitted, a **learnable mask value** replaces the feature.
- The **number of distance and pocket restraints followed a geometric distribution.**

**Restraint file format** — `chai-lab/examples/restraints/README.md` CSV columns: `restraint_id, chainA, res_idxA, chainB, res_idxB, connection_type, confidence, min_distance_angstrom, max_distance_angstrom, comment`. Units are Ångströms; "the distances indicate an **upper bound** on how far apart we expect those two residues to be." The **`confidence` and `min_distance` fields are currently not used by the model** and exist "for future-proofing." — [chai-lab restraints README](https://github.com/chaidiscovery/chai-lab/blob/main/examples/restraints/README.md)

**Reported effectiveness (Chai-1's own ablation)** — [Chai-1 preprint](https://www.biorxiv.org/content/10.1101/2024.10.10.615955v2.full):
- Antibody–antigen blind baseline: **~35% DockQ success rate**.
- "Conditioning on **four randomly sampled specified-site residues more than doubles the DockQ success rate** across all quality cutoffs compared to baseline."
- Absolute high-quality predictions remain limited (**4–8%** at the strictest thresholds).
- Chai Discovery's own framing: prompting with known information "improves performance by double-digit percentage points"; "conditioning on just four sampled specified-site residues can more than double the prediction success rate."

**Independent ablation (peer-reviewed)** — Clifford et al., *Protein Science* 2026 / bioRxiv 2025.09.17.676770 — [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.09.17.676770v2.full); [Protein Science](https://onlinelibrary.wiley.com/doi/10.1002/pro.70730):
- "A pocket restraint is a **soft restraint (i.e., not strictly enforced)** that any residue in one chain be in contact with a specific residue in another."
- **Optimal pocket distance = 10 Å** for both light and heavy chains; **6–11 Å performed equivalently; 5 Å and 12 Å performed worst.** (A distance-sensitivity curve with a broad plateau — directly implementable.)
- Ground-truth-derived pocket restraints on **both** antibody chains at 10 Å vs. no restraints: acceptable DockQ (≥0.23) **~35% → ~60%**; medium (≥0.49) **~20% → ~50%**; high (≥0.80) **~5% → ~20%**. Heavy chain only: ~50% / ~35% / ~10%.
- With *predicted* (not ground-truth) sites the gains collapse to a few points: BepiPocket +4% acceptable/medium, +2% high; DiscoPocket with MSA-enhanced Chai-1 48% vs. 42% acceptable, 40% vs. 33% medium, 15% vs. 12% high.
- **Specified site redundancy fell >5-fold (0.56 → 0.10 for BepiPocket; 0.62 → 0.06 for DiscoPocket)** — restraints diversify which specified site is sampled across seeds.
- Gains correlate with antigen model quality: antigen RMSD < 1.0 Å gave ~10% improvement vs. ~2% for poor antigen predictions.
- Also independently reported: "providing n=2 randomly selected contact restraints based on experimental ground truth significantly improved the interface DockQ scores between the viral protein and the antibody chains."

### Inferences
- Chai's **pocket** restraint (`min over chain`) and Boltz's **pocket** restraint (whole-chain mask) are the same orientation-indifferent primitive. Chai's **contact** restraint is registry-bearing. Chai's **docking** restraint — dense one-hot distance bins over *token subsets* — is the only published primitive designed to transmit a full relative arrangement, and is the natural conditioning analogue of the user's distogram-space loss on reference pairs.
- The 6–11 Å plateau with degradation at 5 Å and 12 Å indicates the model is being used as a *prior shaper*, not a geometry solver: too tight over-constrains, too loose stops localizing. A restraint set designed to fix orientation should use several pairs at ~8 Å rather than one pair at a tight distance.
- The ground-truth-vs-predicted gap (25-point gain vs. 4-point gain) means essentially all published restraint-conditioning gains are **computational, oracle-restraint numbers**. The user's reference-arrangement restraints are closer to the oracle case, so the larger numbers are the applicable ones — conditional on the reference arrangement being correct.

### Gaps
- **Chai-2 restraint support is undocumented.** I found no public specification of a restraint input format, schema, or restrained-vs-unrestrained ablation for Chai-2. Chai-2 is presented as a zero-shot antibody/designed chain *design* model; whether it exposes the Chai-1 restraint features is not stated in any source I could reach.
- Chai-1's docking-restraint ablation is not broken out separately in the technical report; only pocket/specified site conditioning is quantified.
- The "10% probability" statement is ambiguous between "included 10% of the time" and "dropped out 10% of the time" as fetched; I could not disambiguate from the text retrieved.

---

## Q3. AlphaLink and AlphaLink2: AlphaFold conditioned on crosslinking-MS restraints

### Takeaway
AlphaLink2 is the closest published precedent for "restraints fix a complex the unrestrained model gets wrong": crosslinks are injected **into the pair representation** of a UniFold-based AlphaFold2-Multimer and the network is **fine-tuned** on simulated crosslinks, lifting DockQ on 8 hard heteromeric CASP15 targets from a 0.14 baseline to 0.48–0.62 depending on the source. It is also the only source that reports a **per-restraint false-discovery rate as part of the input format**, and the only one reporting that a *single* real crosslink flipped a complex from unconfident to confident.

### Cited Findings

**Encoding and base model** — [AlphaLink2, Nat. Commun. 2024 / bioRxiv 2023.06.07.544059v2](https://www.biorxiv.org/content/10.1101/2023.06.07.544059v2.full):
- The method "integrat[es] crosslinking mass spectrometry (MS) data directly into the **pair representation** of AlphaFold2."
- Code base was "chang[ed] from OpenFold to **Uni-Fold**"; the network was **fine-tuned with simulated succinimidyl 4,4-azipentanoate (SDA) crosslinks**.
- Simulated training/benchmark crosslink data: **10% sequence coverage and 20% false-discovery rate**, giving a median of **33 links per protein–protein interaction (including 7 false links)**.
- Training-time SDA simulation target distance: **25 Å Cα–Cα**; a crosslink is scored as satisfied at **< 30 Å Cα–Cα**. — [AlphaLink2 README](https://github.com/Rappsilber-Laboratory/AlphaLink2)

**Input format (implementable)** — [AlphaLink2 README](https://github.com/Rappsilber-Laboratory/AlphaLink2):
- "AlphaLink takes as input a CSV (.csv or .txt) with a list of crosslinked residue pairs with a **false-discovery rate (FDR)**."
- Columns, space-separated: `residueFrom chain1 residueTo chain2 FDR`; residues **1-indexed**; chains lettered A–Z by FASTA order. Example:
  ```
  1 A 50 B 0.2
  5 A 5 A 0.1
  ```
- At run time the restraints are passed as a **pickled, gzipped dictionary** (`/path/to/crosslinks.pkl.gz`); `parse_mzidentml.py` converts mzIdentML.

**Quantitative results** — [bioRxiv 2023.06.07.544059v2](https://www.biorxiv.org/content/10.1101/2023.06.07.544059v2.full):
- 8 challenging heteromeric CASP15 targets with simulated SDA crosslinks: **median DockQ 0.14 → 0.48**, with **7 of 8 targets reaching DockQ ≥ 0.23** (acceptable).
- **Conflicting figure:** the abstract-level summary circulating for the same work states the simulated-SDA improvement as **"DockQ score from 0.14 to 0.62 on average,"** "which matches the average DockQ = 0.62 of the best predictions in CASP15." The 0.48 figure is a *median* from the v2 full text and 0.62 is quoted as a *mean*; this may be the whole discrepancy, but I could not reconcile them inside one document. — [Nature Communications record](https://www.nature.com/articles/s41467-024-51771-2) (paywalled/IDP-gated when fetched) vs. [bioRxiv v2 full text](https://www.biorxiv.org/content/10.1101/2023.06.07.544059v2.full)
- *Bacillus subtilis* in-cell DSSO data: **135 dimeric PPIs, median 1 crosslink per PPI**; median model confidence **0.42 → 0.60**; 46 interactions reached >0.75 confidence (a 35% gain).
- Single-restraint case: "the model confidence of the **CodY–YppF** interaction improves from **0.25 to 0.81 based on a single crosslink**."
- Sampling economy: crosslinks "focus sampling on the interesting regions, reducing the amount of sampling required," versus CASP15 competitors using "up to 2400x more sampling" (elsewhere quoted as 120x).

**Orientation / binding-mode evidence** — [bioRxiv 2023.06.07.544059v2](https://www.biorxiv.org/content/10.1101/2023.06.07.544059v2.full):
- CRL4 complex: AlphaLink "position[ed] the viral **Vpr** protein inside the experimental density in agreement with the crystal structure," while AlphaFold "places Vpr **incorrectly**." This is a placement/orientation claim, not merely a proximity claim.
- Mechanism framing used throughout: crosslinking MS improves modelling "by **helping to identify interfaces**, focusing sampling, and improving model selection."
- Model selection: "Selecting first by **crosslink satisfaction** and then for model confidence, we succeed in selecting high-quality predictions **even when the model confidence is not discriminative**."
- Antibody–antigen targets, where "co-evolutionary signal is lower," reached "at least acceptable solutions" with restraints.

### Inferences
- The FDR field is the one published mechanism for telling a *conditioned* model that a restraint might be wrong. Nothing in Boltz or Chai has an equivalent (Chai's `confidence` column exists but is explicitly unused). If the user's reference arrangement is uncertain, no mainstream AF3-class predictor can be told so through its restraint input.
- The "crosslink satisfaction then confidence" selection result transfers directly and cheaply: *restraint satisfaction as a selection/ranking criterion is reported to work where model confidence does not*. For the user's rotated/flipped decoys — which the site-level criteria cannot distinguish and which may carry high ipSAE — ranking candidates by reference-pair restraint satisfaction is a documented, independently-useful filter that does not require any conditioning mechanism at all.
- AlphaLink2's effectiveness is attributable to **fine-tuning**, not to the injection point alone. A frozen AF3-class model with no restraint-trained weights cannot be expected to reproduce these gains from an input hack.

### Gaps
- **The exact restraint encoding is not disclosed.** Direct read of the v2 preprint: "The paper does not provide explicit technical details about the encoding mechanism… does not specify whether a distogram-like probability distribution is used, distance bin parameters (range, width), distribution shape formulation, [or] how false-discovery-rate uncertainty is mathematically encoded." The original AlphaLink (Stahl et al., *Nat. Biotechnol.* 2023, [10.1038/s41587-023-01704-z](https://www.nature.com/articles/s41587-023-01704-z)) is where the bin-level encoding would be specified; I was not able to retrieve its methods section within this task's budget. **Bin widths, distance-distribution shape, and FDR→feature mapping are unresolved.**
- No statement found about **how many crosslinks are needed**, or about sparse restraints leaving orientation underdetermined. The data points bracket the question (median 1 real crosslink helps; median 33 simulated links were used for the CASP15 numbers) but no analysis of restraint count vs. accuracy was reported.
- No restraint-dropout or restraint-count curriculum details disclosed.

---

## Q4. AlphaFold3 and its open reimplementations: is there any restraint-conditioning mechanism?

### Takeaway
**Stock AlphaFold3 has no restraint-conditioning input** — only MSAs, templates, and bonded-atom-pair topology. Everything in the AF3 ecosystem that looks like a restraint is either (a) a *topological* hack (model the crosslinker as a covalently bonded ligand), or (b) *diffusion-time guidance* (a loss, not a conditioning input). The open AF3 reimplementations — Chai-1, Boltz-2, Protenix — are where restraint conditioning actually lives, because they retrained for it.

### Cited Findings
- "Chai-1 **reproduces [the] AlphaFold 3 network incorporating distance restraints**… Chai-1 has been **trained to accept extra input features in the form of maximum distance restraints between token pairs**." — [resTrain preprint](https://www.biorxiv.org/content/10.64898/2026.07.02.736010v1.full)
- The absence of a native restraint input in AlphaFold2/3 is a long-standing open request: — [google-deepmind/alphafold issue #181, "Use contact/distance restraints"](https://github.com/google-deepmind/alphafold/issues/181)
- **The covalent-ligand workaround for AF3** (this is conditioning of a kind — but *topological*, not a soft distance prior): crosslinks were "modeled as **covalently-bound ligands**, enabling the incorporation of experimentally derived spatial restraints directly into the structure prediction process," with "the possible bonds to amino acid residues… defined using the **`bondedAtomPairs` parameter**." Supports **13 crosslinker types** (DSSO, DSS, DSG, BS3, azide-A-DSBSO, others). — [Improving AlphaFold 3 structural modeling by incorporating explicit crosslinks, bioRxiv 2024.12.03.626671v4](https://www.biorxiv.org/content/10.1101/2024.12.03.626671v4.full)
- **Experiment-guided AF3 is guidance, not conditioning:** "The diffusion model in AlphaFold3 allows the incorporation of **arbitrary restraints on atomic coordinates at each step of the reverse diffusion process during inference**"; "experimental likelihoods enter as **guidance terms during sampling**," with "likelihoods of experimental observations calculated given each individual ensemble member and on ensemble averages (for example, calculated electron density, average interatomic distances or order parameters)," and "using their **gradient as guidance** during the diffusion process." — [Experiment-guided AlphaFold3, Nat. Biotechnol. 2026](https://www.nature.com/articles/s41587-026-03166-5); [bioRxiv 2025.10.11.681796](https://www.biorxiv.org/content/10.1101/2025.10.11.681796.full.pdf); [Inverse problems with experiment-guided AlphaFold, arXiv:2502.09372](https://arxiv.org/pdf/2502.09372)
- Boltz-2's other conditioning channel, for comparison of mechanism: **method conditioning** is a "one-hot encoding for different experimental method types" on the single (token) representation, trained across X-ray, EM, solution NMR, solid-state NMR, MD, and distillation from AF2/Boltz-1. — [Boltz-2 preprint](https://www.biorxiv.org/content/10.1101/2025.06.14.659707v1.full)

### Inferences
- The AF3 covalent-ligand trick is the only way to express something close to a **hard** distance constraint as an AF3-class *input*, and it works by exploiting a mechanism (bonded ligand topology) that the model *was* trained on. The general lesson for the user's setting: repurposing an existing, trained conditioning channel beats inventing an untrained input channel.
- If the user's frozen predictor is genuinely AF3-architecture-without-restraint-training, there is **no input channel to write a distance restraint into**. The user's distogram loss is then not a second-best option; it is the only option short of swapping to Boltz-2 or Chai-1.

### Gaps
- I found no documentation of restraint conditioning in **OpenFold3** or **Protenix** at the schema level within this task; resTrain refers to Protenix as an input-conditioning method but I did not verify its schema. OpenDDE specifically is not covered by any public source I could find.

---

## Q5. HADDOCK ambiguous interaction restraints (AIRs): functional form, why ambiguity is deliberate, active/passive

### Takeaway
HADDOCK's AIR is the canonical *deliberately orientation-indifferent* restraint: an effective distance `d_eff = [Σ (1/r⁶)]^(−1/6)` summed over **all** atom pairs between an active residue of one partner and all active+passive residues of the other, with a default **2 Å** upper bound. The 1/r⁶ form is chosen explicitly so that "the AIRs are satisfied **as soon as any two atoms of the two proteins are in contact**" — i.e. the restraint specifies *which surfaces touch* and deliberately leaves the relative orientation to the energy function. This is the sharpest available statement of the conceptual limit the user is running into.

### Cited Findings
- **Effective distance formula, verbatim:** `deff = [Sum(1/r6)]^-1/6`. — [HADDOCK2.2 manual, generate_air_help](https://www.bonvinlab.org/software/haddock2.2/generate_air_help/)
- **Rationale for 1/r⁶ (verbatim framing):** "a 1/r^6 sum averaging is used, **not by analogy to NOE restraints, but because this mimics the attractive part of a Lennard-Jones potential and ensures that the AIRs are satisfied as soon as any two atoms of the two proteins are in contact**. More specifically, the effective distance d_eff will **always be shorter than the shortest distance entering the sum**." — [HADDOCK2.2 generate_air_help](https://www.bonvinlab.org/software/haddock2.2/generate_air_help/) (also surfaced verbatim via the HADDOCK2.4 AIR manual page)
- **Default upper distance bound:** "the current **upper distance limit default value is 2A**." The manual notes this looks short but that d_eff becomes much shorter than any individual distance because of the high ambiguity (thousands of distance terms in the sum). — [HADDOCK2.2 generate_air_help](https://www.bonvinlab.org/software/haddock2.2/generate_air_help/)
- **Restraint definition / scope of the sum:** AIRs are "defined as **ambiguous intermolecular distances between any atom of an active residue of molecule A and any atom of both active and passive residues of molecule B** (and inversely for molecule B)." — [HADDOCK2.4 AIR manual](https://www.bonvinlab.org/software/haddock2.4/airs/)
- **Active residues:** "experimentally identified to be involved in the interaction between the two molecules **AND** solvent accessible (either main chain or side chain **relative accessibility should be typically > 40%**)" — with the HADDOCK3 manual noting the threshold can be relaxed "as low as 15%" and describing active residues as "of central importance for the interaction… such as residues whose knockouts abolish the interaction or those where the chemical shift perturbation is higher." — [HADDOCK2.4 AIR manual](https://www.bonvinlab.org/software/haddock2.4/airs/); [HADDOCK3 user manual, restraints](https://www.bonvinlab.org/haddock3-user-manual/bpg/restraints.html)
- **Passive residues:** "all **solvent accessible surface neighbors of active residues (<6.5 Å)**"; they "contribute for the interaction, but are deemed of less importance. **If such a residue does not belong in the interface there is no scoring penalty.**" — [HADDOCK3 user manual, restraints](https://www.bonvinlab.org/haddock3-user-manual/bpg/restraints.html); [HADDOCK2.4 education tutorial](https://www.bonvinlab.org/education/HADDOCK24/HADDOCK24-protein-protein-basic/)
- **Random removal — ambiguity enforced stochastically as well as structurally:** AIRs are subject to random removal — "for each docking trial, a **fraction of these restraints will be randomly removed, which ensures a wider sampling**" — whereas "**unambiguous restraints** are **not** subject to random removal, therefore **all of them must be satisfied**." — [HADDOCK3 user manual, restraints](https://www.bonvinlab.org/haddock3-user-manual/bpg/restraints.html)
- **Bias, not enforcement:** "AIRs are included in the energy function being minimized, [so] the resulting complexes will be **biased towards them**." — [HADDOCK2.4 tutorial](https://www.bonvinlab.org/education/HADDOCK24/HADDOCK24-protein-protein-basic/)
- **Surface-contact restraint defaults (a related restraint class):** upper distance limit "**7 Å** (both molecules contain CA and/or P atoms) or **4.5 Å** (only one molecule contains CA and/or P atoms) or **2 Å** (no molecule contains CA and/or P atoms)." — [HADDOCK2.4 AIR manual](https://www.bonvinlab.org/software/haddock2.4/airs/)
- **Restraint satisfaction as a pose discriminator (independent precedent):** "The pyDockRST software uses the **percentage of satisfied distance restraints**, together with the electrostatics and desolvation binding energy, **to identify correct docking orientations**." — [Present and future challenges and limitations in protein–protein docking, *Proteins* 2010](https://onlinelibrary.wiley.com/doi/10.1002/prot.22564)

### Inferences
- The HADDOCK design is an explicit statement that **an ambiguous/pocket-style restraint is incapable of determining orientation, by construction, and that this is intentional**: HADDOCK gets the orientation from the physical energy function (vdW, electrostatics, desolvation) during rigid-body and flexible refinement, not from the restraint. A learned predictor has no equivalent physical energy stage; its "energy function" is its learned prior. So when a learned predictor's prior puts the designed chain in the wrong rotation, an ambiguous restraint has nothing to correct it with.
- The user's failure (correct site, correct face, wrong rotation) is exactly the residual degeneracy an AIR leaves behind. The fix in HADDOCK's own idiom is to add **unambiguous restraints** — specific, named, non-removable pairs — which is the structural analogue of Boltz's `CONTACT` class and Chai's `contact`/`docking` restraints, and of the user's named reference contact pairs.
- HADDOCK's random removal of a fraction of AIRs per trial is a notable design choice: it treats restraints as individually unreliable *by default*, which is the opposite of how a conditioned learned predictor treats them.

### Gaps
- **I could not retrieve a primary source stating the AIR energy equation itself** (the square-well/flat-bottom expression, the force constant `k_AIR`, and whether the penalty outside the well is harmonic or linear). Both the HADDOCK2.2 and 2.4 manual pages and the HADDOCK3 best-practice guide omit it; the Dominguez/Boelens/Bonvin 2003 *JACS* PDF would not parse, and the Utrecht "Driven Structural Modelling" chapter returned HTTP 403. The widely-repeated form `E_AIR = k·(d_eff − d_upper)²` for `d_eff > d_upper` and 0 otherwise is consistent with CNS NOE restraint conventions, **but I am not citing it as established because I could not verify it against a primary source.**
- No HADDOCK source I reached states a required *number* of restraints, or analyzes orientation determinacy as a function of restraint count.

---

## Q6. Flat-bottom / square-well restraint potentials: exact formulas, and whether they are reported preferable to harmonic

### Takeaway
Boltz's flat-bottom potential is **linear outside the bounds, not harmonic** — a constant-magnitude gradient `±k` — and every restraint class in Boltz (PoseBusters geometry, connections, VDW overlap, chirality, stereo, planarity, templates, contacts) is implemented by the same `FlatBottomPotential` base class. I found the exact implementation but **no published head-to-head comparison establishing flat-bottom as preferable to harmonic** for restraint-driven complex prediction.

### Cited Findings

**Exact Boltz implementation** — `src/boltz/model/potentials/potentials.py:231–277`, `FlatBottomPotential.compute_function`:
```python
neg_overflow_mask = value < lower_bounds
pos_overflow_mask = value > upper_bounds
energy = torch.zeros_like(value)
energy[neg_overflow_mask] = (k * (lower_bounds - value))[neg_overflow_mask]
energy[pos_overflow_mask] = (k * (value - upper_bounds))[pos_overflow_mask]
...
dEnergy[neg_overflow_mask] = -1 * k...   # constant gradient
dEnergy[pos_overflow_mask] =  1 * k...
```
So: `E = 0` inside `[lower, upper]`; `E = k·(lower − d)` below; `E = k·(d − upper)` above. **L1, not L2.** Unbounded sides are `±inf`. A `negation_mask` mechanism (lines 248–260) swaps bounds to express "this pair must NOT be within the threshold." — [potentials.py](https://github.com/jwohlwend/boltz/blob/main/src/boltz/model/potentials/potentials.py)

**Every Boltz restraint is a flat-bottom** — all of these inherit `FlatBottomPotential` (`potentials.py`): `PoseBustersPotential` (386), `ConnectionsPotential` (425), `VDWOverlapPotential` (437), `SymmetricChainCOMPotential` (498), `StereoBondPotential` (532), `ChiralAtomPotential` (553), `PlanarBondPotential` (573), `TemplateReferencePotential` (599), `ContactPotentital` (653). `k = torch.ones_like(...)` in every case — the per-class strength is carried by `guidance_weight`, not by `k`. — [potentials.py](https://github.com/jwohlwend/boltz/blob/main/src/boltz/model/potentials/potentials.py)

**Representative buffer/bound values (all from `get_potentials`, `potentials.py:670–785`)**:
- `ConnectionsPotential`: `upper_bounds = parameters["buffer"] = 2.0` Å, `guidance_weight = 0.15`, `resampling_weight = 1.0`.
- `PoseBustersPotential`: `bond_buffer = 0.125`, `angle_buffer = 0.125`, `clash_buffer = 0.10` (multiplicative on RDKit bounds), `guidance_weight = 0.01`.
- `VDWOverlapPotential`: `buffer = 0.225`, `guidance_weight = PiecewiseStepFunction(thresholds=[0.4], values=[0.125, 0.0])`.
- `ChiralAtomPotential` / `StereoBondPotential`: `buffer = 0.52360` rad (= 30°); `PlanarBondPotential`: `buffer = 0.26180` rad (= 15°).
- `SymmetricChainCOMPotential`: lower bound only, `buffer = ExponentialInterpolation(...)`, `guidance_weight = 0.5`.
- `ContactPotentital`: `guidance_interval = 4`, `guidance_weight = PiecewiseStepFunction(thresholds=[0.25, 0.75], values=[0.0, 0.5, 1.0])` — **zero for the first quarter of the reverse diffusion trajectory, 0.5 in the middle, 1.0 for the last quarter.**
- `TemplateReferencePotential`: `upper_bounds = feats["template_force_threshold"]` on Cβ reference positions, applied only where `template_force` is set; `guidance_weight = 0.1`, `guidance_interval = 2` (`potentials.py:599–651`).
- **`force: true` on a template requires a `threshold`:** `schema.py:1693–1698` — `"Template {id} must have threshold specified if force is set to True"`. — [schema.py](https://github.com/jwohlwend/boltz/blob/main/src/boltz/data/parse/schema.py)

**Chai's restraint semantics are also flat-bottom-shaped by construction:** pocket is `min_{j∈C} ‖x_i − x_j‖ ≤ θ_P` and contact is `‖x_i − x_j‖ ≤ θ_D` — upper-bound-only inequalities with no reward for being closer. — [Chai-1 preprint](https://www.biorxiv.org/content/10.1101/2024.10.10.615955v2.full)

**HADDOCK's AIR is also an upper-bound-only restraint on an effective distance** (`d_eff ≤ 2 Å` default), with no penalty for shorter — [HADDOCK2.2 generate_air_help](https://www.bonvinlab.org/software/haddock2.2/generate_air_help/)

### Inferences
- The convergent design across Boltz, Chai, and HADDOCK is: **upper-bound-only, zero-inside, no reward for over-satisfaction.** A harmonic restraint on a *target* distance (rather than a bound) implicitly asserts the distance is known, and over-constrains; this is the most likely reason none of these systems uses one.
- Boltz's choice of **linear** rather than quadratic outside the well means the restraint gradient does not vanish as the violation shrinks and does not blow up for large violations — a bounded, scale-free pull. For the user's loss term, this maps to: use an L1 hinge on the bound (`relu(d − d_upper)`), not a squared error on a reference distance. The squared form will be dominated by the largest-violating pair, which in a 180° flip is precisely the pairs that are hardest to fix.
- The `ContactPotentital` guidance schedule (0.0 → 0.5 → 1.0 across the trajectory) is the single most transferable hyperparameter in this set: Boltz deliberately **does not apply restraints early in sampling**, letting the model's prior establish a global arrangement first, then tightening. A restraint loss applied at uniform weight throughout is doing something different, and arguably worse, than what the production system does.
- The `negation_mask` mechanism (express "must NOT be within θ") is the structural counterpart to the exclusionary `UNSELECTED` conditioning class, and is a direct, implementable answer to a rotation/flip failure: restrain the decoy-registry pairs *apart* as well as the reference pairs *together*.

### Gaps
- **No published ablation comparing flat-bottom vs. harmonic restraints for restraint-driven complex prediction.** I found no paper that tests both potential shapes on the same benchmark. The preference for flat-bottom is universal in implementations but, as far as I can determine, never justified empirically in print for this specific purpose.

---

## Q7. CRUCIAL: does any source report that restraints fix ORIENTATION specifically, as opposed to merely bringing chains into proximity?

### Takeaway
**Yes, but only as individual case studies, never as a systematic measurement.** Three documented cases show a restrained model correcting a *placement/orientation* error, including one directly analogous to the user's problem (a **designed chain** mispositioned at 6.47 Å RMSD, corrected to 0.81 Å by a **single** input restraint). But no source reports an orientation- or registry-specific metric across a benchmark — every aggregate number is DockQ or a DockQ success rate, both of which conflate interface identity with orientation. And HADDOCK's own documentation states that its ambiguous restraints are satisfied the instant *any* two atoms touch, i.e. that orientation is explicitly *not* what an ambiguous restraint determines.

### Cited Findings

**Case 1 — designed chain, AF3 + one explicit crosslink (closest analogue to the user's failure)** — [bioRxiv 2024.12.03.626671v4](https://www.biorxiv.org/content/10.1101/2024.12.03.626671v4.full):
- SLC19A3 designed chain complex (PDB **9G5K**): **without crosslinks, RMSD = 6.47 Å with "incorrect designed chain placement"; with one explicit crosslink, RMSD = 0.81 Å.** iPTM **0.26 → 0.82**.
- Control where the unrestrained model was already right: TNF-α/Infliximab (PDB 4G3Y) **0.88 Å → 1.31 Å** with crosslinks — i.e. restraints slightly *degraded* an already-correct prediction.
- The study's framing, as retrieved: "even a **single crosslink** substantially improves prediction accuracy," and the mechanism "appears to go beyond mere chain proximity—it **guides the neural network to correctly orient and position** the interacting partners at the binding interface." **Caveat:** this last sentence came through a summarizing fetch and I cannot confirm it is the authors' own wording rather than the summarizer's gloss; the RMSD/iPTM numbers are the reliable part.
- Stated limitation: the approach "may struggle with residues involved in multiple crosslinks," and the one-crosslink-per-residue rule prevents modeling simultaneous multi-site crosslinks at the same residue.

**Case 2 — AlphaLink2, CRL4/Vpr** — [bioRxiv 2023.06.07.544059v2](https://www.biorxiv.org/content/10.1101/2023.06.07.544059v2.full): AlphaLink "position[ed] the viral Vpr protein inside the experimental density in agreement with the crystal structure," while AlphaFold "places Vpr **incorrectly**."

**Case 3 — resTrain, ligand pose** — [bioRxiv 2026.07.02.736010v1](https://www.biorxiv.org/content/10.64898/2026.07.02.736010v1.full): PDB **8OV7** with **two** 8 Å restraints on ligand heavy atoms: ligand RMSD **1.24 Å vs. 12.6 Å** for unrestrained AF3. Framed as "two restraints on ligand heavy atoms achieve correct pocket positioning, implying orientation emerges from multiple spatial constraints."

**Counter-evidence / explicit limitation on orientation**:
- HADDOCK, by design: the 1/r⁶ effective distance "ensures that the AIRs are **satisfied as soon as any two atoms of the two proteins are in contact**." An ambiguous restraint is therefore indifferent to orientation once contact exists. — [HADDOCK2.2 generate_air_help](https://www.bonvinlab.org/software/haddock2.2/generate_air_help/)
- resTrain reports, for antibody–antigen, that "providing a **single imprecise contact** between chains can be enough to **fully determine the geometry of the interaction**." **Conflict flag:** the same fetched summary also asserted that the method "addresses proximity but not orientation" and that "no explicit statements indicate the method specifies binding mode." These two statements are inconsistent with each other within one summary; I treat the quoted sentence as more reliable than the summarizer's framing, and flag the whole item as needing direct verification of the preprint. — [bioRxiv 2026.07.02.736010v1](https://www.biorxiv.org/content/10.64898/2026.07.02.736010v1.full)
- The classical docking literature treats alternative orientations as a recognized, distinct failure: "Weak or transient interactions… present challenging cases where the **existence of alternative bound orientations** and encounter complexes complicates the binding energy landscape," and restraint *satisfaction fraction* is used as a scoring term "to **identify correct docking orientations**" (pyDockRST). — [Present and future challenges and limitations in protein–protein docking, *Proteins* 2010](https://onlinelibrary.wiley.com/doi/10.1002/prot.22564)
- The Chai-1 restrained-specified site study reports a *secondary* effect that is orientation-adjacent but not orientation: **specified site redundancy dropped >5-fold (0.56 → 0.10)** — restraints changed *which* specified site was sampled across seeds, not the rotation at a fixed specified site. — [bioRxiv 2025.09.17.676770v2](https://www.biorxiv.org/content/10.1101/2025.09.17.676770v2.full)

### Inferences
- **The literature does not separate the two effects the user needs separated.** DockQ mixes Fnat (contact identity), LRMS (ligand RMSD after receptor superposition — orientation-sensitive) and iRMS; every aggregate restraint result is reported in DockQ, and no paper reports a rotation-angle or registry metric. The user's own measurement (12–52° rotation / 156–178° flip at 13–28 Å target-aligned designed chain RMSD, with site and face correct) is **a more orientation-specific readout than anything published for restrained predictors**, which is itself a notable finding: the question is open, not answered negatively.
- The strongest *mechanistic* argument that conditioning can do what a loss cannot is Boltz's `UNSELECTED` class: it supplies a negative, exclusionary prior ("these pairs are the contacts; others are not") that is expressible as an input but not as an additive attractive loss. A flip that preserves site and face is precisely a hypothesis that an exclusionary signal rules out and an attractive signal does not.
- The counter-argument is equally concrete: resTrain's loss-optimized pair bias beat input-conditioned Chai-1 on the same single restraint (72% vs 56%, Q8). So **"conditioning beats loss" is not established**; what the evidence supports is "a restraint that enters the pair representation beats a restraint that only touches the output," and *both* resTrain and conditioning do that, while a distogram-output loss backpropagated to sequence does not.
- All three orientation case studies used restraints that are **hard or near-hard topological/geometric facts** (a covalent crosslinker; a crosslink with FDR; a ligand heavy-atom distance), not soft chain-level pocket hints. The pattern across all evidence: orientation is fixed by *specific named pairs*, never by *ambiguous surface restraints*.

### Gaps
- **No systematic, benchmark-level measurement of restraint conditioning on orientation/binding-mode/registry exists in anything I could find.** No paper reports rotation angle, interface registry, or approach angle as a function of restraints. This is a genuine hole in the literature, and the user's project is positioned to fill it.
- No source compares conditioning vs. loss on the *same* model with the *same* restraints while measuring orientation specifically.

---

## Q8. Does any source report the number of restraints needed, or that too few leave orientation underdetermined?

### Takeaway
Several sources report **restraint-count effects** and they point one way: **1 restraint often suffices to fix the interface; ~2–4 are reported for pose/orientation-level correctness; nobody reports a rigid-body-determinacy analysis.** The single most informative head-to-head: with **one** 8 Å restraint, a loss-optimized pair bias hit 72% success and input-conditioned Chai-1 hit 56%, against 44% unrestrained — so one restraint moves the interface a lot, but leaves a large residual failure rate in both regimes.

### Cited Findings

**resTrain head-to-head on 25 antibody–antigen complexes, one approximate 8 Å restraint between specified site and heavy chain** — [bioRxiv 2026.07.02.736010v1](https://www.biorxiv.org/content/10.64898/2026.07.02.736010v1.full):

| Method | Restraint mechanism | Success (DockQ > 0.23) |
|---|---|---|
| AF2-Multimer, unrestrained | — | 44% (11/25) |
| AF3, 1000 seeds, unrestrained | — | 60% (15/25) |
| **Chai-1, same 1 restraint** | **input conditioning** | **56% (14/25)** |
| **AF2-resTrain, same 1 restraint** | **loss-optimized pair bias** | **72% (18/25)** |

- resTrain mechanism: "a **transductive** approach consisting in the optimisation of a **bias term to the pair representation** in AF2 or AF3-based predictors" — a **Restraint Pair Bias (RPB)** applied across all Evoformer/Pairformer blocks, optimized by gradient descent with network parameters **frozen**, using a masked loss over only the restrained residue pairs. Losses by restraint type: **softmax cross-entropy for exact distances, sigmoid cross-entropy for approximate/maximum distances, KL-divergence for distance distributions.** Converges "in a handful of gradient descent steps (mere minutes of GPU time)."
- Explicit contrast drawn with conditioning: input-conditioning methods "accept external restraints as additional features during inference **without guarantee of compliance**"; resTrain's "loss convergence ensures compliance, versus Chai-1/Protenix treating restraints as **soft**."
- Restraint-count claims: "often **a single pair distance is sufficient**"; resTrain "outperforms Distance-AF (which [states] **several amino acid pair distance restraints are needed**)." Ligand case needed **two** restraints (8OV7: 1.24 Å vs 12.6 Å unrestrained).
- Other resTrain numbers: Cfold conformational sampling (238 protein pairs) — 31 states improved by ΔTM > 0.1 vs. 16 for AFsample2; NMR NOE (ARTINA, 90 structures) — 88% of structures had decreased violations, 25% reduction in aggregated violations, 19% fewer violated restraints.
- **Reliability caveat:** this is a 2026 preprint retrieved through a single summarizing fetch, with an unusual DOI prefix, and the summary contained an internal contradiction on the orientation question (Q7). The numbers above should be verified against the preprint directly before being load-bearing.

**Chai-1 restraint count** — [Chai-1 preprint](https://www.biorxiv.org/content/10.1101/2024.10.10.615955v2.full): baseline 35% → "conditioning on **four** randomly sampled specified-site residues **more than doubles** the DockQ success rate across all quality cutoffs." One specified-site residue at 8 Å already "increased [performance] substantially." High-quality success remained 4–8%. The number of restraints during training followed a **geometric distribution**, i.e. the model was trained predominantly on *few* restraints.

**Other restraint counts**:
- AF3 + explicit crosslinks: **one** crosslink took a designed chain complex from 6.47 Å to 0.81 Å RMSD. — [bioRxiv 2024.12.03.626671v4](https://www.biorxiv.org/content/10.1101/2024.12.03.626671v4.full)
- AlphaLink2 real data: **median 1 crosslink** per PPI across 135 *B. subtilis* dimers was enough to lift median confidence 0.42 → 0.60, with CodY–YppF going 0.25 → 0.81 on a single crosslink. Simulated CASP15 runs used a **median of 33 links including 7 false links** per interaction. — [bioRxiv 2023.06.07.544059v2](https://www.biorxiv.org/content/10.1101/2023.06.07.544059v2.full)
- Chai-1 K-Ras/antibody study: "providing **n=2** randomly selected contact restraints based on experimental ground truth significantly improved the interface DockQ scores." — [bioRxiv 2025.09.16.676163](https://www.biorxiv.org/content/10.1101/2025.09.16.676163.full.pdf)
- Boltz-1 trains on exactly **one** sampled pocket per invocation of the conditioning loop, geometrically subsampled (`binder_pocket_sampling_geometric_p: 0.3`), with **one** randomly chosen token pair per contact-conditioning draw (`featurizerv2.py:766–1017`). — [featurizerv2.py](https://github.com/jwohlwend/boltz/blob/main/src/boltz/data/feature/featurizerv2.py)

### Inferences
- **Rigid-body geometry, labeled as my inference and not found in any source:** a relative placement of two rigid bodies has 6 DOF. One pairwise distance constrains 1; three non-collinear pairwise distances between non-collinear atom triples are needed to fix all 6 up to a reflection, and four to remove the reflection. A single-pair restraint therefore leaves 5 DOF — including the entire rotation about the contact axis and the flip — to be supplied by the model's prior. **This is exactly the user's failure mode: the site is pinned, the rotation is not.** The published "one restraint suffices" results should be read as "one restraint plus a good prior suffices"; where the prior is wrong about rotation, restraint *count and geometric spread* become the active variable, and nobody has measured that.
- Practical consequence: to pin orientation, restraints should be chosen for **geometric spread across the interface** (e.g. three or four reference pairs spanning distinct designable regions and distinct specified-site positions, not clustered in one loop), not merely for count. No source states this, but it follows directly from the DOF argument and is consistent with the ligand case needing two restraints and the Chai specified site result needing four residues.
- The resTrain-vs-Chai-1 comparison is the most directly relevant single result in this entire review, and its answer to the user's open question is **"not necessarily."** With the same one restraint, the *loss-optimized pair-representation bias* beat the *trained input conditioning*. The decisive variable in that comparison is not conditioning-vs-loss but **where the restraint acts**: both act on the pair representation; a distogram-output loss backpropagated through to sequence acts much further downstream.

### Gaps
- **No source analyzes restraint count vs. orientation determinacy.** No paper reports an N-restraints sweep with an orientation metric; the only sweeps are over distance threshold (Chai pocket: 5/6–11/12 Å) and over restraint type.
- No source states a minimum restraint count for pose determination, nor tests restraint *placement/spread* at fixed count.
- No source reports what happens when restraints are mutually consistent with two registries (the flip-degenerate case) — i.e. whether restraints chosen symmetrically about an interface can *fail* to break a flip.

---

## Cross-cutting summary for the report writer

**Taxonomy of restraint mechanisms found, ordered by how deep in the model they act:**

| Mechanism | Systems | Acts on | Trained for it? | Guarantees compliance? |
|---|---|---|---|---|
| Trained pair-representation conditioning | Boltz-2 (`ContactConditioning` → `z_init`), Chai-1 (pocket/contact/docking features), AlphaLink2 (crosslinks → pair rep) | initial pair rep `z` | **Yes** | No — soft prior |
| Optimized pair-representation bias | resTrain (RPB, frozen weights) | pair rep, all blocks | No (optimized per target) | Claimed yes, via loss convergence |
| Topological conditioning (covalent linker) | AF3 `bondedAtomPairs` + crosslinker ligand | input topology | Yes (bonded ligands are in-distribution) | Near-hard |
| Diffusion-time guidance potential | Boltz-2 `ContactPotentital` (`force: true`, `contact_guidance_update=True` by default), experiment-guided AF3 | sampled coordinates | No | No — biases sampling |
| Output-space differentiable loss | the user's distogram restraint; ColabDock | distogram / coordinate output | No | No |
| Classical restraint energy term | HADDOCK AIRs | CNS energy function | n/a | No — "biased towards them" |

**The three findings most likely to matter for the user's rotation/flip problem:**
1. **Pocket-style restraints are orientation-indifferent by construction** — HADDOCK says so explicitly (1/r⁶ d_eff "satisfied as soon as any two atoms are in contact"), and Boltz's `pocket` (whole-chain mask) and Chai's `pocket` (`min` over chain) have the same property. Only named-pair `CONTACT`/`contact`/`docking` restraints carry registry.
2. **Boltz-2's `UNSELECTED` class is a negative, exclusionary prior** (`featurizerv2.py:1018–1023`) — supplying any restraint flips every other pair from "unknown" to "explicitly not a contact." This is the one documented thing conditioning does that an additive attractive loss structurally cannot, and it is exactly the kind of signal that rules out a site-preserving, face-preserving flip. The loss-side analogue is an explicit *repulsive* term on decoy-registry pairs — Boltz even has the primitive for it (`negation_mask`, `potentials.py:248–260`).
3. **Boltz applies contact guidance on a ramp, not uniformly** — `guidance_weight = PiecewiseStepFunction(thresholds=[0.25, 0.75], values=[0.0, 0.5, 1.0])` with `guidance_interval=4` (`potentials.py:758–775`). Zero weight for the first quarter of sampling. A uniformly-weighted restraint loss is doing something the production system deliberately avoids.

**Honest statement of the crux:** no source establishes that conditioning fixes orientation where a loss cannot. The single head-to-head available (resTrain 72% vs. Chai-1 56% on one 8 Å restraint, both against 44% unrestrained) favors the **loss-optimized** route — but both act on the pair representation, which is much earlier in the model than an output-distogram loss. The distinction the evidence actually supports is **depth of action**, not conditioning-vs-loss.
