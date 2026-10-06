# Epitope / binding-site specification and enforcement in diffusion-based and inverse-folding binder design pipelines

**Scope note.** Notes current as of 2026-10. Where a mechanism is read directly from source, the file path and line numbers are given. Local paths refer to clones on this machine; GitHub URLs are given alongside so the report-writer can cite publicly.

**Terminology used consistently below (this is the distinction the user cares about):**
- **CONDITIONING** = an input tensor/feature the generative model *sees* and was *trained* to respond to. No gradient on the design objective; the model has simply learned "when this channel is on, put the interface here."
- **LOSS** = a differentiable term added to the optimisation objective; creates a gradient that pulls the binder toward the site.
- **IN-TRAJECTORY REJECTION** = a mid-generation geometric check that aborts and restarts the trajectory. Not a gradient, not a post-hoc filter.
- **FILTER** = a post-hoc, site-aware acceptance test applied to finished designs.

---

## Q1. RFdiffusion: how are hotspot residues specified, and what do they do mechanistically?

### Takeaway
RFdiffusion hotspots are pure **CONDITIONING**: a one-hot channel appended to the target's 1-D template features in a binder-design-finetuned checkpoint. There is no loss term and no restraint — the only enforcement is that the network was trained on interfaces where that channel was on. Because training masked 80-100% of the true hotspots, the provided hotspots are deliberately a *weak hint* rather than a specification, and the public code contains no site-specificity filter at all.

### Cited Findings
- Parameter name and format: `ppi.hotspot_res=[A30,A33,A34]` — chain letter plus residue number as numbered in the input PDB, comma-separated inside brackets — [RFdiffusion README](https://github.com/RosettaCommons/RFdiffusion)
- Full example invocation: `./scripts/run_inference.py 'contigmap.contigs=[B1-100/0 100-100]' 'ppi.hotspot_res=[A30,A33,A34]' inference.output_prefix=test_outputs/binder_test inference.num_designs=10` — [RFdiffusion README](https://github.com/RosettaCommons/RFdiffusion)
- Training definition of a hotspot: *"In the paper we define a hotspot as a residue on the target protein which is within 10A Cbeta distance of the binder."* — [RFdiffusion README](https://github.com/RosettaCommons/RFdiffusion)
- **Masking fraction (critical):** *"Of all of the hotspots which are identified on the target 0-20% of these hotspots are actually provided to the model and the rest are masked."* — [RFdiffusion README](https://github.com/RosettaCommons/RFdiffusion)
- Recommended count: *"We normally recommend between 3-6 hotspots, you should run a few pilot runs before generating thousands of designs to make sure the number of hotspots you are providing will give results you like."* — [RFdiffusion README](https://github.com/RosettaCommons/RFdiffusion)
- The model is a *separate finetuned checkpoint*, not the base model: RFdiffusion *"is fine-tuned on multi-chain proteins and protein-protein complexes from the PDB, learning to generate a binding protein given the hotspot residues of the target protein"* — [Watson et al., Nature 2023](https://www.nature.com/articles/s41586-023-06415-8)
- Authors' stated efficacy claim: *"Specific 'hotspot' residues can be input to a fine-tuned RFdiffusion model, and with these inputs, binders almost universally target the correct site."* — [RFdiffusion README](https://github.com/RosettaCommons/RFdiffusion)
- The stated *reason* hotspots exist is exactly the failure mode in question — stopping the binder from going to an incidental attractive patch: *"if you crop your target and potentially expose hydrophobic core residues which were buried before the crop, how can you guarantee the binder will go to the intended interface site on the surface of the target, and not target the tantalizing hydrophobic patch you have just artificially created?"* — [RFdiffusion README](https://github.com/RosettaCommons/RFdiffusion)
- Alternative checkpoint for topology diversity, invoked with `inference.ckpt_override_path=models/Complex_beta_ckpt.pt`; authors note it *"generates a greater diversity of topologies, but has not been extensively experimentally validated"* — [RFdiffusion README](https://github.com/RosettaCommons/RFdiffusion)
- **Hotspot conditioning alone is weak enough that a second conditioning channel was added in a follow-up.** β-pairing–targeted RFdiffusion conditions on *which target β-strand the binder should pair with*, and reports **9.2% of β-strand-interface-conditioned designs** passing quality metrics versus **0.98% for RFdiffusion conditioned on target hotspots alone** (computational metrics) — [Improved protein binder design using β-pairing targeted RFdiffusion, Nat Commun 2025](https://www.nature.com/articles/s41467-025-67866-3); preprint [bioRxiv 2024.10.11.617496](https://www.biorxiv.org/content/10.1101/2024.10.11.617496v1.full.pdf)
- Community practice guide on hotspot choice (secondary, not peer-reviewed, but concrete): [adaptyvbio/protein-design-skills hotspot-selection.md](https://github.com/adaptyvbio/protein-design-skills/blob/main/skills/rfdiffusion/references/hotspot-selection.md)

### Inferences
- Because 0-20% of real hotspots are shown during training, RFdiffusion is explicitly trained to expect that the true interface is *larger than and extends beyond* the hotspots supplied. Supplying 3-6 hotspots therefore constrains the interface *centre* only loosely; it cannot pin the interface *extent*. For a pipeline that is seeing a 36-50-residue interface form in the wrong place, this is the relevant lesson: hotspot conditioning is a prior over interface location, not a constraint.
- The 0.98% → 9.2% jump from adding a second, geometrically more specific conditioning channel is the strongest published evidence that hotspot-only site specification is the limiting factor for difficult targets, rather than binder foldability.
- Nothing in the RFdiffusion inference code computes a hotspot-contact metric on the finished design. Site specificity in the stock pipeline is entirely upstream (conditioning) with no downstream verification.

### Gaps
- I could not locate an **exact reported fraction of RFdiffusion designs whose interface overlaps the specified hotspots** (the quantitative version of "almost universally target the correct site"). The README states it qualitatively; the Nature 2023 supplementary may contain a number but I did not retrieve a verbatim figure. Do not invent one.
- No RFdiffusion paper statement found that *quantifies* designs binding an unintended site.

---

## Q2. RFantibody: how is epitope targeting done for antibody/nanobody CDR design, and what is the reported epitope-specificity success rate?

### Takeaway
RFantibody uses the same one-hot hotspot **CONDITIONING** channel (written into `t1d` dimension 23), but with two substantive differences from vanilla RFdiffusion: (a) the hotspot training definition is CDR-specific (mean Cβ distance to the nearest 5 CDR residues < 8 Å) and up to **100%** of hotspots can be shown, making the conditioning much stronger; and (b) the inference script contains an actual **IN-TRAJECTORY REJECTION** mechanism — `terminate_bad_targeting` — that measures hotspot-to-designed-loop Cβ distance mid-diffusion and restarts the trajectory if the design is not heading for the epitope. This is the single most directly transferable mechanism found in this research. The authors still state that the lack of an effective *filter* is the pipeline's main limitation.

### Cited Findings — mechanism (conditioning)
- CLI: `-h, --hotspots` with help text *"Hotspot residues on target, e.g., \"A100,A105,A110\""* — [src/rfantibody/cli/inference.py:42-43](https://github.com/RosettaCommons/RFantibody/blob/main/src/rfantibody/cli/inference.py)
- The CLI translates directly into the RFdiffusion-style config key: `cmd.append(f"ppi.hotspot_res=[{','.join(hotspot_list)}]")` — [src/rfantibody/cli/inference.py:126-128](https://github.com/RosettaCommons/RFantibody/blob/main/src/rfantibody/cli/inference.py)
- Full example command: `rfdiffusion -t target.pdb -f framework.pdb -o designs/ab -n 10 -l "H1:7,H2:6,H3:5-13" -h "B146,B170,B177"` — [RFantibody README](https://github.com/RosettaCommons/RFantibody)
- Hotspots are parsed to a boolean mask over target residues, with a hard assertion on PDB-indexed form: `assert all([i[0].isalpha() for i in hotspot_res]), "Hotspot residues need to be provided in pdb-indexed form. E.g. A100,A103"`; and a warning if none given: `ic("WARNING! No hotspot residues were provided to the model at inference time")` — [src/rfantibody/rfdiffusion/inference/ab_pose.py:90-112](https://github.com/RosettaCommons/RFantibody/blob/main/src/rfantibody/rfdiffusion/inference/ab_pose.py)
- **Where the conditioning physically enters the network:** `features['t1d'][...,22] = self.ab_item.hotspots[None,None]` — [src/rfantibody/rfdiffusion/inference/model_runners.py:530](https://github.com/RosettaCommons/RFantibody/blob/main/src/rfantibody/rfdiffusion/inference/model_runners.py); the dimension is configured as `preprocess.hotspot_dim: 23` (1-indexed) — [scripts/config/inference/antibody.yaml:11](https://github.com/RosettaCommons/RFantibody/blob/main/scripts/config/inference/antibody.yaml); applied in several template-construction branches as `ret_t1d[:,:,hotspot_dim-1] = item.hotspots` with the comment *"Subtract 1 from hotspot dim since hotspot dim is 1 indexed"* — [src/rfantibody/rfdiffusion/inference/ab_util.py:270-272, 326-328, 361-363, 442-444](https://github.com/RosettaCommons/RFantibody/blob/main/src/rfantibody/rfdiffusion/inference/ab_util.py)
- Training definition (CDR-specific, differs from RFdiffusion's 10 Å binder-wide rule): *"a target residue as a hotspot if it has an average Cβ distance to the closest 5 antibody CDR residues of less than 8 Angstroms"* — [RFantibody README](https://github.com/RosettaCommons/RFantibody)
- **Masking fraction (0-100%, vs 0-20% for RFdiffusion):** *"Of all of the hotspots which are identified on the target 0-100% of these hotspots are actually provided to the model and the rest are masked."* — [RFantibody README](https://github.com/RosettaCommons/RFantibody)
- Authors' own sensitivity warning: *"RFantibody is more sensitive to exactly which hotspots are selected than vanilla RFdiffusion is."* — [RFantibody README](https://github.com/RosettaCommons/RFantibody)
- Target chain must be chain `T`, heavy `H`, light `L`, in order H,L,T, with CDR loops declared via PDB remarks (`REMARK PDBinfo-LABEL: 32 H1`) — the HLT format — [RFantibody README](https://github.com/RosettaCommons/RFantibody)
- Also conditioned (separately from hotspots) on an antibody framework template and a docking orientation; the network receives *"a target structure, user-specified 'hotspot' epitope residues, and a template of the antibody framework"*, with the paper calling accurate epitope targeting *"a key advance of this work"* — [Bennett et al., Nature 2025](https://www.nature.com/articles/s41586-025-09721-5); preprint [bioRxiv 2024.03.14.585103v2](https://www.biorxiv.org/content/10.1101/2024.03.14.585103v2.full)
- Code comment indicating fixed-dock and hotspot conditioning are alternative, partly redundant mechanisms: *"Nate thinks it is not necessary to do both fixed dock and hotspot"* — [src/rfantibody/rfdiffusion/inference/ab_util.py:236-240](https://github.com/RosettaCommons/RFantibody/blob/main/src/rfantibody/rfdiffusion/inference/ab_util.py)

### Cited Findings — mechanism (in-trajectory rejection; THE transferable one)
- Config keys and defaults — [scripts/config/inference/antibody.yaml:14-23](https://github.com/RosettaCommons/RFantibody/blob/main/scripts/config/inference/antibody.yaml):
  ```yaml
  antibody:
    terminate_bad_targeting: null
    hotspot_termination_threshold: 10
    hotspot_termination_failures_permitted: 20
  ```
  (`terminate_bad_targeting` is the *diffusion timestep t* at which the check fires; `null` = disabled by default.)
- Implementation, inside the reverse-diffusion loop, wrapped in a `while True:` retry loop — [scripts/rfdiffusion_inference.py:158-178](https://github.com/RosettaCommons/RFantibody/blob/main/scripts/rfdiffusion_inference.py):
  ```python
  Cb = generate_Cbeta(N=px0[:,0], Ca=px0[:,1], C=px0[:,2])
  dist = torch.cdist(Cb[sampler.ab_item.hotspots], Cb[sampler.ab_item.loop_mask]) # [hotspot_L, loop_L]
  mindist = torch.min(dist, dim=1).values   # min distance for each hotspot
  overallmin = torch.min(mindist)           # distance of the closest hotspot to a loop
  print(f'Overall min distance hotspot to designed loop: {overallmin}')
  if conf.antibody.terminate_bad_targeting == t and overallmin > conf.antibody.hotspot_termination_threshold:
      print("Not targeting correctly")
      failed+=1
      if failed>=conf.antibody.hotspot_termination_failures_permitted:
          sys.exit("This set of inputs is not efficiently targeting the hotspots")
      continue
  ```
  So: on the predicted-clean structure `px0` at timestep `t`, compute hotspot-Cβ to designed-CDR-loop-Cβ distances; if the *closest* hotspot is further than **10 Å** from any designed loop residue, discard and restart; after **20** such failures abort the whole job as badly specified.
- The same quantity is also computed and printed for every finished design (overall min and mean of per-hotspot min distances), and hotspots are written into the output PDB B-factor column (`bfacts[sampler.ab_item.hotspots] = 0`) — [scripts/rfdiffusion_inference.py:226-258](https://github.com/RosettaCommons/RFantibody/blob/main/scripts/rfdiffusion_inference.py). This is a ready-made, site-aware per-design metric.
- The filtering stage (RF2) is also made *epitope-aware*: `--hotspot-show-prop` default `0.1`, passed as `inference.hotspot_show_proportion=0.1`; README: *"By default this will run with 10 recycling iterations and with 10% of hotspots provided to the model."* Example of raising it: `rf2 -p antibody.pdb -o predictions/ --hotspot-show-prop 0.5` — [src/rfantibody/cli/inference.py:323-324, 419](https://github.com/RosettaCommons/RFantibody/blob/main/src/rfantibody/cli/inference.py)

### Cited Findings — reported success / specificity
- Minimal in-silico filter thresholds: **RF2 pAE < 10** and **RMSD (design versus RF2 predicted) < 2 Å** — [RFantibody README](https://github.com/RosettaCommons/RFantibody)
- Authors' explicit statement that the filter is the bottleneck: *"The lack of an effective filter is the main limitation of the RFantibody pipeline at the moment."* — [RFantibody README](https://github.com/RosettaCommons/RFantibody)
- Scale guidance: a pilot *"we were able to identify VHH binders from a set of 95 designs"*, but general guidance that *"design campaigns in the 10k range will be required"* — [RFantibody README](https://github.com/RosettaCommons/RFantibody)
- Epitope specificity verified **experimentally, not computationally**, and by competition rather than by a design-stage metric: for TcdB RBD *"binding was completely abolished upon addition of a previously designed ... de novo binder"*; for SARS-CoV-2 RBD *"Binding was to the expected epitope, confirmed by competition"*; cryo-EM showed designs engaged *"the target epitope as designed"* — [bioRxiv 2024.03.14.585103v2](https://www.biorxiv.org/content/10.1101/2024.03.14.585103v2.full)
- Cross-reactivity (a different kind of specificity): for TcdB, *"no binding observed to the highly related (70% sequence homology) Clostridium sordellii lethal toxin L"* — [bioRxiv 2024.03.14.585103v2](https://www.biorxiv.org/content/10.1101/2024.03.14.585103v2.full)
- **No aggregate percentage of designs binding an unintended epitope is reported** in the preprint text retrieved — [bioRxiv 2024.03.14.585103v2](https://www.biorxiv.org/content/10.1101/2024.03.14.585103v2.full)
- Reported affinities for epitope-directed VHHs: RSV site III **1.4 μM**, influenza hemagglutinin **78 nM** (experimentally measured) — [Bennett et al.](https://www.biorxiv.org/content/10.1101/2024.03.14.585103v2.full)
- **Retrospective confidence-metric analysis directly relevant to the user's ipSAE-style selection:** *"AF3 iPTM score is predictive of binding success (AUC = 0.86)"*, yet *"only 9% of our ordered VHH designs have an iPTM score > 0.6"* and *"5 out of the 6 experimentally-confirmed designs pass this threshold"* (>0.85 for scFvs) — [bioRxiv 2024.03.14.585103v2](https://www.biorxiv.org/content/10.1101/2024.03.14.585103v2.full)
- Atomic agreement in the Nature 2025 version: a designed influenza VHH cryo-EM structure at **3.0 Å** matching the model at **1.45 Å backbone RMSD** and **0.8 Å for CDR3** — [RosettaCommons, "RFAntibody creates epitope-specific antibody binders with atomic-level agreement", 24 Nov 2025](https://rosettacommons.org/2025/11/24/rfantibody-creates-epitope-specific-antibody-binders-with-atomic-level-agreement/); [Nature 2025](https://www.nature.com/articles/s41586-025-09721-5)

### Inferences
- `terminate_bad_targeting` is functionally an *epitope-aware early-abort on the denoised prediction*. The analogue for a gradient-based predictor pipeline is: at some fraction of the optimisation, compute min Cβ distance from each specified epitope residue to the designed region and abandon/reseed if the minimum exceeds a threshold (~10 Å is the published value). It costs nothing differentiable and turns a site-agnostic objective into a site-gated search.
- That RFantibody shows up to 100% of hotspots during training, while RFdiffusion shows at most 20%, implies RFantibody's conditioning is a far closer approximation to "this is the whole epitope." The corresponding cost is the authors' own observation that it is *more* sensitive to hotspot choice.
- The authors' pairing of (i) strong conditioning, (ii) in-trajectory geometric rejection, and (iii) an epitope-aware re-prediction filter, plus (iv) their statement that filtering is still the bottleneck, is a direct warning that site-agnostic interface-confidence filtering of the kind the user describes is known to be insufficient even in the best-resourced published pipeline.

### Gaps
- No published numeric "fraction of designs that hit the specified epitope" for RFantibody. The repo gives the *tooling* to measure it (hotspot-to-loop min distance) but no reported distribution.
- The default `terminate_bad_targeting: null` means the mechanism is **off by default**; I found no paper text stating which timestep value was used in the published campaigns. Unknown.

---

## Q3. AlphaProteo: how is the target site specified, and what filters check binding at the intended site?

### Takeaway
AlphaProteo takes the target structure plus **optional** hotspot residues as **CONDITIONING** to a target-structure-conditioned generative model. Crucially, the published filtering pipeline is described only as "a model or procedure that predicts whether a design will bind" — **no site-specificity filter is described**, and verification that binding occurred at the intended epitope was done *after* the wet-lab screen, by interface point mutagenesis.

### Cited Findings
- Specification: *"we input a structure of the 'target' protein and optionally designate 'hotspot' residues representing the target epitope."* Per-target hotspot residue counts are in Table S1 — [Zambaldi et al., arXiv:2409.08022 (HTML v1)](https://arxiv.org/html/2409.08022v1)
- Architecture framing is *"target-structure-conditioned binder design"*, and the generator *"outputs a structure and sequence of a candidate binder for that target"* — i.e. the site enters as model conditioning, not as a post-hoc constraint — [arXiv:2409.08022v1](https://arxiv.org/html/2409.08022v1)
- Filtering is acknowledged but **not numerically specified** in the retrievable text: *"we generate a large number of design candidates and then filter them to a smaller set prior to experimental testing"*, with the filter described only as *"a model or procedure that predicts whether a design will bind"*. No ipTM/pAE/pLDDT/Rosetta thresholds are given — [arXiv:2409.08022v1](https://arxiv.org/html/2409.08022v1)
- **No computational filter for hotspot recovery or epitope overlap is described.** Intended-interface verification was experimental and retrospective: *"To test whether our designs bind their targets via the intended interactions, we measured binding of our top binders after mutating 1-3 residues at the target-binding interface."* — [arXiv:2409.08022v1](https://arxiv.org/html/2409.08022v1)
- **Experimentally validated** per-target success rates (binders / tested) — [arXiv:2409.08022v1](https://arxiv.org/html/2409.08022v1):

  | Target | Success rate | n tested |
  |---|---|---|
  | BHRF1 | 88% | 94 |
  | VEGF-A | 33% | 94 |
  | IL-7RA | 25% | 94 |
  | PD-L1 | 15% | 159 |
  | IL-17A | 14% | 63 |
  | SC2RBD | 12% | 172 |
  | TrkA | 9% | 131 |
  | TNFα | 0% | 54 |
- Headline claim: *"3- to 300-fold better binding affinities and higher experimental success rates than the best existing methods on seven target proteins"*, with designs usable after *"only one round of medium-throughput screening"* — [arXiv:2409.08022](https://arxiv.org/abs/2409.08022)

### Inferences
- AlphaProteo's design of the *verification* step (interface point mutants, post-screen) is itself evidence that even DeepMind did not have a trustworthy computational site-specificity check; they resolved the question by mutagenesis. That is a strong external data point for the user's situation: site-agnostic confidence metrics were not treated as sufficient evidence of epitope engagement by the authors either.
- Because hotspots are *optional*, the model must contain a learned prior over "where binders go on this fold." A pipeline without that prior (a frozen structure predictor with a sequence-gradient objective) has no equivalent, which is consistent with the user seeing a plausible-but-displaced interface.

### Gaps
- The exact in-silico filter metrics and numeric thresholds are **not disclosed** in the accessible arXiv text. Table S1 (hotspot counts per target) and the filter details are in supplementary material I could not retrieve verbatim. Mark as undisclosed rather than guessing.
- Whether any AlphaProteo binder was found to bind at an unintended site is not reported.

---

## Q4. EvoBind / EvoBind2: how exactly is the site constrained?

### Takeaway
EvoBind is the one pipeline in this set where the epitope is enforced **purely by a LOSS**, and the loss is remarkably simple: the mean over peptide atoms of the distance to the *nearest atom belonging to a user-listed receptor interface residue*, divided by the mean peptide pLDDT. There is no conditioning and no flat bottom — the term keeps pulling until it is minimised. EvoBind2's headline feature is the *opposite* of what the user needs: it makes target residues optional (default = all residues), which degenerates the distance term into "bind anywhere."

### Cited Findings — the site-specifying flag
- Flag: `flags.DEFINE_list('receptor_if_residues', None, 'Comma separated list of receptor interface residues (start at zero).')` — [src/mc_design.py:63-64](https://github.com/patrickbryant1/EvoBind/blob/master/src/mc_design.py)
- Example value from the shipped driver script: `RECEPTORIFRES="4,5,8,11,12,45,47,54,55,57,58,59,65,66,72,74,81,83,102,104,105,106,107,108,109,110,111,112"`, with the comment *"###Receptor interface residues - provide with --receptor_if_residues=$RECEPTORIFRES if using"* — [design_local.sh](https://github.com/patrickbryant1/EvoBind/blob/master/design_local.sh)
- Other relevant flags / defaults from the same driver: `--peptide_length=10`, `--num_iterations=1000`, `--max_recycles=8`, `--model_names=model_1`, `--predict_only=False`, optional `--cyclic_offset=1` — [design_local.sh](https://github.com/patrickbryant1/EvoBind/blob/master/design_local.sh); flags at [src/mc_design.py:58-101](https://github.com/patrickbryant1/EvoBind/blob/master/src/mc_design.py)
- **Indexing inconsistency to flag:** the flag help says *"(start at zero)"* while an inline comment in the loss code says *"#Start at 1 - same for receptor_if_residues"* — [src/mc_design.py:64 vs :240](https://github.com/patrickbryant1/EvoBind/blob/master/src/mc_design.py)

### Cited Findings — the loss, verbatim
- The epitope term, computed on all-atom coordinates of the AF2 prediction — [src/mc_design.py ~lines 243-263](https://github.com/patrickbryant1/EvoBind/blob/master/src/mc_design.py):
  ```python
  #Get atoms belonging to if res for the receptor
  receptor_if_pos = []
  for ifr in receptor_if_residues:
      receptor_if_pos.extend([*np.argwhere(receptor_resno==ifr)])
  receptor_if_pos = np.array(receptor_if_pos)[:,0]

  #Calc 2-norm - distance between peptide and interface
  mat = np.append(peptide_coords, receptor_coords[receptor_if_pos], axis=0)
  a_min_b = mat[:,np.newaxis,:] - mat[np.newaxis,:,:]
  dists = np.sqrt(np.sum(a_min_b.T ** 2, axis=0)).T
  l1 = len(peptide_coords)
  contact_dists = dists[:l1,l1:]   # first dim = peptide atoms, second = receptor if-atoms

  #Get the closest atom-atom distances across the receptor interface residues.
  closest_dists_peptide = contact_dists[np.arange(contact_dists.shape[0]), np.argmin(contact_dists,axis=1)]
  peptide_plDDT = plddt[-peptide_length:]
  return closest_dists_peptide.mean(), peptide_plDDT.mean(), unrelaxed_protein
  ```
- The combined objective and the acceptance rule (greedy Monte-Carlo, one random mutation per step) — [src/mc_design.py:385-386 and 400-416](https://github.com/patrickbryant1/EvoBind/blob/master/src/mc_design.py):
  ```python
  loss = if_dist_peptide*1/plddt
  if loss<min(sequence_scores['loss']):
      peptide_sequence = new_sequence
  ```
  i.e. **loss = (mean peptide-atom→nearest-epitope-atom distance) / (mean peptide pLDDT)**. No weights, no schedule, no flat bottom, no clash term.
- Algorithm docstring: *"1. Initialize an array with ra[n]domly distributed sequence probabilities ... 4. Mutate the sequence. 5. Score the current peptide based on the distance from the peptide atoms to the interface and the peptide plDDT (loss). 6. Accept the new sequence as a starting point if the loss is lower. 7. Return to step 4."* — [src/mc_design.py:316-328](https://github.com/patrickbryant1/EvoBind/blob/master/src/mc_design.py)
- Original method paper — [EvoBind: in silico directed evolution of peptide binders with AlphaFold, bioRxiv 2022.07.23.501214](https://www.biorxiv.org/content/10.1101/2022.07.23.501214v1.full): designs binders *"towards a target interface by updating the sequence based on several loss terms that capture the interaction potential and the overall ability of AlphaFold to predict the structure"*, using *"only the sequence of the target protein and specified interface residues"*.

### Cited Findings — EvoBind2 (the untargeted variant)
- README framing: *"In silico directed evolution of peptide binders based only on a protein target sequence. It is not necessary to specify any target residues within the protein sequence or the length of the binder (although this is possible)."* Optional arguments listed as *"Peptide length - default=10"* and *"Target residues within the receptor sequence - default=all"* — [EvoBind README](https://github.com/patrickbryant1/EvoBind)
- Stated rationale for *removing* the site constraint: *"this lack of constraint removes any procedural bias, as preconceived notions about what constitutes a good binding structure, sequence, or site may be incorrect"* — [Design of linear and cyclic peptide binders from protein sequence information, Commun Chem 2025](https://www.nature.com/articles/s42004-025-01601-3)
- EvoBind2 objective: *"designs over a relaxed sequence-structure space that is searched by mutating one residue randomly at a time to minimise a loss function based on peptide plDDT ... and the average distance to target residues"*; AlphaFold-Multimer is used *"as an additional check to avoid adversarial designs"* — [Commun Chem 2025](https://www.nature.com/articles/s42004-025-01601-3); preprint [bioRxiv 2024.06.20.599739v2](https://www.biorxiv.org/content/10.1101/2024.06.20.599739v2.full)
- The AFM adversarial-check scoring code lives in a separate module: `src/AFM_eval/afm_evo_loss_calc.py` — [EvoBind repo tree](https://github.com/patrickbryant1/EvoBind)

### Inferences
- EvoBind's loss is exactly the kind of term absent from the user's pipeline, and it is one line: `mean_over_binder_atoms(min_dist_to_epitope_atoms)`. Because the reduction is `min` over epitope atoms and `mean` over *binder* atoms, the term rewards the *whole binder* approaching the epitope, not just one contact — which also means it will squash a large binder onto a small epitope. For a nanobody with CDRs it would be more appropriate to take the mean over *epitope* residues of the min distance to *CDR* atoms (which is precisely the RFantibody reduction order: min over loop residues per hotspot, then min/mean over hotspots).
- Dividing by pLDDT rather than adding a weighted term means the two objectives are coupled multiplicatively; there is no tunable weight. This is unusually crude and the user should not copy the form, only the distance term.
- EvoBind2's "default = all receptor residues" setting is a direct, published instance of the degenerate case the user is in: the distance objective is satisfiable at any surface site, so the design goes wherever AF2's energy landscape is most favourable.

### Gaps
- No reported quantitative rate of "designed peptide bound the specified epitope" for either EvoBind version; the papers report predicted-structure and some experimental binding, not site-recovery fractions.
- I did not retrieve EvoBind2's exact loss implementation (the repo's `mc_design.py` shown above is v1-style); the Commun Chem description above is paper text, not verified source.

---

## Q5. Pipelines using explicit inter-chain distance / contact restraints: what form, and what weights?

### Takeaway
Three concrete, implementable restraint forms exist in released code: (1) **ColabDesign/AfDesign's `hotspot` + `i_con` loss**, a soft flat-bottom contact term built from the AF2 distogram with a soft top-k reduction, where the hotspot flag *swaps which side of the interface the per-residue reduction runs over*; (2) **BindCraft**, which wraps exactly that loss with published weights and — uniquely — ships an actual **site-specificity FILTER** (`Hotspot_RMSD`, threshold 6 Å); and (3) **Boltz-2 / BoltzGen conditioning constraints** (`pocket`/`contact` with `max_distance`, and BoltzGen's binding-site channels including an explicit *not*-binding channel).

### Cited Findings — ColabDesign / AfDesign (closest analogue to the user's gradient pipeline)
Verified in the local clone at `/home/yfeng17/Proteina-Complexa/community_models/colabdesign/`; upstream project is [sokrypton/ColabDesign](https://github.com/sokrypton/ColabDesign).
- Hotspots are an *option*, consumed at input prep: signature `..., hotspot=None, ignore_missing=True, **kwargs)` with docstring *"-hotspot = define position/hotspots on target"*, and stored as `self.opt["hotspot"] = prep_pos(hotspot, **self._pdb["idx"])["pos"]` — `/home/yfeng17/Proteina-Complexa/community_models/colabdesign/colabdesign/af/prep.py:189, 197, 228-230`
- **The mechanistically important part — the hotspot flag changes the reduction axis of the interface contact loss** — `.../colabdesign/af/loss.py:38-47`:
  ```python
  binder_id = zeros.at[-bL:].set(mask[-bL:])
  if "hotspot" in opt:
    target_id = zeros.at[opt["hotspot"]].set(mask[opt["hotspot"]])
    i_con_loss = get_con_loss(inputs, outputs, opt["i_con"], mask_1d=target_id, mask_1b=binder_id)
  else:
    target_id = zeros.at[:tL].set(mask[:tL])
    i_con_loss = get_con_loss(inputs, outputs, opt["i_con"], mask_1d=binder_id, mask_1b=target_id)
  ```
  Without hotspots `mask_1d=binder_id`: *every binder residue* must contact *something* on the target → site-agnostic. With hotspots `mask_1d=target_id`: *every hotspot residue* must contact *something* on the binder → site-specific. Same loss function, opposite quantifier.
- The contact loss itself (soft top-k over distogram contact probability) — `.../colabdesign/af/loss.py:341-377`:
  ```python
  def get_con_loss(inputs, outputs, con_opt, mask_1d=None, mask_1b=None, mask_2d=None):
    def min_k(x, k=1, mask=None):
      y = jnp.sort(x if mask is None else jnp.where(mask,x,jnp.nan))
      k_mask = jnp.logical_and(jnp.arange(y.shape[-1]) < k, jnp.isnan(y) == False)
      return jnp.where(k_mask,y,0).sum(-1) / (k_mask.sum(-1) + 1e-8)
    ...
    p = _get_con_loss(dgram, dgram_bins, cutoff=con_opt["cutoff"], binary=con_opt["binary"])
    if "seqsep" in con_opt: m = jnp.abs(offset) >= con_opt["seqsep"]
    else:                   m = jnp.ones_like(offset)
    ...
    p = min_k(p, con_opt["num"], m)
    return min_k(p, con_opt["num_pos"], mask_1d)
  ```
- The flat-bottom/soft-step kernel — `.../colabdesign/af/loss.py:379-388`:
  ```python
  def _get_con_loss(dgram, dgram_bins, cutoff=None, binary=True):
    '''dgram to contacts'''
    if cutoff is None: cutoff = dgram_bins[-1]
    bins = dgram_bins < cutoff
    px = jax.nn.softmax(dgram)
    px_ = jax.nn.softmax(dgram - 1e7 * (1-bins))
    con_loss_cat_ent = -(px_ * jax.nn.log_softmax(dgram)).sum(-1)
    con_loss_bin_ent = -jnp.log((bins * px + 1e-8).sum(-1))
    return jnp.where(binary, con_loss_bin_ent, con_loss_cat_ent)
  ```
  `binary=True` gives **−log P(distance < cutoff)** — a genuine soft flat-bottom: no gradient preference among distances below the cutoff. `binary=False` gives a cross-entropy to the distogram masked (truncated) at the cutoff, which still rewards getting closer within the cutoff.
- **Default parameters** — `.../colabdesign/af/model.py:51-52`:
  ```python
  "con":   {"num":2, "cutoff":14.0,    "binary":False, "seqsep":9, "num_pos":float("inf")},
  "i_con": {"num":1, "cutoff":21.6875, "binary":False,             "num_pos":float("inf")},
  ```
  So the stock interface term asks for **1** contact per selected residue at a **21.6875 Å** Cβ cutoff (an AF2 distogram bin edge), categorical form, over **all** selected positions (`num_pos=inf`).
- Contact maps for monitoring are built from the same cutoffs: `"cmap": get_contact_map(outputs, opt["con"]["cutoff"])`, `"i_cmap": get_contact_map(outputs, opt["i_con"]["cutoff"])`; `get_contact_map(outputs, dist=8.0)` default — `.../colabdesign/af/model.py:212-213` and `.../af/loss.py:233`
- The loss keys available for weighting include `"con"` and `"i_con"` alongside `"pae","i_pae","plddt","exp_res","helix","seq_ent","mlm"` — `.../colabdesign/af/design.py:234`
- *Fork note:* this local clone additionally defines `get_ipsae_loss` / `get_min_ipsae_loss` with `pae_cutoff=15.0` for AF2 and `10.0` for AF3/Boltz (`.../af/loss.py:293-327`, comment *"pae cutoff 15 for af2, 10 for af3 and boltz"*). These ipSAE terms are **not** part of the canonical upstream ColabDesign loss set as far as I verified and should be treated as a fork addition. Note these are also site-agnostic by construction (`mask_1d=binder_id, mask_1b=target_id`), i.e. the same blind spot the user has.

### Cited Findings — BindCraft (AF2 hallucination + MPNN; has a real site filter)
- Target site is specified in the target JSON: `"target_hotspot_residues": "56"` — [settings_target/PDL1.json](https://github.com/martinpacesa/BindCraft/blob/main/settings_target/PDL1.json)
- Accepted formats, and the explicit "or let it choose" fallback: hotspots *"can be either defined individually (\"23,25,27,29,30\"), as residue ranges (\"23,25,27-30,35-45\"), or left empty to let the pipeline determine an optimal binding site"*; *"If no hotspots are defined, the pipeline will select an optimal binding mode based on multiple design criteria."* — [BindCraft wiki](https://github.com/martinpacesa/BindCraft/wiki/De-novo-binder-design-with-BindCraft)
- **The hotspots are a LOSS, not conditioning** — they are handed straight to ColabDesign's `hotspot` option: `af_model.prep_inputs(pdb_filename=starting_pdb, chain=chain, binder_len=length, hotspot=target_hotspot_residues, seed=seed, rm_aa=advanced_settings["omit_AAs"], ...)`, with `if target_hotspot_residues == "": target_hotspot_residues = None` immediately before — [functions/colabdesign_utils.py:21-36](https://github.com/martinpacesa/BindCraft/blob/main/functions/colabdesign_utils.py)
- **Published loss weights and contact parameters** — [settings_advanced/default_4stage_multimer.json](https://github.com/martinpacesa/BindCraft/blob/main/settings_advanced/default_4stage_multimer.json):
  ```json
  "weights_plddt": 0.1, "weights_pae_intra": 0.4, "weights_pae_inter": 0.1,
  "weights_con_intra": 1.0, "weights_con_inter": 1.0,
  "intra_contact_distance": 14.0, "inter_contact_distance": 20.0,
  "intra_contact_number": 2,      "inter_contact_number": 2,
  "weights_helicity": -0.3, "use_i_ptm_loss": true, "weights_iptm": 0.05,
  "use_rg_loss": true, "weights_rg": 0.3,
  "use_termini_distance_loss": false, "weights_termini_loss": 0.1
  ```
  Note `inter_contact_distance: 20.0` / `inter_contact_number: 2` map onto ColabDesign's `i_con` `cutoff`/`num`, and the interface contact term carries weight **1.0** — ten times `weights_pae_inter` (0.1) and twenty times `weights_iptm` (0.05). The *geometric* site term dominates the *confidence* terms.
- Optimisation schedule: `"soft_iterations": 75, "temporary_iterations": 45, "hard_iterations": 5, "greedy_iterations": 15, "greedy_percentage": 1`; MPNN stage `"num_seqs": 20, "max_mpnn_sequences": 2, "sampling_temp": 0.1, "model_path": "v_48_020", "mpnn_weights": "soluble", "mpnn_fix_interface": true`; rejection monitoring `"enable_rejection_check": true, "acceptance_rate": 0.01, "start_monitoring": 600` — [default_4stage_multimer.json](https://github.com/martinpacesa/BindCraft/blob/main/settings_advanced/default_4stage_multimer.json)
- **The site-specificity FILTER.** `settings_filters/default_filters.json` includes `Average_Hotspot_RMSD` with `{"threshold": 6, "higher": false}`, i.e. designs are rejected if the hotspot RMSD exceeds **6 Å** — [settings_filters/default_filters.json](https://github.com/martinpacesa/BindCraft/blob/main/settings_filters/default_filters.json). It is reported per design as `'Hotspot_RMSD': rmsd_site` and appears in the output columns alongside `'Target_RMSD'` — [bindcraft.py:296, 348](https://github.com/martinpacesa/BindCraft/blob/main/bindcraft.py)
- The RMSD helper is explicitly superposition-free: `def unaligned_rmsd(reference_pdb, align_pdb, reference_chain_id, align_chain_id)` builds single-chain subposes and applies `RMSDMetric()` with `set_comparison_pose(reference_chain_pose)` and no alignment step — [functions/pyrosetta_utils.py:176-206](https://github.com/martinpacesa/BindCraft/blob/main/functions/pyrosetta_utils.py)
- Interface residues are enumerated by a 4 Å heavy-atom criterion: `def hotspot_residues(trajectory_pdb, binder_chain="B", atom_distance_cutoff=4.0)` — [functions/biopython_utils.py:138](https://github.com/martinpacesa/BindCraft/blob/main/functions/biopython_utils.py); used for the reported `InterfaceAAs` and by `score_interface` at [pyrosetta_utils.py:45](https://github.com/martinpacesa/BindCraft/blob/main/functions/pyrosetta_utils.py)
- Full default filter set (all PyRosetta/AF2 metrics, thresholds; `higher` = must exceed) — [settings_filters/default_filters.json](https://github.com/martinpacesa/BindCraft/blob/main/settings_filters/default_filters.json): `Average_pLDDT > 0.8`, `Average_pTM > 0.55`, `Average_i_pTM > 0.5`, `Average_i_pAE < 0.35`, `Average_Binder_Energy_Score < 0`, `Average_Surface_Hydrophobicity < 0.35`, `Average_ShapeComplementarity > 0.6` (per-model 0.55), `Average_dG < 0`, `Average_dSASA > 1`, `Average_n_InterfaceResidues > 7`, `Average_n_InterfaceHbonds > 3`, `Average_n_InterfaceUnsatHbonds < 4`, `Average_Binder_Loop% < 90`, `Average_Hotspot_RMSD < 6`
- Alternative filter sets shipped: `relaxed_filters.json`, `no_filters.json`, `peptide_filters.json`, `peptide_relaxed_filters.json`; advanced-setting variants include `*_hardtarget.json` and `betasheet_*` families — [BindCraft repo tree](https://github.com/martinpacesa/BindCraft)
- Paper: [One-shot design of functional protein binders with BindCraft, Nature 2025](https://www.nature.com/articles/s41586-025-09429-6) (experimentally validated across multiple target classes)

### Cited Findings — Boltz-2 / BoltzGen conditioning constraints (current generation, 2025-2026)
- Boltz-2 input YAML supports explicit interface restraints as *model conditioning* — verified at `/home/yfeng17/boltz/src/boltz/data/parse/schema.py:972-985` ([jwohlwend/boltz](https://github.com/jwohlwend/boltz)):
  ```yaml
  - pocket:
      binder: E
      contacts: [[B, 1], [B, 2]]
      max_distance: 6
  - contact:
      token1: ...
      token2: ...
      max_distance: 6
  ```
- Default `max_distance = 6.0` Å (`max_distance = constraint["pocket"].get("max_distance", 6.0)`), and in Boltz-1 only a single pocket binder and only the 6.0 Å value were supported (`"Only one pocket binders is supported in Boltz-1!"`) — `/home/yfeng17/boltz/src/boltz/data/parse/schema.py:1545-1567`
- **BoltzGen** conditions on binding sites with a dedicated selector, enabled at inference: `ProteinSelector(design_neighborhood_sizes=[2,4,...,18], substructure_neighborhood_sizes=[2,4,6,8,10,12,24], structure_condition_prob=1.0, distance_noise_std=1, run_selection=True, specify_binding_sites=True, ss_condition_prob=0.1, select_all=False, chain_reindexing=False)` — `/home/yfeng17/boltzgen/src/boltzgen/task/predict/data_from_yaml.py:165-175`
- BoltzGen's binding-site definition cutoffs and **training task mixture, including an explicit negative ("do not bind here") channel** — `/home/yfeng17/boltzgen/src/boltzgen/data/select/protein.py:58-59, 115-122`:
  ```python
  binding_token_cutoff: float = 15,
  binding_atom_cutoff: float = 5,
  ...
  binding_site_probs = {
      "specify_binding": 0.15,
      "specify_not_binding": 0.075,
      "specify_binding_not_binding": 0.075,
      "specify_none": 0.7,
  }
  ```
  i.e. 22.5% of training samples carry positive binding-site conditioning, 15% carry negative conditioning, 70% carry none.
- A 2025 methods paper explicitly reframing structure predictors as energy-based models for binder design (relevant to gradient-through-predictor pipelines): [BindEnergyCraft, arXiv:2505.21241](https://arxiv.org/pdf/2505.21241)

### Cited Findings — cross-pipeline summary (mechanism class per pipeline)
| Pipeline | Site input name | Mechanism class | Key numbers |
|---|---|---|---|
| RFdiffusion | `ppi.hotspot_res=[A30,A33,A34]` | CONDITIONING only | 10 Å Cβ hotspot definition; 0-20% shown in training; 3-6 recommended — [README](https://github.com/RosettaCommons/RFdiffusion) |
| β-pairing RFdiffusion | target β-strand | CONDITIONING (second channel) | 9.2% vs 0.98% in-silico success — [Nat Commun 2025](https://www.nature.com/articles/s41467-025-67866-3) |
| RFantibody | `-h "B146,B170,B177"` → `ppi.hotspot_res` | CONDITIONING (`t1d[...,22]`) + IN-TRAJECTORY REJECTION | 8 Å mean-to-nearest-5-CDR definition; 0-100% shown; threshold 10 Å, 20 failures permitted — [antibody.yaml](https://github.com/RosettaCommons/RFantibody/blob/main/scripts/config/inference/antibody.yaml), [rfdiffusion_inference.py](https://github.com/RosettaCommons/RFantibody/blob/main/scripts/rfdiffusion_inference.py) |
| AlphaProteo | optional "hotspot residues" | CONDITIONING only; no site filter described | per-target hotspot counts in Table S1 — [arXiv:2409.08022v1](https://arxiv.org/html/2409.08022v1) |
| EvoBind v1 | `--receptor_if_residues` | LOSS only | `loss = mean_atom_min_dist_to_epitope / mean_peptide_pLDDT`; 1000 iterations — [mc_design.py](https://github.com/patrickbryant1/EvoBind/blob/master/src/mc_design.py) |
| EvoBind2 | optional, default = all residues | LOSS, degenerate when untargeted | — [Commun Chem 2025](https://www.nature.com/articles/s42004-025-01601-3) |
| ColabDesign/AfDesign | `prep_inputs(hotspot=...)` | LOSS (`i_con`, flat-bottom distogram contact) | `i_con`: num=1, cutoff=21.6875 Å, binary=False — `af/model.py:52` |
| BindCraft | `target_hotspot_residues` | LOSS (via ColabDesign) + **site FILTER** | `weights_con_inter: 1.0`, `inter_contact_distance: 20.0`, `inter_contact_number: 2`; `Average_Hotspot_RMSD < 6` — [advanced](https://github.com/martinpacesa/BindCraft/blob/main/settings_advanced/default_4stage_multimer.json), [filters](https://github.com/martinpacesa/BindCraft/blob/main/settings_filters/default_filters.json) |
| Boltz-2 | `pocket:` / `contact:` YAML constraints | CONDITIONING (constraint features) | `max_distance` default 6.0 Å — `boltz/data/parse/schema.py:972-985,1557` |
| BoltzGen | `specify_binding_sites=True` | CONDITIONING (incl. negative channel) | token cutoff 15 Å, atom cutoff 5 Å; 15%/7.5%/7.5%/70% task mix — `boltzgen/data/select/protein.py:58,115-122` |

### Inferences
- Across every pipeline examined, the restraint form that appears repeatedly is **a soft flat-bottom on min-distance, reduced per *epitope* residue rather than per binder residue.** RFantibody takes min over CDR-loop residues for each hotspot; ColabDesign-with-hotspot takes `min_k` over binder residues with `mask_1d = hotspot mask`. Both answer "is every specified epitope residue in contact with the designed region?" — which is precisely what a site-agnostic interface confidence score cannot answer.
- BindCraft is the existence proof that a cheap post-hoc site filter is practical: a single superposition-free RMSD of the target chain between the starting pose and the design, with a 6 Å gate, alongside a 4 Å heavy-atom interface-residue enumeration. For the user's situation, the equivalent is trivially computable: enumerate interface residues at 4 Å (or Cβ 8-10 Å) and require ≥ some overlap with the intended epitope set, or require the centroid displacement to be under a few Å.
- The weight hierarchy in BindCraft (`weights_con_inter = 1.0` vs `weights_iptm = 0.05`, `weights_pae_inter = 0.1`) is a published statement that the geometric site term should dominate the confidence terms by roughly an order of magnitude during optimisation. A pipeline that optimises confidence alone has the ratio set to zero.
- BoltzGen's `specify_not_binding` channel is a mechanism absent from all the earlier pipelines and is worth noting: negative site specification ("do not bind here") is trainable and may be more robust than positive specification when the decoy site is known, as it is in the user's case.

### Gaps
- I verified the `Average_Hotspot_RMSD` filter name, threshold (6 Å) and that it is populated from a variable `rmsd_site` at [bindcraft.py:296](https://github.com/martinpacesa/BindCraft/blob/main/bindcraft.py), and I verified the `unaligned_rmsd` helper. **I did not capture the exact call site that computes `rmsd_site`**, so the precise chain arguments and whether an alignment on the binder precedes it are unconfirmed. The report should state the filter exists with a 6 Å threshold and not over-specify its geometry.
- No weight *schedules* (ramps/annealing) on any interface restraint were found in the configs examined; BindCraft's schedule is on the sequence relaxation (soft/temporary/hard/greedy iterations), not on the contact weight.
- BoltzGen/Boltz figures above are read from installed source on this machine; I did not retrieve a corresponding paper passage to cite for the default probabilities and cutoffs.

---

## Q6. Does any pipeline report the specific failure of binding at an unintended site, and how was it measured?

### Takeaway
No paper in this set reports an aggregate rate of designs binding an unintended epitope. The failure mode is acknowledged *as the motivation* for hotspot conditioning (RFdiffusion README), is *operationalised as a geometric abort criterion* in RFantibody's code, and is *resolved empirically* in AlphaProteo by post-screen interface mutagenesis — but it is never quantified. Site specificity is measured, when it is measured at all, by wet-lab competition, mutational escape, or cryo-EM.

### Cited Findings
- **Acknowledged as the motivation, not quantified.** RFdiffusion's documentation frames the whole hotspot feature around the risk of the binder going to *"the tantalizing hydrophobic patch you have just artificially created"* rather than the intended site — [RFdiffusion README](https://github.com/RosettaCommons/RFdiffusion)
- **Operationalised as a geometric criterion.** RFantibody's code prints `Overall min distance hotspot to designed loop` every diffusion step and declares `"Not targeting correctly"` when the closest hotspot is further than `hotspot_termination_threshold` (10 Å) from any designed loop residue, restarting the trajectory and aborting the job after 20 failures with `"This set of inputs is not efficiently targeting the hotspots"` — [scripts/rfdiffusion_inference.py:158-178](https://github.com/RosettaCommons/RFantibody/blob/main/scripts/rfdiffusion_inference.py). The existence of a dedicated failure counter implies the authors observed this failing routinely for some inputs.
- **Measured experimentally by competition.** RFantibody: TcdB RBD designs — *"binding was completely abolished upon addition of a previously designed ... de novo binder"*; SARS-CoV-2 RBD — *"Binding was to the expected epitope, confirmed by competition"* — [bioRxiv 2024.03.14.585103v2](https://www.biorxiv.org/content/10.1101/2024.03.14.585103v2.full)
- **Measured experimentally by interface mutagenesis.** AlphaProteo: *"To test whether our designs bind their targets via the intended interactions, we measured binding of our top binders after mutating 1-3 residues at the target-binding interface."* — [arXiv:2409.08022v1](https://arxiv.org/html/2409.08022v1)
- **Measured by cryo-EM.** RFantibody reports cryo-EM confirming designs engaged *"the target epitope as designed"* — [bioRxiv 2024.03.14.585103v2](https://www.biorxiv.org/content/10.1101/2024.03.14.585103v2.full); Nature 2025 reports a designed influenza VHH cryo-EM structure at 3.0 Å matching the model at 1.45 Å backbone / 0.8 Å CDR3 RMSD — [RosettaCommons, 24 Nov 2025](https://rosettacommons.org/2025/11/24/rfantibody-creates-epitope-specific-antibody-binders-with-atomic-level-agreement/)
- **The confidence-metric blind spot, stated by the authors.** RFantibody: *"The lack of an effective filter is the main limitation of the RFantibody pipeline at the moment."* Thresholds in use are site-agnostic (RF2 pAE < 10; design-vs-prediction RMSD < 2 Å) — [RFantibody README](https://github.com/RosettaCommons/RFantibody)
- **Confidence metrics are only moderately informative even in aggregate.** AF3 ipTM gave AUC 0.86 for binding success on RFantibody designs, but only 9% of ordered VHH designs reached ipTM > 0.6 — [bioRxiv 2024.03.14.585103v2](https://www.biorxiv.org/content/10.1101/2024.03.14.585103v2.full)
- **Low-quality secondary claim, flagged as such.** A commercial-adjacent Medium commentary asserts that *"RFdiffusion's one published cryo-EM structure showed antibody binding in a completely different orientation than predicted,"* contrasting it with Chai-2 — [Medium, "I Tried to Poke Holes in Chai-2's Antibody Design Paper"](https://medium.com/@enginyapici/i-tried-to-poke-holes-in-chai-2s-antibody-design-paper-here-s-what-i-found-7e51f5581c7d). This is an opinion blog with a competitive framing and should **not** be cited as a primary finding; it is noted only because it is the one place I found an explicit claim of an experimentally observed binding-pose mismatch for a designed antibody. The peer-reviewed Nature 2025 version reports the opposite for its cryo-EM structure (1.45 Å backbone agreement).
- BindCraft reports specificity in the cross-reactivity sense rather than the epitope-location sense: *"no off-target binding observed even at 10 μM binder concentration to other allergens"*, with cryo-EM validation — [Nature 2025](https://www.nature.com/articles/s41586-025-09429-6)

### Inferences
- The absence of a published "fraction of designs that bound the wrong site" number is itself the finding: the field's standard reporting unit is *binders obtained per designs tested*, which silently conflates site-correct and site-incorrect binding. The user's observed failure (correct fold, large interface, 14-27 Å displaced, high ipSAE) is therefore not anomalous or well-characterised in the literature — it is the failure mode that published pipelines prevent at the *conditioning/restraint* stage and verify only in the wet lab.
- Every published remedy sits at one of two points, and neither is a confidence metric: **before/during generation** (conditioning channel, contact loss, mid-trajectory abort) or **after the experiment** (competition, mutagenesis, cryo-EM). The only in-silico post-hoc site check found anywhere in released code is BindCraft's `Average_Hotspot_RMSD < 6`.
- The thresholds that recur — 4 Å heavy-atom (BindCraft interface enumeration), 5 Å atom / 15 Å token (BoltzGen), 6 Å (Boltz-2 pocket `max_distance`; BindCraft hotspot RMSD), 8 Å Cβ (RFantibody hotspot training definition), 10 Å Cβ (RFdiffusion hotspot definition; RFantibody abort threshold) — give a defensible range for an epitope-contact gate. A 14-27 Å displacement is far outside all of them, so even the loosest published criterion would reject the user's designs.

### Gaps
- No pipeline publishes a site-recovery rate (e.g. "X% of designs had ≥50% epitope overlap"). I found no reliable source for such a statistic for any of RFdiffusion, RFantibody, AlphaProteo, EvoBind, or BindCraft.
- AlphaProteo's supplementary (Table S1 hotspot counts, filter thresholds) was not retrievable; the published Nature-family version, if any, was not located within the search budget.
- I did not find any pipeline that applies an *explicit differentiable* flat-bottom restraint to a frozen AF3-class predictor's gradients w.r.t. sequence (the user's exact setting). The closest published analogues are ColabDesign's distogram `i_con` with a hotspot mask (AF2) and BindCraft's use of it; whether the same term is well-behaved through an AF3-like diffusion head is not addressed in any source I found.
