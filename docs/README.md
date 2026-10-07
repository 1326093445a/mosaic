# P17 → JN.1 documentation

A **proof of concept for sequence gradients**, not a campaign to produce a
molecule. The question is whether a frozen structure predictor's gradient with
respect to the designed chain's sequence carries enough information to move
that predictor's own two readouts — interface confidence and target-aligned
pose — from a degraded start toward a reference. Every quantity in these
documents is the predictor's own output.

Reorganized 2026-10-06: the live set is this folder, superseded snapshots are
in [`archive/`](archive/).

## Start here

| Document | What it is |
|---|---|
| **[P17_JN1.md](P17_JN1.md)** | **The project record, §§1–28.** The only document that carries results. Read its front matter, then §19 (framing and the experiment ladder), then §§20–27 in order. §27 is the newest result; §28 is a feasibility audit with no result attached. |

## Live companions

| Document | What it is |
|---|---|
| [search_policy_and_pose_literature_review.md](search_policy_and_pose_literature_review.md) | Methodological review of search policy, gradient guidance and what "pose success" can mean. §24 of the record answers from it. |
| [site_targeting_synthesis.md](site_targeting_synthesis.md) | Full synthesis on reaching a *specified* arrangement: the four layers of site specification, why confidence metrics cannot be made site-aware, and parameter values from released code. ~6,600 words, 48 inline citations. §§22.3–26 draw on it. |
| [literature_review/](literature_review/README.md) | The eight deep note files behind that synthesis, one per question, each separating cited findings from inference. Start at its README. The newest, `continuous_relaxation_and_the_discretization_gap.md`, covers relaxed-vs-discrete sequence optimization. |
| [figures/](figures/) | Flow diagram sources and renders used by §12.2. |

## Archive

[`archive/`](archive/) holds documents that are **superseded but still cited**
by the record, kept because dated claims in §§1–18 point at them:

| Document | Why it is archived |
|---|---|
| [p17_status_and_next_steps.md](archive/p17_status_and_next_steps.md) | Status snapshot predating §§20–27. Its launch commands and output layout still stand; its purpose statements and next-step ordering are superseded by §§23.6, 26.5 and 27.7. |
| [protein_search_policy_review.md](archive/protein_search_policy_review.md) | The first search-policy survey (2026-09-21). Corrected several §8 claims at the time; the live companion above replaces it. |
| [numerical_precision_validation_summary.md](archive/numerical_precision_validation_summary.md) | The fp32/bf16 and aggregation validation work behind §§17.7–17.13. Prerequisite history, not a current result. |
| [opendde_validation_history.md](archive/opendde_validation_history.md) | Earlier OpenDDE empirical findings, summarized in §3 and §4. |

## Conventions

The prose was reworded on 2026-10-06 to describe the mechanical experiment
rather than an optimization campaign: *designed chain*, *specified site*,
*designable region*, *contact face*, *anchor residues*. **Code symbols, file
paths and other projects' filter names keep their original spellings** —
`BinderPoseRMSD`, `percent_interface_cdr`, `binder_near_hotspot` — so see the
terminology table in P17_JN1.md's front matter for the mapping. Where a cited
paper's finding is specifically about antibodies, its wording is left as
published, and the literature notes keep the vocabulary of the papers they
summarize, including direct quotations.

Sections of the record are dated and append-only: corrections are recorded in
place with ⚠️ rather than edited away, so a superseded claim and the evidence
that superseded it both stay visible.
