---
name: scope-hypotheses
description: Read the PACE-CSAT business context and baseline EDA, then draft and save a scope proposal with testable hypotheses, methodology pathways built from the tool shelf, threats, and questions for the user. Use for the scope_draft milestone (A2 Scope & Hypotheses) and when revising a scope after user answers or feasibility feedback.
---

# Scope and hypotheses

Commands run from the project root with `.venv\Scripts\python.exe`.

## 1. Read the context

```powershell
.venv\Scripts\python.exe .cursor\skills\scope-hypotheses\scripts\read_context.py
```

It prints the files to read (brief, data dictionary, extracted context, deck
image previews, the tool registry, an example scope), the baseline EDA
observations, which sources exist or could be acquired, the valid values for
every scope field, the user's recorded answers, and feasibility feedback on any
earlier version. Read the brief and data dictionary in full. View the deck
previews that explain the KPI method. Browse the shelf with
`.cursor\skills\correlation-shelf\scripts\list_tools.py --phase analysis`
and `--id <tool>` for detail.

## 2. Draft the scope

Write the draft to `runs/<run>/drafts/scope_draft.json`. Use
`examples/example_scope_proposal.json` as a format reference only. Your
hypotheses and pathways must come from this project's context and baseline EDA.

| Field | Meaning |
|---|---|
| `project_name`, `business_context`, `decision_supported` | Why the project exists and what decision it informs. Required. |
| `markets` | Subset of Canada, United Kingdom, Ireland. |
| `period` | `{start, end}` as YYYY-MM-DD; note whether 2025 is included. |
| `visit_types` | RGR and/or CFV. |
| `primary_scoring` / `compare_scoring` | progressive or conservative. |
| `primary_grain` | visit, or restaurant_period. |
| `restaurant_period` | year, half, quarter, or all (for restaurant_period grain). |
| `primary_csat_window` | A CSAT window around the visit (`pm30` = 30 days either side). |
| `decisions` | `general_service_questions`, `local_core_products`, `proposed_mappings`. Use the recommended option and ask the user when it matters. |
| `min_sample` | `min_n_correlation` (>= 10) and `min_csat_responses`. |
| `hypotheses[]` | `id`, `statement`, `rationale`, `support_criteria`, `pathway[]`. |
| `pathway[]` | `{tool, params, purpose}`; analysis tools only; params must be accepted by the tool. |
| `threats` | What could mislead: confounding, small PACE samples, ceilings, CSAT noise, causality. |
| `user_questions[]` | `{id, question, options, recommended, why_it_matters}`; leave `answer` empty. |
| `out_of_scope` | What this analysis will not claim or deliver. |

Drafting rules:

- Each hypothesis must be falsifiable, with `support_criteria` stating the
  result that would count as support (sign, CI, FDR, consistency across grains or markets).
- Pathways go from simple to robust: describe, correlate, then check
  confounding (within market, controls), then robustness (rank methods, scoring
  variants, time windows). Put `multiple-testing` last in the final hypothesis.
- Grain must fit the data: PACE has about 3-5 visits per restaurant per year,
  so restaurant-month correlations are not viable.
- Restaurant-type, region, ownership, and volume cuts need `restaurant_attributes`.
  Include them if they matter to the business question even when the source is
  missing: feasibility will turn them into a data request for the user.
- Ask only questions whose answer changes the analysis, and recommend an option.
- After the user answers (`user_answers` from read_context), copy each answer
  into the matching question's `answer` field and change the scope to match.

## 3. Save

```powershell
.venv\Scripts\python.exe .cursor\skills\scope-hypotheses\scripts\save_scope_proposal.py --file runs\<run>\drafts\scope_draft.json
```

It validates (fields, tool ids, phases, params, question fields) and writes
`scope_proposal.json`, keeping earlier versions as `scope_proposal.v<N>.json`.
Fix every reported issue and save again. Nothing is saved while issues remain.
