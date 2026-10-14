---
name: feasibility-check
description: Check whether the current data supports each step of the PACE-CSAT scope proposal, add analyst judgment, and save a feasibility review that tells the user what the data can and cannot do and what to acquire. Use for the feasibility milestone (A3 Feasibility), and again whenever the scope or the data inventory changes.
---

# Feasibility check

Commands run from the project root with `.venv\Scripts\python.exe`.

## 1. Deterministic matrix

```powershell
.venv\Scripts\python.exe .cursor\skills\feasibility-check\scripts\check_feasibility.py
```

For every pathway step it checks the registry requirements against the data:
required sources, tool status, segments and controls that need restaurant
attributes, windows, CSAT channel, grain, estimated n against `min_n`, repeat
visits, and visits per restaurant-period. Verdicts: **supported**, **partial**
(runs with a stated compromise), **blocked** (needs data that does not exist).
It writes `feasibility_matrix.json` and prints the counts, the sample-size facts,
the non-supported steps, and the data requests.

## 2. Judgment

The matrix only knows declared requirements. Add what it cannot see:

- Power: is the estimated n enough for the effect size that would matter?
- Grain and window fit: does each step's grain answer its hypothesis?
- Proxies: where a step is partial, is the compromise acceptable?
- Gaps the registry does not list (for example, Delivery CSAT does not exist,
  so a Delivery Hospitality correlation cannot be tested at all).

You may make a verdict **stricter** with a reason, never more lenient. If a step
should run differently, say so in `recommendation`; A2 changes the scope.

Write `runs/<run>/drafts/feasibility_judgment.json`:

```json
{
  "summary": "One sentence for the user.",
  "can_do": ["Plain-language capabilities with the current data"],
  "cannot_do": ["Plain-language limits, each with the missing data"],
  "step_judgments": [{"step_id": "H6.3", "verdict": "partial", "reason": "..."}],
  "extra_data_requests": [{"need": "...", "owner": "...", "why": "...", "unlocks": ["H4"]}],
  "user_questions": [{"id": "F1", "question": "...", "options": ["..."], "recommended": "...", "why_it_matters": "..."}],
  "recommendation": "What the manager should propose to the user."
}
```

## 3. Save

```powershell
.venv\Scripts\python.exe .cursor\skills\feasibility-check\scripts\save_feasibility_review.py --file runs\<run>\drafts\feasibility_judgment.json
```

It refuses when the matrix is older than the proposal or the data inventory
(re-run step 1), when a verdict is relaxed, or when a step id is unknown.
