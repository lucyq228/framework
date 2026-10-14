---
name: review-synthesis
description: Challenge every PACE-CSAT analysis result, reconcile tools that disagree, give each hypothesis a verdict, request targeted reruns, and pick the important insights for the Excel prototype and readout. Use for the review and insights milestones (A7 Review & Synthesis).
---

# Review and synthesis

Commands run from the project root with `.venv\Scripts\python.exe`.

## 1. Deterministic checks

```powershell
.venv\Scripts\python.exe .cursor\skills\review-synthesis\scripts\review_checks.py
```

Per result: n below the scope minimum, CIs that include 0, findings that do
not survive Benjamini-Hochberg FDR, tool flags (such as Simpson's paradox),
partial steps, and mixed signs. Run level: missing steps, tool errors,
hypotheses with no result, and a stale FDR table. Writes `review_checks.json`.

## 2. Review (judgment)

Read `approved_scope.json` (hypotheses and support criteria), `review_checks.json`,
the result files in `results/`, and `eda_descriptives.json`. Then:

1. **Challenge** each hypothesis against its `support_criteria`: sample, CI,
   FDR, direction across markets, grains, scoring variants, and windows, and
   whether a confounder (market, channel mix, visit type, volume) explains it.
2. **Reconcile** disagreements: pooled vs within-market, Pearson vs rank,
   visit vs restaurant grain, between- vs within-restaurant.
3. **Rerun requests** only for problems inside the locked scope (a planned step
   to re-run, or a follow-up tool with params and a reason).

Write `runs/<run>/drafts/review.json`:

```json
{
  "verdict": "pass | pass_with_caveats | fail",
  "summary": "...",
  "hypothesis_verdicts": [{"id": "H1", "verdict": "supported | partially_supported | not_supported | inconclusive",
                           "reason": "...", "evidence": ["H1_3", "H1_4"]}],
  "challenges": ["..."],
  "reconciliations": ["..."],
  "rerun_requests": [{"tool": "segment-correlation", "params": {}, "reason": "..."}]
}
```

```powershell
.venv\Scripts\python.exe .cursor\skills\review-synthesis\scripts\save_review.py --file runs\<run>\drafts\review.json
```

`fail` means the results cannot support a readout as they stand. Use
`pass_with_caveats` when the conclusions hold with stated limits.

## 3. Insights

Pick 4-8 insights that matter for the GSE KPI design. Rank them by importance
x robustness, not by p-value. Tag each one: **aligned** (PACE and CSAT move together),
**diverges** (they disagree, and where), or **inconclusive**. Notable descriptive
EDA facts can be insights; cite them as `eda:<tool-id>`.

Write `runs/<run>/drafts/insights.json`:

```json
{
  "headline": "One sentence.",
  "executive_summary": ["at most 5 bullets"],
  "insights": [{"id": "I1", "title": "...", "tag": "aligned", "importance": "high", "robustness": "medium",
                "so_what": "What it means for the KPI design.",
                "evidence": [{"statement": "Within-market r = 0.37 [0.31, 0.43], n = 1,807", "result": "H1_3"}]}],
  "limitations": ["..."],
  "data_requests": ["..."],
  "next_steps": ["..."]
}
```

```powershell
.venv\Scripts\python.exe .cursor\skills\review-synthesis\scripts\save_insights.py --file runs\<run>\drafts\insights.json
```

Every evidence statement must quote numbers from the cited result. Write
associations, never causes ("restaurants with higher PACE tend to have higher
CSAT", not "PACE drives CSAT").
