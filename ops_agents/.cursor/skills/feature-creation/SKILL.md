---
name: feature-creation
description: Build the PACE-CSAT analysis datasets from the locked scope - question mapping, progressive and conservative PACE Taste and Hospitality scores, cleaned CSAT, CSAT windows aligned to each visit, restaurant-period tables, and the GSE KPI indexes. Use for the analysis_datasets milestone (A5 Feature Creation).
---

# Feature creation

```powershell
.venv\Scripts\python.exe .cursor\skills\feature-creation\scripts\build_datasets.py --actor a5-feature-creation
```

It refuses without a DQ report, and when the DQ report failed unless
`--dq-override-reason "<the user's words, relayed by the manager>"` is passed.

What it builds (in `runs/<run>/datasets/`, Parquet):

| Dataset | Grain | Contents |
|---|---|---|
| `visit_scores` | visit | Taste and Hospitality, progressive and conservative; category, channel, and touchpoint pass flags; failure dimensions |
| `csat_windows` | visit x window x CSAT channel | Top-two-box counts and rates in each window around the visit |
| `restaurant_period_csat`, `restaurant_period` | restaurant x period | Mean PACE, CSAT, response counts, GSE indexes |
| `csat_monthly` | restaurant x month | CSAT by month |
| `market_scorecard` | market | The deck's GSE KPI scorecard |

Scoring follows the deck: a product passes only if all four dimensions pass; a
Hospitality journey passes only if every touchpoint passes; progressive = share
passed, conservative = all passed. The scope's `decisions` control general
Service questions, local core products, and proposed question mappings.

Check the printed summaries against the DQ report: duplicates removed, visits
scored, mean eligible products, channels assessed, and mapping decisions applied.
Report anything that looks inconsistent (for example, far fewer visits scored
than the DQ report counted).
