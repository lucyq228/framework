---
name: data-quality-eda
description: Baseline EDA on the raw PACE and CSAT sources before scoping, data-quality checks under the locked scope, and descriptive EDA on the scored datasets (pass rates, score distributions, failure reasons, CSAT rates, trends, segment profiles). Use for the baseline_eda, data_quality, and descriptive_eda milestones (A4 Data Quality & EDA).
---

# Data quality and EDA

Commands run from the project root with `.venv\Scripts\python.exe`. Each script
prints summaries and observations only; detail is in the JSON artifact.

| Milestone | Script | When | Artifact |
|---|---|---|---|
| baseline_eda | `baseline_eda.py` | Before scoping; no scope needed | `baseline_eda.json`, `context_text.md`, charts |
| data_quality | `data_quality.py` | After approval | `dq_report.json` |
| descriptive_eda | `descriptive_eda.py` | After `analysis_datasets` | `eda_descriptives.json`, charts |

```powershell
.venv\Scripts\python.exe .cursor\skills\data-quality-eda\scripts\<script>.py --actor a4-data-quality-eda
```

**Baseline EDA** runs the connector and profiler, extracts the context
documents, and runs every EDA tool on all markets and dates with default
scoring decisions: coverage (restaurants, visits per restaurant per year by
visit type, CSAT responses), PACE pass rates (progressive and conservative,
RGRV vs CFV, weakest category), CSAT top-two-box rates by market, monthly
trends, failure reasons, segment profiles, and a list of general observations.

**Data quality** checks duplicates, impossible counts (top-box above answered),
restaurants without CSAT, unconfirmed local question codes, unavailable
products, CFV channel counts, general Service questions, visits per restaurant
per year, partial CSAT history, and territory codes. Each check has a status
(pass, info, warn, fail) and a handling rule. The worst status is `overall`.

**Descriptive EDA** repeats the EDA tools on the scored, in-scope data, so
the numbers match the analysis. It becomes the Descriptives tab of the Excel prototype.

How to interpret:

- Say what the numbers mean for the analysis, for example "3.5 visits per
  restaurant-year means restaurant scores move in coarse steps".
- Separate a data defect (fix or exclude) from a real-world pattern (report it).
- A `fail` check is a blocker for the manager, not something to work around.
