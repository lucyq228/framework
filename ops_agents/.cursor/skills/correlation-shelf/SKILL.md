---
name: correlation-shelf
description: The shared correlation assets - 64 registered tools for PACE vs CSAT analysis (association, segmentation by market, channel, visit type and restaurant type, time windows and lags, panel effects, consistency, nonlinearity, regression, relative importance), with structured metadata in tool_registry.yaml. Use to browse tools when designing a scope, to run the approved analysis plan (A6), or to run a recorded follow-up analysis.
---

# Correlation shelf

The data science team owns `tool_registry.yaml` and `corr_core/tools/`. Each
registry entry states the question a tool answers, its method, accepted
params, required sources and minimum n, outputs, interpretation guardrails,
status, and what data would unlock it. Agents choose tools from the registry and
never write their own analysis code.

Commands run from the project root with `.venv\Scripts\python.exe`.

## Browse

```powershell
.venv\Scripts\python.exe .cursor\skills\correlation-shelf\scripts\list_tools.py --phase analysis
.venv\Scripts\python.exe .cursor\skills\correlation-shelf\scripts\list_tools.py --id lag-window-scan
.venv\Scripts\python.exe .cursor\skills\correlation-shelf\scripts\list_tools.py --family segment --available-only
```

| Family | Phase | Tools (`~` partial, `*` blocked: needs data that does not exist) |
|---|---|---|
| Connectors & platform | intake, quality | gcp-connector, confluence-mcp, github-mcp, profile-sources, data-quality-check |
| Exploratory data analysis | eda | eda-coverage, eda-pace-pass-rates, eda-failure-reasons, eda-csat-rates, eda-trends, eda-segment-profile, eda-summary-pack |
| Feature creation | feature | apply-question-mapping, score-pace-taste, score-pace-hospitality, aggregate-csat, align-time-windows, build-analysis-dataset, compute-gse-index |
| Distributions & variation | analysis | describe-distributions, variance-decomposition, csat-reliability |
| Pairwise association | analysis | pairwise-correlation, weighted-correlation, point-biserial, disattenuated-correlation, permutation-test, multiple-testing |
| Component decomposition | analysis | component-correlation, failure-reason-analysis~, product-level-csat*, taste-vs-hospitality-crossover |
| Panel effect | analysis | between-within-correlation, mixed-effects-model, fixed-effects-panel |
| Segmentation & heterogeneity | analysis | segment-correlation, segment-difference-test, simpsons-paradox-check, min-sample-guard |
| Time & longitudinal (lags) | analysis | lag-window-scan, rolling-correlation, change-on-change, seasonality-adjust, daypart-alignment* |
| Consistency analysis | analysis | quadrant-agreement, rank-agreement, rgrv-cfv-gap, scoring-variant-comparison, bland-altman |
| Nonlinearity & thresholds | analysis | binned-means, lowess-spline, breakpoint-regression, nonlinear-dependence |
| Multivariable regression | analysis | controlled-regression, binomial-glm, partial-correlation, operational-drivers* |
| Relative importance & shared information | analysis | dominance-analysis, commonality-analysis, collinearity-check, index-weight-sensitivity |
| Reporting | reporting | chart-pack, build-excel-prototype, build-readout |

Common params for analysis tools: `measures` (taste, hospitality), `scoring`,
`grain` (visit or restaurant_period), `window` (CSAT window), `csat_channel`,
`visit_types`, `period`, `market_group`, `market`, `min_n`,
`min_csat_responses`. Unset params fall back to the locked scope. Tools marked
`fixed_grain` choose their own grain and reject `grain`. Segment cuts use
`segments`: market, market_group, survey_name, channel_tested, csat_channel
always; region, restaurant_type, ownership_type, volume_band, and channel flags
need `restaurant_attributes`.

## Run the approved plan (A6)

```powershell
.venv\Scripts\python.exe .cursor\skills\correlation-shelf\scripts\run_plan.py --actor a6-analysis-execution
.venv\Scripts\python.exe .cursor\skills\correlation-shelf\scripts\run_plan.py --steps H1.2,H3.1 --actor a6-analysis-execution
```

Runs every planned step without a result (or the listed steps), always running
`multiple-testing` last, writes `results/<step_id>.json`, and rebuilds
`analyses_index.json`. The milestone is complete only when no planned step is
missing and the last run had no errors.

## Run a follow-up

Write the params to a file (inline JSON loses its quotes in Windows PowerShell),
for example `runs/<run>/drafts/hosp_by_channel.json` containing
`{"segments": ["channel_tested"], "measures": ["hospitality"]}`, then:

```powershell
.venv\Scripts\python.exe .cursor\skills\correlation-shelf\scripts\run_tool.py --tool segment-correlation `
  --params-file runs\<run>\drafts\hosp_by_channel.json --label hosp_by_channel `
  --followup-reason "Hospitality r differs by market; check whether channel mix explains it"
```

Simple scalar params can use `--param grain=restaurant_period` instead.

A follow-up must state its reason and pass the same feasibility checks as a
planned step. Its result is saved as `results/F_<label>.json` and indexed as a
follow-up, so the reviewer can tell exploratory work from the approved plan.
After any follow-up, re-run the plan's `multiple-testing` step
(`run_plan.py --steps <its step id>`) so the FDR table includes it.
