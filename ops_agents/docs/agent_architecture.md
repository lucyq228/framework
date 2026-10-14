# Agent architecture: PACE–CSAT correlation (Ops Analytics)

Status: **built**. Section 5 describes the implementation and section 6 lists the decisions made.

Source sketch: `context/agents_flow.HEIC` ("AI Agents Flow for Ops Analytics").

## 1. How this differs from the propensity-modeling demo

| | Propensity demo (`cursor-autonomous-agents-data-science-demo`) | This project |
|---|---|---|
| Question type | One target, one model | Many relationship questions across dimensions |
| Main risk | Leakage, overfitting | Confounding, wrong time alignment, tiny PACE samples, over-claiming |
| Pipeline | Fixed: clean → EDA → features → train → simulate | **Chosen per project**: the scope agent picks methodology pathways from a tool shelf |
| Key agent | Modeler | **Scope & Hypotheses agent** (A2) |
| Data | One flat table | Three sources joined at several grains, plus reference data |
| Output | Model + HTML report | Excel prototype + short insight readout |

What carries over: the manager runs in the main chat, specialists are Markdown subagents,
every calculation is a deterministic skill script, the approved scope is fingerprinted and locked,
and hooks log each step to `runs/<run-id>/`.

## 2. Agents

```text
            ┌──────────────── User (main chat) ────────────────┐
            │  input, answers, confirmation and approval        │
            └──────────────▲────────────────────┬──────────────┘
                           │ questions /        │ answers /
                           │ feedback           │ approval
┌──────────────────────────┴────────────────────▼──────────────────────────┐
│ A1 Routing Manager (main chat, LLM)                                       │
│   checks run status → delegates next milestone → reviews completion →     │
│   relays questions to the user → loops back when a step fails             │
└───┬──────────────┬──────────────┬──────────────┬─────────────┬───────────┘
    │              │              │              │             │
 A2 Scope &     A3 Feasibility  A4 Data       A5 Feature    A6 Analysis     A7 Review &
 Hypotheses     (LLM)           Quality       Creation      Execution       Synthesis
 (LLM)                          (tool)        (tool+LLM)    (tool)          (LLM)
    │              │              │              │             │              │
    └──────────────┴──────────────┴──── Shared Correlation Assets (tool shelf) ┘
```

| Agent | Kind | Proposed model | Owns milestone | Reads | Writes |
|---|---|---|---|---|---|
| **A1 Routing Manager** | Main chat | whichever model you pick in chat | orchestration | `run_status` | decisions log |
| **A2 Scope & Hypotheses** | Subagent | `claude-opus-5-5` (strongest reasoning) | `scope_draft` | brief, context docs, deck images, tool registry | `scope_proposal.json` (with `user_questions`), earlier versions kept |
| **A3 Feasibility** | Subagent | `gpt-5.6-sol` | `feasibility` | scope proposal, source profiles, tool registry requirements | `feasibility_matrix.json`, `feasibility_review.json` (verdicts, can/cannot do, data requests) |
| **A4 Data Quality & EDA** | Subagent | `composer-2.5` | `baseline_eda`, `data_quality`, `descriptive_eda` | raw sources, then scored datasets | `baseline_eda.json`, `dq_report.json`, `eda_descriptives.json`, EDA charts |
| **A5 Feature Creation** | Subagent | `composer-2.5` | `analysis_datasets` | locked scope, DQ report | scored PACE tables, CSAT windows, joined analysis datasets |
| **A6 Analysis Execution** | Subagent (tool runner) | `composer-2.5` | `analyses` | locked methodology plan | one result JSON per tool run |
| **A7 Review & Synthesis** | Subagent | `gpt-5.6-sol` (different model family from A2, so the challenge is independent) | `review`, `deliverables` | all results | `review.json`, `insights.json`, Excel workbook, readout |

### A2 Scope & Hypotheses: the most important agent

A2 should think like a senior analytics lead before anyone runs a number.

1. **Understand why the project exists.** The GSE KPI weights PACE and CSAT 50/50. The
   real business questions are whether the two measure related things, whether
   combining them is defensible, and where they diverge and why. A2 restates the
   decision the analysis informs, even though no decision is due yet.
2. **Turn the brief into testable hypotheses**, each with a reason and a pathway, for example:
   - H1 Restaurants with higher progressive PACE Hospitality have higher CSAT Hospitality, *within market*.
   - H2 The association is stronger when CSAT is measured close to the visit date.
   - H3 Unannounced CFV tracks CSAT better than announced RGRV.
   - H4 The association differs by channel (Drive Thru vs Front Counter vs Mobile-placed).
   - H5 Conservative scoring differentiates restaurants less than progressive scoring.
   - H6 There is a threshold: CSAT drops sharply below some PACE level.
   - H7 Taste and Hospitality carry shared information (halo), so the 50/50 split double-counts.
3. **Recommend methodology pathways.** For each hypothesis, A2 chooses an ordered set
   of shelf tools, the grain (market, region, restaurant-period, or visit), the time
   window, the scoring variant, the minimum sample rules, and what result would
   count as support.
4. **Separate decisions from defaults.** A2 writes the questions only the user can answer,
   such as the treatment of the RGRV general "Service" questions, local core
   products, 2025 inclusion, and the reporting grain. Each question gets a recommended option.
5. **Name the threats**: market confounding (Simpson's paradox), PACE
   sample size, ceiling effects, reliability of low-volume CSAT, and the risk of reading
   causation into the results.

A2's proposal goes to A3 before the user sees it, so the user reviews one combined
"here is what we want to do, and here is what the data can and cannot support" proposal.

### A3 Feasibility: what the data can and cannot do

For every pathway, A3 checks each tool's declared requirements (`tool_registry.yaml`)
against the profiled sources and returns one of three verdicts:

- **Supported**: the required columns, grain, and minimum sample size are all present.
- **Partial / proxy**: the pathway can run with a stated compromise (for example, failure
  reasons as a proxy for dimension-level taste).
- **Blocked**: the pathway needs data that does not exist. A3 writes a **data request**
  that says what to acquire, from which owner, at what grain, and which hypotheses it unlocks.

The manager presents the combined scope and feasibility proposal to the user. The user can approve it,
change it, or choose to acquire new data and rerun feasibility. Approval locks
`approved_scope.json` (SHA-256), and all later agents follow it.

### A7 Review & Synthesis: the insights agent

A7 does not just summarise. It does three things:

1. **Challenges** each result. It checks for a sufficient sample, confidence intervals that
   exclude zero, survival of FDR correction, consistency in direction across markets
   and scoring variants, and whether a confounder (market, channel mix, volume) explains the effect.
2. **Reconciles** results when tools disagree, for example a pooled correlation against
   within-market correlations, or Pearson against Spearman.
3. **Picks the insights** that matter for the GSE KPI design, ranks them by
   importance × robustness, and writes the Excel prototype and a 1–2 page
   readout. Each insight is tagged *aligned*, *diverges*, or *inconclusive*.

If a challenge fails, A7 can send a targeted rerun back through A1 (the dashed arrows in the sketch).

### A4 Data Quality & EDA: general data observations

EDA runs twice, so that both the scope and the final readout rest on real numbers:

1. **Baseline EDA (before scoping, raw data, read-only).** This pass covers coverage and volumes: restaurants
   per market, visits per restaurant per year by RGRV/CFV, and CSAT responses per
   restaurant-month. It also gives headline pass rates and top-two-box rates, plus obvious gaps. A2 uses these facts so its
   hypotheses fit the data (for example, "only 3–5 visits per restaurant per year, so
   restaurant-month correlation is not viable"). A3 uses them for sample-size verdicts.
2. **Descriptive EDA (after feature creation, on scored data, inside the locked scope).**
   This pass reports pass rates by product category, product, channel, touchpoint, visit type, market,
   and month. It shows conservative vs progressive score distributions, top failure reasons, and CSAT
   Taste/Hospitality by market, presented/placed channel, month, and weekday. It also covers 2025 vs 2026 trends and
   restaurant-type profiles when restaurant attributes exist.

The EDA output becomes the "Descriptives" tab of the Excel prototype. A7 treats
notable EDA observations as candidate insights, alongside the correlation results.

## 3. Lifecycle

```text
intake → A4 baseline EDA → A2 scope draft → A3 feasibility
      → user feedback loop (repeat until approved) → lock scope
      → A4 data quality → A5 feature creation → A4 descriptive EDA
      → A6 analysis execution → A7 review (may request reruns) → A7 deliverables → finalize
```

The manager never skips a milestone. The scripts refuse to run when a prerequisite artifact is
missing or the locked scope has changed.

## 4. Shared correlation assets (the tool shelf)

Every tool has a structured metadata entry in `tool_registry.yaml` and an
implementation in `corr_core/tools/`. Each entry records the question it answers, its method, the params it accepts, the
required sources, the minimum *n*, its outputs, its interpretation guardrails, its status, and the data that would unlock it.
One `correlation-shelf` skill browses and runs any tool, so adding a tool means adding a registry entry and a function,
not a new skill. Agents choose tools from the registry; the data science team builds and maintains the shelf.

The "With current data" column is A3's expected verdict on the synthetic data.

### 4.1 Connectors and platform (orange in the sketch)

| Tool | Purpose | Demo implementation | With current data |
|---|---|---|---|
| `gcp-connector` | Pull PACE and CSAT extracts from BigQuery | Local file loader with the same interface; BigQuery adapter stubbed | Supported (local) |
| `confluence-mcp` | Read background docs and definitions | Reads `context/` (Markdown, images, DOCX, PDF); real MCP pluggable | Supported (local) |
| `github-mcp` | Version tools, open issues for tool gaps | Writes a `tool_gap` request file; real MCP pluggable | Supported (local) |
| `profile-sources` | Schema, grain, keys, coverage, sample-size profile per source | Script | Supported |
| `data-quality-check` | Duplicates, impossible counts, key match rate, coverage gaps, date parsing, mapping status | Script | Supported |

### 4.2 Exploratory data analysis (A4)

| Tool | What it describes | With current data |
|---|---|---|
| `eda-coverage` | Restaurants, visits per restaurant per year by visit type, CSAT responses per restaurant-month, PACE–CSAT match coverage over time | Supported |
| `eda-pace-pass-rates` | Pass rates by product category, product, channel, touchpoint, visit type, market, month; conservative vs progressive distributions | Supported |
| `eda-failure-reasons` | Top failure reasons by product category and touchpoint | Supported |
| `eda-csat-rates` | Top-two-box Taste and Hospitality by market, presented/placed channel, month, weekday; response volumes | Supported |
| `eda-trends` | Monthly PACE pass rate and CSAT trends; 2025 vs 2026 | Supported |
| `eda-segment-profile` | The same views by restaurant type, ownership, volume band, region | Needs `restaurant_attributes` |
| `eda-summary-pack` | Chart pack plus a structured list of general observations for A2 and A7 | Supported |

### 4.3 Feature creation (A5)

| Tool | Purpose | With current data |
|---|---|---|
| `apply-question-mapping` | Map local → global codes; flag unmapped | Supported (Q27a-CA unmapped, Q10a-UK proposed) |
| `score-pace-taste` | Product → visit → restaurant-period; conservative and progressive | Supported |
| `score-pace-hospitality` | Touchpoint → journey → visit → restaurant-period; conservative and progressive | Supported (general "Service" treatment is a scope decision) |
| `aggregate-csat` | Top-two-box rates by restaurant × period × channel, with n and binomial SE | Supported |
| `align-time-windows` | CSAT window around each visit (±7/14/30/60/90 days, pre/post, calendar month/quarter/year) | Supported |
| `build-analysis-dataset` | Join scored PACE and CSAT at the locked grain; report match rate | Supported |
| `compute-gse-index` | Taste, Hospitality, and GSE KPI per the deck, including weight sensitivity | Supported |

### 4.4 Analysis families (yellow in the sketch)

**Distributions & variation**
| Tool | What it answers | With current data |
|---|---|---|
| `describe-distributions` | Shape, ceiling/floor share, discreteness of PACE scores | Supported |
| `variance-decomposition` | Between- vs within-restaurant variance (ICC) for PACE and CSAT | Supported |
| `csat-reliability` | Sampling noise vs true variation by volume; Spearman–Brown reliability; minimum-n rule | Supported |

**Pairwise association**
| Tool | What it answers | With current data |
|---|---|---|
| `pairwise-correlation` | Pearson, Spearman, Kendall with bootstrap CIs | Supported |
| `weighted-correlation` | Correlation weighted by CSAT responses / PACE visits | Supported |
| `point-biserial` | Conservative pass/fail vs CSAT rate | Supported |
| `disattenuated-correlation` | Correlation corrected for measurement noise on both sides | Supported |
| `permutation-test` | Significance without normality assumptions | Supported |
| `multiple-testing` | Benjamini–Hochberg FDR across every segment and test run | Supported |

**Component decomposition**
| Tool | What it answers | With current data |
|---|---|---|
| `component-correlation` | Which product category, channel, or touchpoint tracks CSAT most | Supported |
| `failure-reason-analysis` | Do appearance / temperature / taste / texture failures relate to CSAT Taste? | **Partial**: reasons exist only on failures; CSAT has no dimension split |
| `product-level-csat` | Product-specific PACE vs product-specific satisfaction | **Blocked**: needs product-level CSAT |
| `taste-vs-hospitality-crossover` | Halo: does PACE Hospitality predict CSAT Taste and vice versa? | Supported |

**Panel effect**
| Tool | What it answers | With current data |
|---|---|---|
| `between-within-correlation` | Between-restaurant vs within-restaurant (over time) association | Supported (thin within-restaurant: 3–5 visits/yr) |
| `mixed-effects-model` | Random intercept by restaurant, fixed effects for market and season | Supported |
| `fixed-effects-panel` | Market or restaurant fixed effects; removes stable confounders | Supported |

**Segmentation & heterogeneity**
| Tool | What it answers | With current data |
|---|---|---|
| `segment-correlation` | Correlation by market, region, restaurant type, ownership, volume, channel, placed channel | Supported if `restaurant_attributes` is acquired; market and channel only without it |
| `segment-difference-test` | Is a segment's correlation really different? (Fisher z, interaction term) | Supported |
| `simpsons-paradox-check` | Pooled vs within-group direction and size | Supported |
| `min-sample-guard` | Suppresses segments below minimum *n* (brief: "only where the sample supports it") | Supported |

**Time & longitudinal**
| Tool | What it answers | With current data |
|---|---|---|
| `lag-window-scan` | Correlation as a function of the CSAT window and lag around the visit | Supported |
| `rolling-correlation` | Stability over months, 2025 vs 2026 | Supported |
| `change-on-change` | Does a PACE change between visits move with a CSAT change? | Supported (small n) |
| `seasonality-adjust` | Remove month/seasonal CSAT effects before correlating | Supported |
| `daypart-alignment` | Match visit time-of-day to CSAT daypart | **Blocked**: needs visit timestamp and CSAT hour/daypart |

**Consistency analysis**
| Tool | What it answers | With current data |
|---|---|---|
| `quadrant-agreement` | High/low PACE × high/low CSAT; who sits in the divergent quadrants | Supported |
| `rank-agreement` | Top/bottom-decile overlap, Kendall rank agreement, tier kappa | Supported |
| `rgrv-cfv-gap` | Announced vs unannounced gap and which one tracks CSAT | Supported |
| `scoring-variant-comparison` | Conservative vs progressive: spread, ties, ranking stability, CSAT association | Supported |
| `bland-altman` | Systematic offset between PACE % and CSAT % on the same scale | Supported |

**Nonlinearity & thresholds**
| Tool | What it answers | With current data |
|---|---|---|
| `binned-means` | CSAT by PACE band, with CIs | Supported |
| `lowess-spline` | Smooth shape of the relationship | Supported |
| `breakpoint-regression` | Is there a PACE threshold below which CSAT drops? | Supported |
| `nonlinear-dependence` | Mutual information / distance correlation | Supported |

**Multivariable regression**
| Tool | What it answers | With current data |
|---|---|---|
| `controlled-regression` | PACE → CSAT controlling for market, type, channel mix, season (WLS) | Supported |
| `binomial-glm` | CSAT counts as a binomial/fractional logit outcome | Supported |
| `partial-correlation` | PACE–CSAT net of the controls | Supported |
| `operational-drivers` | Add staffing, speed of service, crew tenure | **Blocked**: needs ops data |

**Relative importance & shared information**
| Tool | What it answers | With current data |
|---|---|---|
| `dominance-analysis` | Shapley R² share of each component (products, channels, touchpoints) | Supported |
| `commonality-analysis` | Unique vs shared variance of Taste and Hospitality | Supported |
| `collinearity-check` | VIF / condition index among components | Supported |
| `index-weight-sensitivity` | How restaurant rankings move if the 50/50 weights change | Supported |

**Reporting**
| Tool | What it answers | With current data |
|---|---|---|
| `build-excel-prototype` | Required output: Descriptives (EDA) tab, one tab per analysis, scorecard, data notes | Supported |
| `build-readout` | Short insight readout (Markdown → DOCX/HTML) | Supported |
| `chart-pack` | Standard charts per tool | Supported |

### 4.5 What the current data cannot do (A3 will say this to the user)

| Need | Why it is blocked | What to acquire |
|---|---|---|
| Product-level taste correlation | CSAT quality is one overall question | CSAT item-level or product-tagged responses |
| Dimension-level taste (temperature, texture, and so on) | Only failure reasons, no CSAT dimensions | Full PACE dimension results; CSAT dimension questions |
| Delivery hospitality | CSAT has no delivery presented channel | Delivery-aggregator satisfaction feed |
| Daypart alignment | No visit time, no CSAT hour | Visit timestamp; CSAT daypart |
| Operational explanations for divergence | No ops data | Speed of service, staffing, crew tenure |
| Restaurant-type / ownership / region cuts | Not in the three sample sheets | Restaurant master (synthetic `restaurant_attributes.csv` stands in) |
| Pooling Canada and UK&I | Local questions not fully mapped | Confirmed question mapping (Q10a-UK, Q27a-CA pending) |
| Causal claims | Observational data | Not possible from this design; would need a pilot or experiment |

## 5. Cursor implementation

```text
AGENTS.md                         rules every agent reads
.cursor/
  agents/                         a2-scope.md, a3-feasibility.md, a4-data-quality-eda.md,
                                  a5-feature-creation.md, a6-analysis-execution.md, a7-review-synthesis.md
  skills/
    run-correlation-project/      A1 playbook: start_run, run_status, record_decision, record_answers,
                                  request_data, approve_scope, finalize_run
    scope-hypotheses/             A2: read_context, save_scope_proposal (validated, versioned)
    feasibility-check/            A3: check_feasibility (deterministic matrix), save_feasibility_review (stricter only)
    data-quality-eda/             A4: baseline_eda, data_quality, descriptive_eda
    feature-creation/             A5: build_datasets
    correlation-shelf/            A6 (and A2/A3 browsing): list_tools, run_plan, run_tool (follow-ups)
    review-synthesis/             A7: review_checks, save_review, save_insights
    build-deliverables/           A7: build_deliverables (charts, Excel prototype, readout)
  rules/                          locked-scope.mdc, data-privacy.mdc, no-causal-language.mdc
  hooks.json, hooks/              subagent + shell event log per run
corr_core/                        shared Python: paths, events, run state, scope lock, sources, scoring,
                                  features, datasets, stats, charts, registry matching, guarded runner
corr_core/tools/                  the 64 tool implementations, one module per family
tool_registry.yaml                structured metadata for every shelf tool (feasibility matching reads this)
examples/example_scope_proposal.json   format reference for A2 and the test fixture
context/                          brief, data dictionary, deck images
data/raw, data/reference          synthetic sources; data/acquirable holds data the user can request
runs/<run-id>/                    every artifact, drafts/, results/, datasets/, charts/, Excel, readout, trace
tests/                            scoring unit tests, end-to-end run through every script, shelf smoke sweep
```

How the guardrails are enforced in code, not only in prompts:

| Guardrail | Where |
|---|---|
| Feasibility must match the current proposal and data inventory | `save_feasibility_review.py`, `scope.approve`, `state` (stale detection) |
| A3 can only make verdicts stricter | `save_feasibility_review.py` |
| No approval with blocked steps or unanswered questions | `scope.approve` |
| The approved scope cannot change | SHA-256 lock checked by `load_approved_scope` in every later script |
| Analyses run only from the approved plan, or as recorded follow-ups that pass feasibility | `runner.execute` |
| Tools reject params they do not accept (including `grain` on fixed-grain tools) | `registry.allowed_params` |
| The review is tied to the analyses, insights to the review, deliverables to the insights | hashes checked by `state` |
| Insights must cite existing results | `save_insights.py` |

## 6. Decisions made

1. All 64 shelf entries are built. Three are blocked stubs that explain which data would
   unlock them (`product-level-csat`, `daypart-alignment`, `operational-drivers`), and one is partial
   (`failure-reason-analysis`).
2. A2 and A3 write questions and data requests into their artifacts; the manager relays them to
   the user and records the answers (`user_answers.json`) and acquisitions.
3. Models as in section 2.
4. `restaurant_attributes.csv` starts in `data/acquirable/`, so the demo shows the
   "acquire new data, then re-check feasibility" loop.
