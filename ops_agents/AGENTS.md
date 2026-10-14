# Cursor autonomous agents: PACE vs CSAT correlation (Ops Analytics)

This project is a team of Cursor agents that scopes, checks, runs, reviews,
and reports a correlation analysis between PACE (restaurant execution audits)
and CSAT (customer satisfaction). There is no Python orchestrator. The agents
are Markdown files; every calculation is a deterministic script packaged as a skill.

## The team

| Agent | Where it lives | Model | Owns |
|---|---|---|---|
| A1 Routing Manager | Main chat, following `.cursor/skills/run-correlation-project/SKILL.md` | the chat's model | user dialogue, approval, routing |
| A2 Scope & Hypotheses | `.cursor/agents/a2-scope.md` | claude-opus-5-5 | `scope_draft` |
| A3 Feasibility | `.cursor/agents/a3-feasibility.md` | gpt-5.6-sol | `feasibility` |
| A4 Data Quality & EDA | `.cursor/agents/a4-data-quality-eda.md` | composer-2.5 | `baseline_eda`, `data_quality`, `descriptive_eda` |
| A5 Feature Creation | `.cursor/agents/a5-feature-creation.md` | composer-2.5 | `analysis_datasets` |
| A6 Analysis Execution | `.cursor/agents/a6-analysis-execution.md` | composer-2.5 | `analyses` |
| A7 Review & Synthesis | `.cursor/agents/a7-review-synthesis.md` | gpt-5.6-sol | `review`, `insights`, `deliverables` |

The shared correlation assets (the tool shelf) are 64 tools registered in
`tool_registry.yaml` and implemented in `corr_core/tools/`. Agents pick tools
from the registry; the data science team maintains it.

## Lifecycle (never skip a step)

```text
start -> A4 baseline EDA -> A2 scope draft -> A3 feasibility
      -> user feedback loop (answers, data acquisition, revisions) -> user approves -> lock scope
      -> A4 data quality -> A5 feature creation -> A4 descriptive EDA
      -> A6 analyses -> A7 review (may request reruns) -> A7 insights -> A7 deliverables -> finalize
```

Subagents cannot talk to the user. They write questions into their artifacts,
and the manager asks the user and records the answers.

## Non-negotiable rules for every agent

- **The approved scope is a locked contract** (`approved_scope.json`, SHA-256 in
  `scope.lock.json`). Never edit it. A different scope means a new run.
- **Calculations come only from the skill scripts.** No ad-hoc pandas or stats
  code. If a tool is missing, report a tool gap.
- **Never hand-edit run artifacts.** Agents write drafts in `runs/<run>/drafts/`;
  the save scripts validate them.
- **No raw rows in the conversation, and never open `synthetic/`** (it holds the
  planted ground truth used to test the agents).
- **Associations, not causes.** Quote estimates with n and CI.

## Running scripts

Use the project's virtual environment from the project root:

```powershell
.venv\Scripts\python.exe .cursor\skills\<skill>\scripts\<script>.py [--run <run-id>] [--actor <agent>]
```

`--run` defaults to the run named in `runs/CURRENT`. Every script prints compact
JSON (`ok`, summary, highlights) and exits non-zero when it refuses or fails.
Pass structured input as files (`--file`, `--params-file`, `--answers-file`),
not inline JSON: Windows PowerShell strips the quotes.

## Where things are

- `context/`: project brief, data dictionary, deck images (JPG previews in `context/_jpg/`).
- `data/raw/`, `data/reference/`: synthetic sources. `data/acquirable/` simulates data the user can request.
- `corr_core/`: shared Python (paths, run state, scope lock, sources, scoring, features, stats, tools).
- `tool_registry.yaml`: metadata for every shelf tool; feasibility reads it.
- `.cursor/skills/`: the manager playbook and one skill per agent task.
- `runs/<run-id>/`: every artifact, the Excel prototype, the readout, and `live_run.log`.
