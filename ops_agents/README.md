# PACE vs CSAT correlation: a Cursor-native agent flow

A manager and six specialist subagents scope, check, run, review, and report a
correlation analysis between PACE (restaurant execution audits) and CSAT
(customer satisfaction) for Canada and UK&I. They use a shelf of 64 registered
analysis tools. Everything runs inside Cursor: the agents are Markdown files and
every calculation is a deterministic skill script. The data is synthetic.

See `docs/agent_architecture.md` for the design and `AGENTS.md` for the rules
every agent follows.

## Set up

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts\generate_synthetic_data.py   # already done; re-run to regenerate
```

Open **this folder** as the Cursor workspace root, so that `.cursor/agents`,
`.cursor/skills`, `.cursor/rules`, and `.cursor/hooks.json` load.

## Run it

In a new Cursor chat, type:

```text
/run-correlation-project Run the PACE vs CSAT correlation analysis for Canada and UK&I.
```

What to expect:

1. A4 profiles the data and runs baseline EDA (pass rates, CSAT rates, coverage).
2. A2 (Opus) reads the brief and the deck and proposes hypotheses and pathways.
3. A3 checks each step against the data. With the starting data, restaurant-type
   cuts are **blocked**, and A3 asks for the restaurant master.
4. The manager shows you one proposal covering what the data can and cannot do,
   the data requests, and the open questions. Agree to acquire
   `restaurant_attributes` to see the acquire-and-recheck loop, answer the questions,
   and approve.
5. A4 checks data quality, A5 builds the scored datasets, A4 runs descriptive EDA,
   A6 runs the plan, and A7 challenges the results, picks the insights, and builds
   the Excel prototype and readout.

Each run lives in `runs/<run-id>/`. Follow `live_run.log` there as it happens.

## What the synthetic data contains

`data/raw/` holds daily CSAT by restaurant and channel, PACE product checks, and
PACE service checks for 260 restaurants, 2025-01 to 2026-09. `data/reference/`
holds the local-to-global question mapping. `data/acquirable/` holds the
restaurant master, which the user can "acquire" during scoping.

Planted signals the agents should find (ground truth in `synthetic/`, which the agents never read):

- PACE and CSAT are positively associated, and much more strongly within market
  than pooled (market-level shifts run in opposite directions: Simpson's paradox).
- CSAT measured close to the visit tracks PACE better than full-year CSAT.
- Announced RGRV visits score higher than unannounced CFV visits, and CFV tracks CSAT better.
- Built-in data-quality issues: duplicates, impossible counts, restaurants
  without CSAT, unavailable products, unconfirmed local questions, general Service questions.

## Test

```powershell
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe tests\smoke_pipeline.py --sweep   # every shelf tool, without agents
```

`tests/test_end_to_end.py` drives every skill script as the agents would. It
checks that the guardrails refuse what they should (approval before
feasibility, relaxed verdicts, stale feasibility after new data, edits after the
lock, unplanned analyses without a reason, blocked tools, insights without
evidence) and that the planted signals are recovered.
