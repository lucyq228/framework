---
name: build-deliverables
description: Build the PACE-CSAT deliverables from the saved insights - chart pack, Excel prototype (README, Insights, Scorecard, Descriptives, Hypotheses, one sheet per analysis, Restaurant Detail, Data Notes), and the short insight readout in Markdown and HTML. Use for the deliverables milestone after insights.json is saved.
---

# Build deliverables

```powershell
.venv\Scripts\python.exe .cursor\skills\build-deliverables\scripts\build_deliverables.py --actor a7-review-synthesis
```

It refuses unless `insights.json` is current (tied to the current review). It
writes into the run folder:

- `<project>_prototype.xlsx`: the Excel prototype the brief asks for;
- `insight_readout.md` and `insight_readout.html`: the 1-2 page readout;
- `charts/`: scorecard and per-result charts;
- `deliverables.json`: what was built, tied to the insights by hash.

Changing `insights.json` afterwards makes the deliverables stale, so rebuild them.
Open the readout and check that the headline, insight order, and limitations
match what you saved.
