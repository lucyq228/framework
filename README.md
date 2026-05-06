## Summary

This project identifies stable, interpretable drivers of changes in guest count (GC), average check (AC), and sales at a store-week level.

The modeling framework is not designed to find one “best” model. Instead, it runs many structured regression experiments across reasonable modeling choices to evaluate which drivers remain stable across different specifications.

Experiments vary across:
- feature domains: menu, promo, pricing, speed, CSAT, macro, media, loyalty
- panel control strategies: Fixed Effects, Mundlak / CRE, HLM, pooled models
- algorithms: OLS, Ridge, Elastic Net
- time splits and evaluation windows
- target metrics: GC, AC, and derived sales

For each experiment, the system logs:
- selected features
- model type and algorithm
- coefficients
- signs
- statistical diagnostics where applicable
- model fit metrics for sanity checks

Feature stability is evaluated based on:
- how often a feature appears across experiments
- sign consistency
- coefficient stability
- statistical significance where applicable
- driver rank and contribution consistency

Feature impact is calculated using both:
1. the model-estimated coefficient, and  
2. the observed change in the feature over a defined time window, such as YoY or QoQ.

Conceptually:

```text
Feature contribution = coefficient × feature change
