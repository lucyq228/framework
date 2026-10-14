# PACE–CSAT Correlation Analysis: project brief

Transcribed from the GSE KPI design-discussion deck and the requirements
slide in this folder (`IMG_2598`–`IMG_2610`). This is the document the scope
agent reads first.

## 1. Why this project exists

McDonald's is designing a **Gold Standard Execution (GSE) KPI**. Each index
gives equal weight to what the restaurant *executed* and what the customer
*experienced*:

| Index | Restaurant execution (50%) | Customer experience (50%) |
|---|---|---|
| Taste Index | PACE Taste: Gold Standard across all 5 products | CSAT Taste: top-two-box taste (quality) satisfaction rate |
| Hospitality Index | PACE Hospitality: Gold Standard across all 4 channels | CSAT Hospitality: top-two-box hospitality (friendliness) satisfaction rate |

The two indexes combine into the **Gold Standard Execution KPI**.

Worked example from the deck:

| Measure | Calculation | Result | Weight | Contribution |
|---|---|---|---|---|
| PACE Taste | 75 passing visits ÷ 100 eligible | 75% | 50% | 37.5 |
| CSAT Taste | 8,000 satisfied ÷ 10,000 valid | 80% | 50% | 40.0 |
| **Taste Index** | | **77.5%** | | |
| PACE Hospitality | 68 ÷ 100 | 68% | 50% | 34.0 |
| CSAT Hospitality | 6,000 ÷ 10,000 | 60% | 50% | 30.0 |
| **Hospitality Index** | | **64.0%** | | |
| **GSE KPI** | mean of the two indexes | **70.8%** | | |

Adding both execution and perception to one KPI only makes sense if we know how they behave together. The deck does not establish that yet, and this project is meant to answer it.

## 2. Requirements (from the "PACE-CSAT Correlation Analysis" slide)

| Item | Requirement |
|---|---|
| Problem statement | Understand how historical PACE execution and CSAT outcomes behave together, including where they align and where they diverge. Explore the relationship between Execution (PACE) and Perception (CSAT) using historical data, while recognising limitations of the available 2026 data. |
| Markets | Canada and UK&I, subject to confirming that localised PACE questions can be mapped to equivalent global questions. |
| Output | Excel-based analytical prototype supported by a short insight readout. |
| Grain of analysis | Start at market/country level. Test more granular geographic or organisational cuts only where the available sample supports meaningful analysis. |
| Time period | 2026, with the option to assess 2025 data if additional observations are required. |
| Scoring | Use **progressive** PACE scores for statistical analysis; compare conservative and progressive outputs. |
| Workflow | Methodology & Scope Lock → PACE Historical Reconstruction → CSAT Data Matching & Validation → Statistical Analysis & Interpretation |
| Timeline | No immediate decision depends on completion. Timing to be confirmed. |

## 3. PACE scoring logic

PACE is an audit programme with two visit types:

- **RGRV (Running Great Restaurant Visit)**: announced, about 1 per year.
  Tests whether the restaurant *knows* the standard. Tests all applicable
  channels. In the data it is `survey_id = 204`, `survey_name = RGR`.
- **CFV (Customer First Visit)**: unannounced, 2–4 per year. Tests what the
  customer *actually gets*. Each visit tests exactly one channel. In the data
  it is `survey_id = 203`, `survey_name = CFV`.

The RGRV–CFV gap is a restaurant-level diagnostic. RGRV and CFV results should
stay separately visible beneath the KPI.

### PACE Taste (product checks)

1. **Product result**: each product is assessed on four dimensions
   (appearance, taste, temperature, texture). A product PASSES only when every
   applicable dimension passes.
2. **Visit result** across the 5 eligible categories (Beef, Chicken, Fries,
   Beverage, Dessert; a Local Core Product may replace an unavailable global
   category and inherits its 20% weight):
   - *Conservative*: PASS only if all eligible products pass.
   - *Progressive*: products passed ÷ eligible products (e.g. 4/5 = 80%).
3. **Restaurant result** across eligible RGRV + CFV visits (each counts equally):
   - *Conservative*: visits passed ÷ eligible visits (e.g. 2/3 = 66.7%).
   - *Progressive*: average visit score (e.g. (100+80+100)/3 = 93.3%).
4. **Taste Index** = 50% PACE Taste + 50% CSAT Taste.

### PACE Hospitality (service checks)

1. **Journey result**: each assessed channel journey is scored on its
   applicable touchpoints (Order, Pay, Present, Care). The journey PASSES only
   when every applicable touchpoint passes. Denominator = touchpoints present
   in the channel (Curbside and Delivery have no Order/Pay).
2. **Visit result** across assessed channels (Drive Thru, In-Restaurant,
   Delivery, Curbside). Delivery is RGRV-only; Curbside is CFV-only. Only
   assessed channels count.
   - *Conservative*: PASS only if all assessed channels pass.
   - *Progressive*: channels passed ÷ channels assessed.
3. **Restaurant result** and **Hospitality Index**: same roll-up and 50/50
   weighting as Taste.

A failed touchpoint fails the whole journey ("not a partial-credit index"), but
the touchpoint detail is kept for coaching.

## 4. Open dependencies called out in the deck

- Markets where a Local Core Product replaces a global category still need
  identification and validation (illustrative treatment only).
- Final KPI aggregation and reporting level are still to be confirmed.
- Localised PACE questions must map to global questions before Canada and UK&I
  can be pooled.

## 5. What the correlation analysis must be careful about

These are the analyst's notes, not part of the deck:

- PACE has very few visits per restaurant per year (about 3–5), so restaurant-level
  PACE scores are coarse and bounded (0–100% in steps of 20–33%).
- CSAT is daily and high-volume, but noisy for low-volume restaurants.
- PACE scores a single moment; CSAT spans a period. Aligning the time windows matters.
- Market-level differences in survey culture can confound pooled correlations.
- The analysis is observational and must not be presented as causal.
