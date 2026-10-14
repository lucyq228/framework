# Data dictionary (synthetic)

All data in `data/` is synthetic. It reproduces the schemas of the three
sheets in the "GSE KPI Data Sample" workbook (`data_csat`,
`data_product_sample`, `data_services_sample` screenshots). The generator is
`scripts/generate_synthetic_data.py` (seed 27).

Coverage: 260 restaurants (Canada 110, United Kingdom 120, Ireland 30),
2025-01-01 to 2026-09-30, 1,828 PACE visits (520 RGRV, 1,308 CFV), about
956k CSAT responses.

## Raw extracts (`data/raw/`), the three sources in the screenshots

### `csat_daily.csv`: CSAT counts per restaurant, day, and channel pair

| Column | Meaning |
|---|---|
| `glin` | Restaurant global location ID (12-digit string; keep as text) |
| `terr_cd` | Territory code: 124 Canada, 826 United Kingdom, 372 Ireland |
| `cal_dt` | Calendar date, `mm/dd/yy` text |
| `cust_surv_psnt_pos_chnl_ds` | Channel where the order was **presented**: Front Counter, Table Service, Drive Thru, Curbside |
| `cust_surv_plcd_pos_chnl_ds` | Channel where the order was **placed**: Front Counter, Self Order Kiosk, Mobile, Drive Thru |
| `frnd_high_stsf_cnt_qt` | Top-two-box friendliness responses → **CSAT Hospitality** numerator |
| `frnd_ansr_cnt_qt` | Valid friendliness responses → CSAT Hospitality denominator |
| `qlty_high_stsf_cnt_qt` | Top-two-box quality responses → **CSAT Taste** numerator |
| `qlty_ansr_cnt_qt` | Valid quality responses → CSAT Taste denominator |

A row exists only when at least one response was received. CSAT has no
Delivery channel and no product-level or dimension-level (appearance,
temperature, and so on) breakdown.

### `pace_product_responses.csv`: PACE Taste, one row per product check

| Column | Meaning |
|---|---|
| `glin`, `visit_date` (`mm/dd/yy`), `visit_id`, `survey_id` | Visit keys; `survey_id` 204 = RGRV, 203 = CFV |
| `question_code` | Q9a Beef, Q10a Chicken/Fish, Q15a Fries/Hash Brown, Q19a Beverage, Q20a Dessert (the 5 index categories); Q21a Hot Coffee, Q11a-MC McCafé Food, Q13a-MC McCafé Drink (diagnostic, not in the index); local codes below |
| `question_text` | Question wording |
| `standard_business_measure_id`, `standard_business_measure` | Always 1 / Quality |
| `question_answer` | Yes (meets standard), No, N/A (product unavailable) |
| `question_exception_flag` | 0 pass, 1 fail, -1 not applicable |
| `product_id`, `product_name` | Product checked in that category |
| `points_received`, `points_possible` | 3/3 on pass, 0/3 on fail, 0/0 on N/A |
| `reasons` | JSON-like list. On a fail it names the failed dimension, for example `["Product not meeting McDonald's standard for temperature"]`, or `["Other"]` |

### `pace_service_responses.csv`: PACE Hospitality, one row per service question

| Column | Meaning |
|---|---|
| `glin`, `visit_date`, `visit_id`, `survey_id`, `survey_name` (CFV / RGR) | Visit keys |
| `question_id`, `question_code`, `question_text` | Question. DT* = Drive Thru, IR*/CU4 = In-Restaurant, S20/S4-MC/S12 = Delivery, CB* = Curbside, S1-MC/S8/S11 = general RGRV service |
| `order_touchpoint`, `pay_touchpoint`, `present_touchpoint`, `care_touchpoint` | 0/1 flags for which touchpoint the question belongs to (IR4 is both Order and Pay) |
| `survey_area` | Channel area: General Drive Thru Service / Drive Thru, General In Restaurant Service / In Restaurant, Delivery, Curbside, or Service (general, not channel-specific) |
| `points_received`, `points_possible` | 4/4 Correct, 2/4 Partial, 0/4 Exception, 0/0 N/A |
| `question_answer` | Correct, Partial, Exception, N/A |
| `question_exception_flag` | 0 pass, 1 fail (Exception or Partial), -1 not applicable / channel not tested |
| `reasons` | Failure reason list, for example `["Tone not friendly/polite"]` |

CFV visits list the untested channels as N/A rows, so the assessed channel has to be
inferred from the rows that are not N/A.

## Reference data, which the real project would still need to acquire

| File | Location | Contents | Why it matters |
|---|---|---|---|
| `question_mapping.csv` | `data/reference/` (available) | Local → global question code, mapping status (confirmed / proposed / unmapped) | Needed before Canada and UK&I can be pooled |
| `restaurant_attributes.csv` | `data/acquirable/` (not yet available) | `glin`, market, market group, region, restaurant type, ownership, volume band, drive-thru / table service / delivery / curbside / McCafé flags, open date | Needed for every breakdown beyond market. Not in the three sample sheets |

`data/acquirable/` simulates data that exists elsewhere in the business but has
not been delivered to the project. Agents do not read it. When the user agrees
to acquire a source, `request_data.py` copies it into `data/reference/`, which
changes the data inventory, so feasibility must be re-run.

## Known data-quality traits (deliberate, for the Data Quality agent to find)

- Dates are `mm/dd/yy` strings; `glin` must be read as text.
- About 0.15% of CSAT rows have more top-box responses than answered responses.
- About 0.1% duplicate CSAT rows and 0.2% duplicate product rows.
- 3 restaurants have PACE visits but no CSAT at all; a few restaurants opened in
  2025 and have partial history.
- About 4% of visits have Dessert unavailable (N/A).
- Localised codes appear from 2026: `Q9a-CA`, `Q10a-UK`, `Q27a-CA` (Poutine local core,
  unmapped), `DT5-CA`.
- RGRV-only "Service" area questions are not tied to a channel; how to treat
  them in the Hospitality score is a scope decision.

## Off-limits

`synthetic/` holds the planted ground truth for testing the tools. Agents must not read it.
