"""Generate synthetic PACE and CSAT data for the PACE-CSAT correlation demo.

The three raw extracts mirror the "GSE KPI Data Sample" workbook schemas
(CSAT Sample, Product Sample, Service Sample). Two reference files that the
real project would still need to acquire (restaurant attributes and the
local-to-global question mapping) are generated separately so the scope agent
can reason about what is and is not available.

Planted signals are written to synthetic/ground_truth.json. Agents must never
read that file; it exists so tests can check that the tools recover them.

Usage:
    .venv\\Scripts\\python.exe scripts\\generate_synthetic_data.py [--seed 27]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REF = ROOT / "data" / "reference"
# Sources the project does not have yet; request_data.py "acquires" them into data/reference.
ACQUIRABLE = ROOT / "data" / "acquirable"
TRUTH = ROOT / "synthetic"

START = pd.Timestamp("2025-01-01")
END = pd.Timestamp("2026-09-30")
MONTHS = pd.period_range(START, END, freq="M")

MARKETS = {
    "Canada": dict(
        terr_cd=124, market_group="Canada", n=110,
        regions=[("Ontario", .42), ("Quebec", .24), ("West", .26), ("Atlantic", .08)],
        pace_shift=0.0, csat_shift=0.15, table_share=.25, curbside_share=.45, mccafe_share=.75,
    ),
    "United Kingdom": dict(
        terr_cd=826, market_group="UK&I", n=120,
        regions=[("London", .20), ("South East", .20), ("Midlands", .18), ("North", .24),
                 ("Scotland", .10), ("Wales", .08)],
        pace_shift=0.90, csat_shift=-0.60, table_share=.60, curbside_share=0.0, mccafe_share=.80,
    ),
    "Ireland": dict(
        terr_cd=372, market_group="UK&I", n=30,
        regions=[("Dublin", .45), ("Rest of Ireland", .55)],
        pace_shift=0.70, csat_shift=-0.35, table_share=.50, curbside_share=0.0, mccafe_share=.60,
    ),
}

# restaurant_type: (share, drive-thru probability, PACE hospitality effect, CSAT hospitality effect)
RESTAURANT_TYPES = {
    "Freestanding": (.45, .97, 0.00, 0.00),
    "Urban In-line": (.20, .05, 0.05, 0.05),
    "Mall / Food Court": (.12, .00, -0.05, -0.15),
    "Travel / Motorway": (.10, .90, -0.20, -0.35),
    "Retail Park": (.13, .95, 0.00, 0.00),
}
VOLUME_MEAN = {"Low": 1.5, "Medium": 5.0, "High": 11.0}
MONTHLY_DEV_SD = .60
CSAT_LOCAL_SD = .35
RGR_PREP_MEAN, RGR_PREP_SD = .70, .50
TASTE_SENSITIVITY = .30
MOBILE_SENSITIVITY_REDUCTION = .12

# ---------------------------------------------------------------- PACE product
TASTE_CATEGORIES = {
    # category: (question_code, question_text, products [(id, name)], pass-logit effect)
    "Beef": ("Q9a", "Finished product quality check - Main menu Beef / Breakfast Item 1: Sandwich meets McDonald's standards for appearance, taste, temperature and texture",
             [(1, "Big Mac"), (2, "Quarter Pounder"), (9, "Double Cheeseburger"), (12, "Sausage Egg McMuffin")], 0.0),
    "Chicken": ("Q10a", "Finished product quality check - Main menu Chicken or Fish / Breakfast Item 2: Sandwich/Nuggets meet McDonald's standards",
                [(5, "McNuggets"), (6, "McChicken"), (4, "Filet")], -0.2),
    "Fries": ("Q15a", "Finished product quality check - Fries / Hash Brown:", [(20, "Fries"), (21, "Hash Brown")], -0.5),
    "Beverage": ("Q19a", "Finished product quality check - Soft Drink/Shake/Blended Ice Beverage Product:",
                 [(53, "Soft Drink"), (55, "Milkshake")], 0.6),
    "Dessert": ("Q20a", "Finished product quality check - Dessert:", [(39, "McFlurry"), (40, "Sundae")], -0.4),
}
DIAGNOSTIC_CATEGORIES = {
    "Hot Coffee": ("Q21a", "Finished product quality check - Hot Coffee / Espresso Product:",
                   [(29, "Espresso"), (31, "Cappuccino"), (26, "Latte")], 0.2),
    "McCafe Food": ("Q11a-MC", "McCafé Food Product 1 (ambient air product) : Does the McCafé product meet standards? If not, check why.",
                    [(25, "Croissant"), (27, "Muffin")], 0.1),
    "McCafe Drink": ("Q13a-MC", "McCafé Drink : (Espresso or Long Black) Does the McCafé product meet standards? If not, check why:",
                     [(29, "Espresso"), (54, "Iced Latte")], 0.0),
}
FAIL_REASONS = {
    "Beef": {"appearance": .35, "temperature": .30, "taste": .15, "texture": .10, "Other": .10},
    "Chicken": {"temperature": .35, "texture": .30, "appearance": .15, "taste": .10, "Other": .10},
    "Fries": {"temperature": .50, "texture": .30, "taste": .10, "appearance": .05, "Other": .05},
    "Beverage": {"taste": .45, "temperature": .30, "appearance": .15, "Other": .10},
    "Dessert": {"texture": .40, "appearance": .35, "temperature": .10, "Other": .15},
    "Hot Coffee": {"temperature": .45, "taste": .35, "appearance": .10, "Other": .10},
    "McCafe Food": {"appearance": .45, "texture": .35, "Other": .20},
    "McCafe Drink": {"taste": .45, "temperature": .35, "Other": .20},
    "Local Core": {"temperature": .40, "texture": .30, "taste": .20, "Other": .10},
}

# --------------------------------------------------------------- PACE service
SERVICE_QUESTIONS = {
    # channel: [(question_id, question_code, touchpoints, text)]
    "Drive Thru": [
        (1429, "DT3", ("order",), "Order: Do crew interact in a prompt, friendly and effective manner, asking if the guest is using their McDonald's Mobile App?"),
        (1515, "DT4", ("pay",), "Pay: Did crew interact in a polite, friendly and effective manner and provide clear instructions?"),
        (1567, "DT5", ("present",), "Present: Was Presenter friendly and effective and did they provide clear instructions? If pulled forward, were you provided with clear instructions?"),
        (1580, "DT8", ("care",), "Care: Did crew thank you and offer a genuine farewell at the window?"),
    ],
    "In Restaurant": [
        (1346, "IR4", ("order", "pay"), "Order & Pay: Was Order Taker friendly, effective and did they provide clear instructions? Kiosk orders: Was the Kiosk in full working order and was help offered?"),
        (1436, "CU4", ("present",), "Present: Was Presenter friendly, wearing a condiment apron and asked if you have everything you need?"),
        (1433, "IR5", ("present",), "Present: Was Presenter friendly and effective? Table Service orders: Was Presenter wearing a condiment apron and asked if you have everything you need?"),
        (1569, "IR6", ("care",), "Dining Area: Did you experience genuine hospitality on the dining area?"),
    ],
    "Delivery": [
        (6287, "S20", ("present",), "Present: Receipt or pick ticket attached to bag and used to visibly check order number on courier device. Crew are courteous to the courier."),
        (6051, "S4-MC", ("present",), "Accuracy: Are condiments, napkins and straws/stirrers stocked and received throughout the shift? If not, check why."),
        (6135, "S12", ("care",), "Farewell: Guest thanked and provided a genuine/personalized parting comment. If not, check why."),
    ],
    "Curbside": [
        (7120, "CB2", ("present",), "Present: Was the order brought to the bay promptly with a friendly greeting?"),
        (7124, "CB3", ("care",), "Care: Did crew confirm the order and offer a genuine farewell?"),
    ],
}
GENERAL_SERVICE = [
    (5570, "S1-MC", ("care",), "Hospitality: Is the restaurant actively focused on building McCafé business results through Hospitality/interacting with the Customer?"),
    (5546, "S8", ("care",), "Interaction: Did crew and managers interact with customers in a polite and friendly manner (smile, eye contact, friendly tone, polite)?"),
    (6045, "S11", ("present",), "Present: Full order presented, employees friendly and attentive. If not, check why."),
]
SURVEY_AREA = {
    ("CFV", "Drive Thru"): "General Drive Thru Service", ("RGR", "Drive Thru"): "Drive Thru",
    ("CFV", "In Restaurant"): "General In Restaurant Service", ("RGR", "In Restaurant"): "In Restaurant",
    ("CFV", "Delivery"): "Delivery", ("RGR", "Delivery"): "Delivery",
    ("CFV", "Curbside"): "Curbside", ("RGR", "Curbside"): "Curbside",
}
CHANNEL_EFFECT = {"Drive Thru": -0.35, "In Restaurant": 0.0, "Delivery": -0.20, "Curbside": 0.10}
TOUCHPOINT_EFFECT = {"order": 0.10, "pay": 0.30, "present": -0.10, "care": -0.35}
SERVICE_REASONS = {
    "order": ["Order not repeated back", "No mobile app prompt", "Tone not friendly/polite"],
    "pay": ["Unclear payment instructions", "Tone not friendly/polite"],
    "present": ["Table Service presenter not wearing condiment apron", "Did not ask if guest had everything needed",
                "Order incomplete at hand-off", "No pull-forward instructions"],
    "care": ["Not attentive to guests", "No thank you / farewell", "Guest Experience Leader not scheduled/in place",
             "Dining area not maintained"],
}

# ----------------------------------------------------------------------- CSAT
CSAT_COMBOS = [
    # (presented channel, placed channel, base weight, requires)
    ("Front Counter", "Front Counter", .18, None),
    ("Front Counter", "Self Order Kiosk", .20, None),
    ("Front Counter", "Mobile", .22, None),
    ("Table Service", "Self Order Kiosk", .08, "table"),
    ("Table Service", "Mobile", .06, "table"),
    ("Table Service", "Front Counter", .02, "table"),
    ("Drive Thru", "Drive Thru", .14, "dt"),
    ("Drive Thru", "Mobile", .08, "dt"),
    ("Curbside", "Mobile", .02, "curbside"),
]
HOSP_SENSITIVITY = {"Drive Thru": .45, "Front Counter": .30, "Table Service": .30, "Curbside": .20}
PRESENTED_FRND = {"Drive Thru": -.25, "Front Counter": 0.0, "Table Service": .20, "Curbside": .10}
PLACED_FRND = {"Mobile": .15, "Self Order Kiosk": .05, "Front Counter": 0.0, "Drive Thru": 0.0}
PLACED_QLTY = {"Mobile": .10, "Self Order Kiosk": .05, "Front Counter": 0.0, "Drive Thru": -0.05}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fmt_date(d: pd.Timestamp) -> str:
    return d.strftime("%m/%d/%y")


def pick(rng, items, probs=None):
    return items[rng.choice(len(items), p=probs)]


def build_restaurants(rng) -> pd.DataFrame:
    rows, n_global = [], 0
    type_names = list(RESTAURANT_TYPES)
    type_p = np.array([RESTAURANT_TYPES[t][0] for t in type_names])
    for market, cfg in MARKETS.items():
        regions, region_p = zip(*cfg["regions"])
        for _ in range(cfg["n"]):
            n_global += 1
            rtype = pick(rng, type_names, type_p)
            has_dt = rng.random() < RESTAURANT_TYPES[rtype][1]
            vol_p = [.2, .45, .35] if rtype in ("Freestanding", "Travel / Motorway") else [.35, .45, .20]
            opened_recently = rng.random() < .03
            open_date = (START + pd.Timedelta(days=int(rng.integers(60, 330)))) if opened_recently else \
                pd.Timestamp(year=int(rng.integers(1975, 2024)), month=int(rng.integers(1, 13)), day=1)
            rows.append(dict(
                glin=f"1955{100000 + n_global * 7:08d}",
                terr_cd=cfg["terr_cd"], market=market, market_group=cfg["market_group"],
                region=pick(rng, list(regions), np.array(region_p)),
                restaurant_type=rtype,
                ownership_type="Company Operated" if rng.random() < .15 else "Franchised",
                volume_band=pick(rng, ["Low", "Medium", "High"], vol_p),
                has_drive_thru=int(has_dt),
                has_table_service=int(rng.random() < cfg["table_share"]),
                has_delivery=int(rng.random() < .85),
                has_curbside=int(has_dt and rng.random() < cfg["curbside_share"]),
                has_mccafe=int(rng.random() < cfg["mccafe_share"]),
                open_date=open_date.strftime("%Y-%m-%d"),
            ))
    return pd.DataFrame(rows)


def build_latent(rng, n: int):
    """Return (capability, monthly execution); both are [taste, hosp] latents."""
    base = rng.multivariate_normal([0, 0], [[1, .45], [.45, 1]], size=n)
    phi, sd = .6, MONTHLY_DEV_SD
    dev = np.zeros((n, len(MONTHS), 2))
    dev[:, 0] = rng.normal(0, sd, (n, 2))
    for t in range(1, len(MONTHS)):
        dev[:, t] = phi * dev[:, t - 1] + rng.normal(0, sd * np.sqrt(1 - phi ** 2), (n, 2))
    return base, base[:, None, :] + dev


def visit_schedule(rng, open_date: pd.Timestamp):
    visits = []
    for year, last_month, n_cfv_opts in ((2025, 12, [2, 3, 4]), (2026, 9, [1, 2, 3])):
        lo = max(pd.Timestamp(year=year, month=1, day=15), open_date + pd.Timedelta(days=30))
        hi = pd.Timestamp(year=year, month=last_month, day=28)
        if lo >= hi:
            continue
        span = (hi - lo).days
        visits.append(("RGR", lo + pd.Timedelta(days=int(rng.integers(0, span)))))
        for _ in range(int(rng.choice(n_cfv_opts, p=[.25, .5, .25]))):
            visits.append(("CFV", lo + pd.Timedelta(days=int(rng.integers(0, span)))))
    return sorted(visits, key=lambda v: v[1])


def month_index(d: pd.Timestamp) -> int:
    return (d.year - START.year) * 12 + d.month - START.month


def reason_list(rng, category: str) -> str:
    dist = FAIL_REASONS[category]
    r = pick(rng, list(dist), np.array(list(dist.values())))
    if r == "Other":
        return '["Other"]'
    return f'["Product not meeting McDonald\'s standard for {r}"]'


def build_pace(rng, rest: pd.DataFrame, capability: np.ndarray, latent: np.ndarray):
    product_rows, service_rows, visit_truth = [], [], []
    visit_id = 8_200_000
    prep = rng.normal(RGR_PREP_MEAN, RGR_PREP_SD, len(rest))
    for i, r in rest.iterrows():
        cfg = MARKETS[r.market]
        type_pace_h = RESTAURANT_TYPES[r.restaurant_type][2]
        own_h = .10 if r.ownership_type == "Company Operated" else 0.0
        for survey_name, vdate in visit_schedule(rng, pd.Timestamp(r.open_date)):
            visit_id += int(rng.integers(1, 40))
            survey_id = 204 if survey_name == "RGR" else 203
            m = month_index(vdate)
            if survey_name == "RGR":
                zt, zh = capability[i]
                announced = prep[i]
            else:
                zt, zh = latent[i, m]
                announced = 0.0
            visit_noise = rng.normal(0, .3)
            base = dict(glin=r.glin, visit_date=fmt_date(vdate), visit_id=visit_id, survey_id=survey_id)
            visit_truth.append(dict(glin=r.glin, visit_id=visit_id, visit_month=str(MONTHS[m]),
                                    survey_name=survey_name, z_taste=round(zt, 4), z_hosp=round(zh, 4)))

            # ---- product (taste) checks
            local_2026 = vdate.year == 2026
            cats = list(TASTE_CATEGORIES.items()) + list(DIAGNOSTIC_CATEGORIES.items())[:1]
            if r.has_mccafe:
                cats += list(DIAGNOSTIC_CATEGORIES.items())[1:]
            for cat, (qcode, qtext, products, eff) in cats:
                pid, pname = products[rng.integers(len(products))]
                if local_2026 and r.market == "Canada" and cat == "Beef" and rng.random() < .35:
                    qcode, qtext = "Q9a-CA", qtext + " (Canada local questionnaire)"
                if local_2026 and r.market_group == "UK&I" and cat == "Chicken":
                    qcode, qtext = "Q10a-UK", "UK&I local: Chicken / Veggie main item meets standards for appearance, taste, temperature and texture"
                    if rng.random() < .25:
                        pid, pname = 61, "Veggie Dippers"
                row = dict(base, question_code=qcode, question_text=qtext,
                           standard_business_measure_id=1, standard_business_measure="Quality",
                           product_id=pid, product_name=pname, points_possible=3)
                if cat == "Dessert" and rng.random() < .04:
                    product_rows.append(dict(row, question_answer="N/A", question_exception_flag=-1,
                                             points_received=0, points_possible=0, reasons='["Product unavailable"]'))
                    continue
                logit = 1.9 + .95 * zt + cfg["pace_shift"] + announced + eff + visit_noise
                passed = rng.random() < sigmoid(logit)
                product_rows.append(dict(row, question_answer="Yes" if passed else "No",
                                         question_exception_flag=0 if passed else 1,
                                         points_received=3 if passed else 0,
                                         reasons="[null]" if passed else reason_list(rng, cat)))
            if r.market == "Canada" and local_2026 and rng.random() < .4:
                logit = 1.9 + .95 * zt + announced + visit_noise
                passed = rng.random() < sigmoid(logit)
                product_rows.append(dict(base, question_code="Q27a-CA",
                                         question_text="Local Core Product (Canada): Poutine meets McDonald's standards",
                                         standard_business_measure_id=1, standard_business_measure="Quality",
                                         product_id=70, product_name="Poutine", points_possible=3,
                                         question_answer="Yes" if passed else "No",
                                         question_exception_flag=0 if passed else 1,
                                         points_received=3 if passed else 0,
                                         reasons="[null]" if passed else reason_list(rng, "Local Core")))

            # ---- service (hospitality) checks
            available = ["In Restaurant"] + (["Drive Thru"] if r.has_drive_thru else [])
            if survey_name == "RGR":
                tested = available + (["Delivery"] if r.has_delivery else [])
            else:
                options = available + (["Curbside"] if r.has_curbside else [])
                w = np.array([{"Drive Thru": .55, "In Restaurant": .35, "Curbside": .10}[c] for c in options])
                tested = [pick(rng, options, w / w.sum())]
            listed = sorted(set(tested) | set(available), key=list(SERVICE_QUESTIONS).index)
            for channel in listed:
                for qid, qcode, tps, qtext in SERVICE_QUESTIONS[channel]:
                    if local_2026 and r.market == "Canada" and qcode == "DT5":
                        qcode = "DT5-CA"
                    flags = {f"{t}_touchpoint": int(t in tps) for t in ("order", "pay", "present", "care")}
                    row = dict(base, survey_name=survey_name, question_id=qid, question_code=qcode,
                               question_text=qtext, **flags, survey_area=SURVEY_AREA[(survey_name, channel)],
                               points_possible=4)
                    not_applicable = channel not in tested or (qcode == "IR5" and not r.has_table_service)
                    if not_applicable:
                        service_rows.append(dict(row, points_received=0, points_possible=0, question_answer="N/A",
                                                 question_exception_flag=-1, reasons="[null]"))
                        continue
                    logit = (2.3 + 1.0 * zh + cfg["pace_shift"] + announced + CHANNEL_EFFECT[channel]
                             + TOUCHPOINT_EFFECT[tps[0]] + type_pace_h + own_h + visit_noise)
                    if rng.random() < sigmoid(logit):
                        service_rows.append(dict(row, points_received=4, question_answer="Correct",
                                                 question_exception_flag=0, reasons="[null]"))
                    else:
                        partial = rng.random() < .3
                        reason = pick(rng, SERVICE_REASONS[tps[0]])
                        service_rows.append(dict(row, points_received=2 if partial else 0,
                                                 question_answer="Partial" if partial else "Exception",
                                                 question_exception_flag=1, reasons=f'["{reason}"]'))
            if survey_name == "RGR":
                for qid, qcode, tps, qtext in GENERAL_SERVICE:
                    flags = {f"{t}_touchpoint": int(t in tps) for t in ("order", "pay", "present", "care")}
                    logit = 2.3 + 1.0 * zh + cfg["pace_shift"] + announced + TOUCHPOINT_EFFECT[tps[0]] + visit_noise
                    passed = rng.random() < sigmoid(logit)
                    service_rows.append(dict(base, survey_name=survey_name, question_id=qid, question_code=qcode,
                                             question_text=qtext, **flags, survey_area="Service",
                                             points_received=4 if passed else 0, points_possible=4,
                                             question_answer="Correct" if passed else "Exception",
                                             question_exception_flag=0 if passed else 1,
                                             reasons="[null]" if passed else f'["{pick(rng, SERVICE_REASONS[tps[0]])}"]'))

    product_cols = ["glin", "visit_date", "visit_id", "survey_id", "question_code", "question_text",
                    "standard_business_measure_id", "standard_business_measure", "question_answer",
                    "question_exception_flag", "product_id", "product_name", "points_received",
                    "points_possible", "reasons"]
    service_cols = ["glin", "visit_date", "visit_id", "survey_id", "survey_name", "question_id", "question_code",
                    "question_text", "order_touchpoint", "pay_touchpoint", "present_touchpoint", "care_touchpoint",
                    "survey_area", "points_received", "points_possible", "question_answer",
                    "question_exception_flag", "reasons"]
    return (pd.DataFrame(product_rows)[product_cols], pd.DataFrame(service_rows)[service_cols],
            pd.DataFrame(visit_truth))


def build_csat(rng, rest: pd.DataFrame, latent: np.ndarray, skip_glins: set) -> pd.DataFrame:
    days = pd.date_range(START, END, freq="D")
    m_idx = np.array([month_index(d) for d in days])
    month = days.month.values
    season_f = np.where(np.isin(month, [7, 8]), -.12, 0.0) + np.where(month == 12, -.08, 0.0)
    season_q = np.where(np.isin(month, [1, 2]), -.05, 0.0)
    trend = np.where(days.year.values == 2026, .05, 0.0)
    weekend = days.dayofweek.values >= 5
    frames = []
    for i, r in rest.iterrows():
        if r.glin in skip_glins:
            continue
        cfg = MARKETS[r.market]
        open_mask = days >= pd.Timestamp(r.open_date) + pd.Timedelta(days=14)
        combos = [c for c in CSAT_COMBOS if c[3] is None
                  or (c[3] == "table" and r.has_table_service)
                  or (c[3] == "dt" and r.has_drive_thru)
                  or (c[3] == "curbside" and r.has_curbside)]
        w = np.array([c[2] for c in combos])
        w = w / w.sum()
        lam = VOLUME_MEAN[r.volume_band] * np.where(weekend, 1.2, 1.0) * open_mask
        total = rng.poisson(lam)
        counts = np.array([rng.multinomial(n, w) for n in total])  # (days, combos)
        zt = latent[i, m_idx, 0]
        zh = latent[i, m_idx, 1]
        type_csat_h = RESTAURANT_TYPES[r.restaurant_type][3]
        local_f, local_q = rng.normal(0, CSAT_LOCAL_SD, 2)
        for j, (presented, placed, _, _) in enumerate(combos):
            n = counts[:, j]
            keep = n > 0
            if not keep.any():
                continue
            n = n[keep]
            beta = HOSP_SENSITIVITY[presented] - (MOBILE_SENSITIVITY_REDUCTION if placed == "Mobile" else 0.0)
            zh_k, zt_k = zh[keep], zt[keep]
            day_noise = rng.normal(0, .15, keep.sum())
            p_f = sigmoid(.85 + cfg["csat_shift"] + local_f + beta * zh_k + .6 * np.minimum(zh_k + .9, 0)
                          + PRESENTED_FRND[presented] + PLACED_FRND[placed] + type_csat_h
                          + season_f[keep] + trend[keep] - .05 * weekend[keep] + day_noise)
            p_q = sigmoid(1.25 + cfg["csat_shift"] + local_q + TASTE_SENSITIVITY * zt_k + PLACED_QLTY[placed]
                          + season_q[keep] + trend[keep] + rng.normal(0, .15, keep.sum()))
            frnd_ansr = n - rng.binomial(n, .05)
            qlty_ansr = n - rng.binomial(n, .03)
            frames.append(pd.DataFrame(dict(
                glin=r.glin, terr_cd=r.terr_cd, cal_dt=days[keep],
                cust_surv_psnt_pos_chnl_ds=presented, cust_surv_plcd_pos_chnl_ds=placed,
                frnd_high_stsf_cnt_qt=rng.binomial(frnd_ansr, p_f), frnd_ansr_cnt_qt=frnd_ansr,
                qlty_high_stsf_cnt_qt=rng.binomial(qlty_ansr, p_q), qlty_ansr_cnt_qt=qlty_ansr,
            )))
    csat = pd.concat(frames, ignore_index=True)
    csat = csat.sort_values(["glin", "cal_dt", "cust_surv_psnt_pos_chnl_ds", "cust_surv_plcd_pos_chnl_ds"])
    csat["cal_dt"] = csat["cal_dt"].dt.strftime("%m/%d/%y")
    return csat.reset_index(drop=True)


def inject_quality_issues(rng, csat, product, service):
    issues = {}
    idx = rng.choice(len(csat), size=int(len(csat) * .0015), replace=False)
    csat.loc[idx, "frnd_high_stsf_cnt_qt"] = csat.loc[idx, "frnd_ansr_cnt_qt"] + 1
    issues["csat_high_gt_answered_rows"] = int(len(idx))
    dup = csat.sample(frac=.001, random_state=int(rng.integers(1e6)))
    csat = pd.concat([csat, dup]).sort_index(kind="stable").reset_index(drop=True)
    issues["csat_duplicate_rows"] = int(len(dup))
    pdup = product.sample(frac=.002, random_state=int(rng.integers(1e6)))
    product = pd.concat([product, pdup]).sort_index(kind="stable").reset_index(drop=True)
    issues["product_duplicate_rows"] = int(len(pdup))
    return csat, product, service, issues


def write_reference(rest: pd.DataFrame):
    REF.mkdir(parents=True, exist_ok=True)
    ACQUIRABLE.mkdir(parents=True, exist_ok=True)
    stale = REF / "restaurant_attributes.csv"
    if stale.exists():
        stale.unlink()
    rest.to_csv(ACQUIRABLE / "restaurant_attributes.csv", index=False)
    mapping = pd.DataFrame([
        ("Q9a-CA", "Q9a", "Product", "Beef", "confirmed", "Canada localised wording of the global beef check"),
        ("Q10a-UK", "Q10a", "Product", "Chicken", "proposed", "UK&I chicken/veggie item; Veggie Dippers inclusion pending validation"),
        ("Q27a-CA", "", "Product", "Local Core", "unmapped", "Local core product; no global equivalent. Weighting rule pending exception-market validation"),
        ("DT5-CA", "DT5", "Service", "Drive Thru present", "confirmed", "Canada bilingual wording of DT5"),
    ], columns=["local_question_code", "global_question_code", "survey_type", "category", "mapping_status", "notes"])
    mapping.to_csv(REF / "question_mapping.csv", index=False)


def write_sample_workbook(csat, product, service):
    path = ROOT / "data" / "GSE_KPI_Data_Sample_synthetic.xlsx"
    with pd.ExcelWriter(path) as xw:
        csat.head(400).to_excel(xw, sheet_name="CSAT Sample", index=False)
        product.head(400).to_excel(xw, sheet_name="Product Sample", index=False)
        service.head(400).to_excel(xw, sheet_name="Service Sample", index=False)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=27)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    for d in (RAW, REF, ACQUIRABLE, TRUTH):
        d.mkdir(parents=True, exist_ok=True)

    rest = build_restaurants(rng)
    capability, latent = build_latent(rng, len(rest))
    product, service, visit_truth = build_pace(rng, rest, capability, latent)
    no_csat = set(rng.choice(rest.glin.values, size=3, replace=False))
    csat = build_csat(rng, rest, latent, no_csat)
    csat, product, service, issues = inject_quality_issues(rng, csat, product, service)

    csat.to_csv(RAW / "csat_daily.csv", index=False)
    product.to_csv(RAW / "pace_product_responses.csv", index=False)
    service.to_csv(RAW / "pace_service_responses.csv", index=False)
    write_reference(rest)
    sample = write_sample_workbook(csat, product, service)
    visit_truth.to_csv(TRUTH / "visit_latent_truth.csv", index=False)

    truth = dict(
        seed=args.seed, period=[str(START.date()), str(END.date())],
        note="Planted signals for tool validation. Agents must not read this folder.",
        latent=dict(taste_hosp_restaurant_corr=.45, monthly_ar1_phi=.6, monthly_dev_sd=MONTHLY_DEV_SD,
                    csat_local_unmeasured_sd=CSAT_LOCAL_SD),
        pace_effects=dict(market_shift={k: v["pace_shift"] for k, v in MARKETS.items()},
                          rgrv="scores capability (no monthly deviation) plus restaurant prep bonus "
                               f"N({RGR_PREP_MEAN}, {RGR_PREP_SD}); CFV scores the visit month's execution",
                          product_category=
                          {k: v[3] for k, v in {**TASTE_CATEGORIES, **DIAGNOSTIC_CATEGORIES}.items()},
                          channel=CHANNEL_EFFECT, touchpoint=TOUCHPOINT_EFFECT),
        csat_effects=dict(market_shift={k: v["csat_shift"] for k, v in MARKETS.items()},
                          hosp_sensitivity_by_presented_channel=HOSP_SENSITIVITY,
                          mobile_placed_sensitivity_reduction=MOBILE_SENSITIVITY_REDUCTION,
                          hosp_threshold="extra slope 0.6 below latent -0.9 (nonlinear drop)",
                          taste_sensitivity=TASTE_SENSITIVITY,
                          seasonality="hospitality -0.12 Jul-Aug, -0.08 Dec; taste -0.05 Jan-Feb",
                          restaurant_type_hosp={k: v[3] for k, v in RESTAURANT_TYPES.items()}),
        expected_findings=[
            "Simpson's paradox: UK&I has higher PACE but lower CSAT, so pooled cross-market correlation is "
            "attenuated or reversed while within-market correlation is positive.",
            "CSAT measured in the same month as the PACE visit correlates better than annual CSAT (time alignment).",
            "Announced RGRV pass rates are inflated by a restaurant-specific prep bonus; the RGRV-CFV gap varies "
            "by restaurant and RGRV tracks same-month CSAT slightly worse than CFV.",
            "Hospitality correlation is strongest for Drive Thru presented orders and weakest for Mobile-placed orders.",
            "Low-volume restaurants have few monthly CSAT responses; reliability weighting or a minimum-n rule is needed.",
            "Hospitality CSAT drops non-linearly for the weakest executors (threshold below the latent -0.9 point).",
            "Travel / Motorway restaurants have lower CSAT hospitality that PACE does not explain (divergence).",
            "About 35% of restaurant-level CSAT variance comes from local factors PACE does not measure.",
        ],
        data_quality_issues=dict(issues, restaurants_without_csat=sorted(no_csat),
                                 dessert_unavailable_rate=.04,
                                 local_question_codes=["Q9a-CA", "Q10a-UK", "Q27a-CA", "DT5-CA"],
                                 date_format="mm/dd/yy strings",
                                 general_service_area="RGR-only 'Service' questions are not channel-specific"),
    )
    (TRUTH / "ground_truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")

    n_visits = product.visit_id.nunique()
    print(json.dumps(dict(
        restaurants=len(rest), pace_visits=n_visits,
        visits_by_type=service.drop_duplicates("visit_id").survey_name.value_counts().to_dict(),
        product_rows=len(product), service_rows=len(service), csat_rows=len(csat),
        csat_responses=int(csat.qlty_ansr_cnt_qt.sum()), sample_workbook=str(sample.relative_to(ROOT)),
    ), indent=2))


if __name__ == "__main__":
    main()
