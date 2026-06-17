import numpy as np
import pandas as pd

np.random.seed(42)

# ============================================================
# 1. Create sample QSR store-week data
# ============================================================

n_stores = 8
n_weeks = 26

stores = [f"S{i:02d}" for i in range(1, n_stores + 1)]
weeks = pd.date_range("2025-01-06", periods=n_weeks, freq="W-MON")

rows = []

for store in stores:
    store_size_effect = np.random.normal(0, 0.08)

    for w_idx, week in enumerate(weeks):
        season = np.sin(2 * np.pi * w_idx / 52)

        media_spend = np.random.gamma(shape=2.0, scale=900) + 250 * season

        promo_depth_pct = np.clip(
            np.random.normal(0.10 + 0.03 * (w_idx % 7 == 0), 0.03),
            0,
            0.25
        )

        price_index = np.random.normal(
            1.00 + 0.01 * w_idx / n_weeks,
            0.015
        )

        p90_speed_sec = np.random.normal(
            240 - 8 * season + 10 * promo_depth_pct,
            12
        )

        digital_mix_pct = np.clip(
            np.random.normal(0.32 + 0.12 * w_idx / n_weeks, 0.04),
            0.15,
            0.65
        )

        competitor_price_index = np.random.normal(
            1.00 + 0.008 * season,
            0.015
        )

        unemployment_rate = np.random.normal(
            0.045 + 0.002 * np.cos(2 * np.pi * w_idx / 52),
            0.002
        )

        holiday_week = int(w_idx in [2, 15, 25])

        rows.append({
            "store_id": store,
            "week_start": week,
            "week_num": w_idx + 1,
            "media_spend": media_spend,
            "promo_depth_pct": promo_depth_pct,
            "price_index": price_index,
            "p90_speed_sec": p90_speed_sec,
            "digital_mix_pct": digital_mix_pct,
            "competitor_price_index": competitor_price_index,
            "unemployment_rate": unemployment_rate,
            "holiday_week": holiday_week,
            "store_size_effect": store_size_effect,
        })

df = pd.DataFrame(rows)

# ============================================================
# 2. Standardize business driver variables
# ============================================================

domain_vars = [
    "media_spend",
    "promo_depth_pct",
    "price_index",
    "p90_speed_sec",
    "digital_mix_pct",
    "competitor_price_index",
    "unemployment_rate",
    "holiday_week",
]

for col in domain_vars:
    df[f"z_{col}"] = (df[col] - df[col].mean()) / df[col].std()

# ============================================================
# 3. Simulate sales data
# In real work, you would replace this part with actual sales.
# ============================================================

true_betas = {
    "z_media_spend": 0.030,
    "z_promo_depth_pct": 0.040,
    "z_price_index": -0.045,
    "z_p90_speed_sec": -0.025,
    "z_digital_mix_pct": 0.020,
    "z_competitor_price_index": 0.018,
    "z_unemployment_rate": -0.012,
    "z_holiday_week": 0.050,
}

df["season_sin"] = np.sin(2 * np.pi * (df["week_num"] - 1) / 52)
df["season_cos"] = np.cos(2 * np.pi * (df["week_num"] - 1) / 52)

base_log_sales = np.log(85000)
noise = np.random.normal(0, 0.035, size=len(df))

df["log_sales"] = (
    base_log_sales
    + df["store_size_effect"]
    + 0.030 * df["season_sin"]
    - 0.010 * df["season_cos"]
    + sum(true_betas[k] * df[k] for k in true_betas)
    + noise
)

df["sales"] = np.exp(df["log_sales"])

# ============================================================
# 4. Build model matrix
# Target is centered log sales
# ============================================================

y_mean = df["log_sales"].mean()
y = (df["log_sales"] - y_mean).values

# Store fixed effects
store_dummies = pd.get_dummies(
    df["store_id"],
    prefix="store",
    drop_first=True
).astype(float)

base_features = [
    "z_media_spend",
    "z_promo_depth_pct",
    "z_price_index",
    "z_p90_speed_sec",
    "z_digital_mix_pct",
    "z_competitor_price_index",
    "z_unemployment_rate",
    "z_holiday_week",
    "season_sin",
    "season_cos",
]

X_df = pd.concat(
    [df[base_features].astype(float), store_dummies],
    axis=1
)

X = X_df.to_numpy(dtype=float)
feature_names = X_df.columns.tolist()

# ============================================================
# 5. Define priors
# ============================================================

# Prior means: your business belief before seeing current data
prior_mean_map = {
    "z_media_spend": 0.020,              # media should be positive
    "z_promo_depth_pct": 0.030,          # promo should lift sales
    "z_price_index": -0.030,             # higher price should hurt sales
    "z_p90_speed_sec": -0.020,           # slower speed should hurt sales
    "z_digital_mix_pct": 0.010,          # digital adoption helps
    "z_competitor_price_index": 0.010,   # competitor higher price helps us
    "z_unemployment_rate": -0.010,       # weaker macro hurts
    "z_holiday_week": 0.030,             # holidays lift sales
    "season_sin": 0.000,
    "season_cos": 0.000,
}

# Prior standard deviations:
# Larger = weaker prior, smaller = stronger prior
prior_sd_map = {
    "z_media_spend": 0.050,
    "z_promo_depth_pct": 0.050,
    "z_price_index": 0.050,
    "z_p90_speed_sec": 0.050,
    "z_digital_mix_pct": 0.050,
    "z_competitor_price_index": 0.050,
    "z_unemployment_rate": 0.050,
    "z_holiday_week": 0.080,
    "season_sin": 0.100,
    "season_cos": 0.100,
}

# Priors for store fixed effects
for col in store_dummies.columns:
    prior_mean_map[col] = 0.000
    prior_sd_map[col] = 0.150

m0 = np.array([prior_mean_map[f] for f in feature_names], dtype=float)
s0 = np.array([prior_sd_map[f] for f in feature_names], dtype=float)

V0 = np.diag(s0 ** 2)

# ============================================================
# 6. Bayesian posterior calculation
# ============================================================

# Assumed weekly log-sales noise.
# In a real Bayesian model, you can estimate sigma too.
sigma = 0.040

V0_inv = np.linalg.inv(V0)

Vn = np.linalg.inv(
    V0_inv + (X.T @ X) / sigma**2
)

mn = Vn @ (
    V0_inv @ m0 + (X.T @ y) / sigma**2
)

posterior_sd = np.sqrt(np.diag(Vn))

results = pd.DataFrame({
    "feature": feature_names,
    "prior_mean": m0,
    "prior_sd": s0,
    "posterior_mean": mn,
    "posterior_sd": posterior_sd,
    "posterior_lower_95": mn - 1.96 * posterior_sd,
    "posterior_upper_95": mn + 1.96 * posterior_sd,
})

# ============================================================
# 7. Driver-level posterior estimates
# ============================================================

driver_features = list(true_betas.keys())

driver_results = results[
    results["feature"].isin(driver_features)
].copy()

driver_results["approx_pct_impact_per_1sd"] = (
    100 * (np.exp(driver_results["posterior_mean"]) - 1)
)

print("\nPosterior driver estimates:")
print(driver_results.round(4))

# ============================================================
# 8. Predict sales
# ============================================================

df["pred_log_sales"] = y_mean + X @ mn
df["pred_sales"] = np.exp(df["pred_log_sales"])

rmse = np.sqrt(np.mean((df["sales"] - df["pred_sales"]) ** 2))
mape = np.mean(np.abs(df["sales"] - df["pred_sales"]) / df["sales"]) * 100

print("\nModel fit:")
print({
    "RMSE_sales": round(rmse, 2),
    "MAPE_pct": round(mape, 2)
})

# ============================================================
# 9. Contribution estimate
# ============================================================

for f in driver_features:
    beta_post = results.loc[
        results["feature"] == f,
        "posterior_mean"
    ].iloc[0]

    clean_name = f.replace("z_", "")
    df[f"contrib_log_{clean_name}"] = df[f] * beta_post

contrib_cols = [
    c for c in df.columns
    if c.startswith("contrib_log_")
]

contribution_summary = []

for c in contrib_cols:
    domain = c.replace("contrib_log_", "")

    avg_log_contribution = df[c].mean()
    avg_abs_log_contribution = df[c].abs().mean()

    contribution_summary.append({
        "domain": domain,
        "avg_log_contribution": avg_log_contribution,
        "avg_abs_log_contribution": avg_abs_log_contribution,
        "approx_avg_abs_pct_contribution": 100 * (np.exp(avg_abs_log_contribution) - 1),
    })

contribution_summary = pd.DataFrame(contribution_summary).sort_values(
    "approx_avg_abs_pct_contribution",
    ascending=False
)

print("\nContribution summary:")
print(contribution_summary.round(4))

# ============================================================
# 10. Counterfactual impact
# Set each driver to average level, z = 0
# ============================================================

predicted_sales_current = df["pred_sales"].sum()

counterfactual_summary = []

for f in driver_features:
    X_cf = X_df.copy()

    # z = 0 means this driver is set to its average level
    X_cf[f] = 0.0

    pred_cf = np.exp(
        y_mean + X_cf.to_numpy(dtype=float) @ mn
    ).sum()

    incremental_sales_vs_average = predicted_sales_current - pred_cf

    counterfactual_summary.append({
        "driver": f.replace("z_", ""),
        "predicted_sales_current": predicted_sales_current,
        "predicted_sales_if_driver_at_average": pred_cf,
        "incremental_sales_vs_average": incremental_sales_vs_average,
        "incremental_sales_pct_of_predicted": incremental_sales_vs_average / predicted_sales_current * 100,
    })

counterfactual_summary = pd.DataFrame(counterfactual_summary).sort_values(
    "incremental_sales_vs_average",
    ascending=False
)

print("\nCounterfactual impact summary:")
print(counterfactual_summary.round(2))

# ============================================================
# 11. Show sample data
# ============================================================

print("\nSample store-week data:")
print(
    df[
        [
            "store_id",
            "week_start",
            "sales",
            "media_spend",
            "promo_depth_pct",
            "price_index",
            "p90_speed_sec",
            "digital_mix_pct",
            "competitor_price_index",
            "unemployment_rate",
            "holiday_week",
        ]
    ].head(10).round(3)
)
