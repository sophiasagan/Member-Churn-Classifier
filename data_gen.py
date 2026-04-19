import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N = 5_000

tenure_years       = np.random.exponential(scale=6, size=N).clip(0.1, 40)
product_count      = np.random.choice([1, 2, 3, 4, 5], size=N, p=[0.25, 0.30, 0.25, 0.12, 0.08])
has_direct_deposit = np.random.binomial(1, 0.55, size=N)
login_freq_90d     = np.random.poisson(lam=18, size=N).clip(0, 120)
avg_balance_trend  = np.random.normal(0, 500, size=N)          # $ change over 90d
nsf_count_6m       = np.random.choice([0, 1, 2, 3, 4, 5], size=N, p=[0.65, 0.18, 0.09, 0.04, 0.02, 0.02])
complaints_6m      = np.random.choice([0, 1, 2, 3], size=N, p=[0.85, 0.10, 0.03, 0.02])
cd_maturing_90d    = np.random.binomial(1, 0.20, size=N)
debit_swipe_delta  = np.random.normal(0, 15, size=N)           # % change in swipe count

# logit = b0 + b_tenure*tenure + b_products*products + ...
# Calibrated so sigmoid(logit).mean() ≈ 0.15
logit = (
    -0.10
    - 0.12 * tenure_years
    - 0.40 * product_count
    - 0.60 * has_direct_deposit
    - 0.025 * login_freq_90d
    - 0.0006 * avg_balance_trend
    + 0.35 * nsf_count_6m
    + 0.70 * complaints_6m
    + 0.30 * cd_maturing_90d
    - 0.018 * debit_swipe_delta
)

prob_churn = 1 / (1 + np.exp(-logit))
churn = np.random.binomial(1, prob_churn)

df = pd.DataFrame({
    "tenure_years":       tenure_years.round(2),
    "product_count":      product_count,
    "has_direct_deposit": has_direct_deposit,
    "login_freq_90d":     login_freq_90d,
    "avg_balance_trend":  avg_balance_trend.round(2),
    "nsf_count_6m":       nsf_count_6m,
    "complaints_6m":      complaints_6m,
    "cd_maturing_90d":    cd_maturing_90d,
    "debit_swipe_delta":  debit_swipe_delta.round(2),
    "churn":              churn,
})

out_path = Path("data/member_features.csv")
out_path.parent.mkdir(exist_ok=True)
df.to_csv(out_path, index=False)

print(f"Saved {len(df):,} rows to {out_path}")
print(f"\nChurn rate: {df.churn.mean():.1%}")
print("\nchurn value_counts (normalize=True):")
print(df["churn"].value_counts(normalize=True).rename({0: "retained", 1: "churned"}).round(4))
