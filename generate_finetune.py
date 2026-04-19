import json
import pickle
import time
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── constants ─────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You write concise risk summaries for credit union member services reps. "
    "Mention top risk factors. Suggest one action. Plain language."
)

FEATURES = [
    "tenure_years", "product_count", "has_direct_deposit", "login_freq_90d",
    "avg_balance_trend", "nsf_count_6m", "complaints_6m", "cd_maturing_90d",
    "debit_swipe_delta",
]

N_HIGH   = 80   # score > 0.65
N_LOW    = 80   # score < 0.20
N_MEDIUM = 40   # everything else
N_TOTAL  = N_HIGH + N_LOW + N_MEDIUM  # 200
N_VAL    = 40
N_TRAIN  = N_TOTAL - N_VAL           # 160

# ── helpers ───────────────────────────────────────────────────────────────────
def format_profile(row: pd.Series) -> str:
    dd       = "Yes" if row.has_direct_deposit else "No"
    cd       = "Yes" if row.cd_maturing_90d    else "No"
    bal_sign = "+" if row.avg_balance_trend >= 0 else "-"
    swp_sign = "+" if row.debit_swipe_delta  >= 0 else "-"
    return (
        f"Member profile (churn score: {row.churn_score:.3f}):\n"
        f"- Tenure: {row.tenure_years:.1f} years\n"
        f"- Products: {int(row.product_count)}\n"
        f"- Direct deposit: {dd}\n"
        f"- Logins last 90 days: {int(row.login_freq_90d)}\n"
        f"- Avg balance trend: {bal_sign}${abs(row.avg_balance_trend):.0f}\n"
        f"- NSF events last 6 months: {int(row.nsf_count_6m)}\n"
        f"- Complaints last 6 months: {int(row.complaints_6m)}\n"
        f"- CD maturing in 90 days: {cd}\n"
        f"- Debit swipe delta: {swp_sign}{abs(row.debit_swipe_delta):.1f}%"
    )


def generate_narrative(client: anthropic.Anthropic, profile_text: str) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": profile_text}],
    )
    return response.content[0].text.strip()


def make_record(profile_text: str, narrative: str) -> dict:
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": profile_text},
            {"role": "assistant", "content": narrative},
        ]
    }


# ── load data & model ─────────────────────────────────────────────────────────
print("Loading data and model...")
df = pd.read_csv("data/member_features.csv")

with open("data/churn_model.pkl", "rb") as f:
    bundle = pickle.load(f)

model      = bundle["model"]
feat_names = bundle["features"]

# ── score all members ─────────────────────────────────────────────────────────
df["churn_score"] = model.predict_proba(df[feat_names].values)[:, 1]
print(f"Scored {len(df):,} members  |  mean score: {df.churn_score.mean():.3f}")

# ── stratified sample ─────────────────────────────────────────────────────────
high   = df[df.churn_score >  0.65].sample(n=N_HIGH,   random_state=42)
low    = df[df.churn_score <  0.20].sample(n=N_LOW,    random_state=42)
medium = df[
    (df.churn_score >= 0.20) & (df.churn_score <= 0.65)
].sample(n=N_MEDIUM, random_state=42)

sample = pd.concat([high, low, medium]).sample(frac=1, random_state=42).reset_index(drop=True)
print(
    f"Sample: {len(high)} high-risk | {len(low)} low-risk | {len(medium)} medium  "
    f"(total {len(sample)})"
)

# ── generate narratives via Anthropic API ─────────────────────────────────────
client  = anthropic.Anthropic()
records = []
errors  = 0

print(f"\nGenerating {N_TOTAL} narratives with claude-sonnet-4-6 ...")
t0 = time.time()

for i, row in sample.iterrows():
    idx          = len(records) + 1
    profile_text = format_profile(row)

    try:
        narrative = generate_narrative(client, profile_text)
        records.append(make_record(profile_text, narrative))
    except anthropic.APIError as e:
        print(f"  [!] row {idx}: API error — {e}")
        errors += 1
        time.sleep(5)
        continue

    # progress every 20
    if idx % 20 == 0:
        elapsed = time.time() - t0
        print(f"  {idx}/{N_TOTAL}  ({elapsed:.0f}s elapsed)")

    # gentle rate-limit buffer
    time.sleep(0.3)

elapsed = time.time() - t0
print(f"\nDone: {len(records)} narratives generated in {elapsed:.0f}s  ({errors} errors)")

# ── split and save ────────────────────────────────────────────────────────────
val_records   = records[:N_VAL]
train_records = records[N_VAL:]

out = Path("data")
out.mkdir(exist_ok=True)

train_path = out / "finetune_train.jsonl"
val_path   = out / "finetune_val.jsonl"

for path, recs in [(train_path, train_records), (val_path, val_records)]:
    with open(path, "w", encoding="utf-8") as f:
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"\nSaved {len(train_records)} rows -> {train_path}")
print(f"Saved {len(val_records)} rows  -> {val_path}")

# ── quick sanity check ────────────────────────────────────────────────────────
print("\n-- Sample record (train[0]) -----------------------------------------")
first = train_records[0]
print("user:", first["messages"][1]["content"][:200], "...")
print("assistant:", first["messages"][2]["content"])
