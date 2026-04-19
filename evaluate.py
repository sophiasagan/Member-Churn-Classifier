"""
evaluate.py — two-stage evaluation for the cu_churn system
  1. Classifier eval  (metrics + calibration + lift + feature importance)
  2. Narrative eval   (fine-tuned OpenAI model → Claude LLM-as-judge)
"""
import json
import pickle
from pathlib import Path

import anthropic
import numpy as np
import openai
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

FEATURES = [
    "tenure_years", "product_count", "has_direct_deposit", "login_freq_90d",
    "avg_balance_trend", "nsf_count_6m", "complaints_6m", "cd_maturing_90d",
    "debit_swipe_delta",
]
SYSTEM_PROMPT = (
    "You write concise risk summaries for credit union member services reps. "
    "Mention top risk factors. Suggest one action. Plain language."
)
JUDGE_SYSTEM = (
    "You are an expert evaluator of credit union member risk narratives. "
    "Score on three dimensions:\n"
    "- accuracy (1-5): Does the narrative correctly reflect the member's top risk factors?\n"
    "- actionability (1-5): Is the suggested action concrete and useful for a member services rep?\n"
    "- tone (1-5): Is the language clear, professional, and appropriate for member services staff?\n"
    "Respond ONLY with a JSON object, e.g. {\"accuracy\": 4, \"actionability\": 3, \"tone\": 5}. "
    "No explanation."
)
SEP = "-" * 68


# ── shared helpers ─────────────────────────────────────────────────────────────
def format_profile(row: pd.Series, score: float) -> str:
    dd       = "Yes" if row.has_direct_deposit else "No"
    cd       = "Yes" if row.cd_maturing_90d    else "No"
    bal_sign = "+" if row.avg_balance_trend >= 0 else "-"
    swp_sign = "+" if row.debit_swipe_delta  >= 0 else "-"
    return (
        f"Member profile (churn score: {score:.3f}):\n"
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


# ══════════════════════════════════════════════════════════════════════════════
# 1. CLASSIFIER EVAL
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 68}")
print("  CLASSIFIER EVALUATION")
print(f"{'=' * 68}\n")

df = pd.read_csv("data/member_features.csv")
with open("data/churn_model.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
X     = df[FEATURES].values
y     = df["churn"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

proba = model.predict_proba(X_test)[:, 1]

# ── standard metrics ───────────────────────────────────────────────────────────
auc_roc   = roc_auc_score(y_test, proba)
avg_prec  = average_precision_score(y_test, proba)
brier     = brier_score_loss(y_test, proba)

print(f"AUC-ROC           : {auc_roc:.4f}")
print(f"Average Precision : {avg_prec:.4f}")
print(f"Brier Score       : {brier:.4f}  (lower = better; naive = {y_test.mean()*(1-y_test.mean()):.4f})")

# ── calibration curve (10 bins) ────────────────────────────────────────────────
print(f"\n{SEP}")
print("Calibration curve (10 bins):")
print(f"{'Bin':>4}  {'Mean pred':>10}  {'Actual rate':>11}  {'Error':>7}")
print(SEP)

frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="uniform")
cal_errors = np.abs(frac_pos - mean_pred)

for i, (mp, fp, err) in enumerate(zip(mean_pred, frac_pos, cal_errors)):
    flag = "  <-- largest gap" if err == cal_errors.max() else ""
    print(f"{i+1:>4}  {mp:>10.3f}  {fp:>11.3f}  {err:>7.3f}{flag}")

mace = cal_errors.mean()
print(f"\nMean Absolute Calibration Error (MACE): {mace:.4f}")
if mace < 0.05:
    verdict = "Well-calibrated"
elif mace < 0.10:
    verdict = "Moderately calibrated — minor over/under-estimation in some bins"
else:
    verdict = "Poorly calibrated — consider re-running isotonic calibration"
print(f"Verdict: {verdict}")

# ── top decile lift ────────────────────────────────────────────────────────────
print(f"\n{SEP}")
n_top          = max(1, len(proba) // 10)
top_idx        = np.argsort(proba)[::-1][:n_top]
top_churn_rate = y_test[top_idx].mean()
base_rate      = y_test.mean()
lift           = top_churn_rate / base_rate
top_capture    = y_test[top_idx].sum() / y_test.sum()

print(f"Top-Decile Lift : {lift:.2f}x")
print(f"  Churn rate in top 10% of scores : {top_churn_rate:.1%}")
print(f"  Base churn rate                 : {base_rate:.1%}")
print(f"  % of all churners captured      : {top_capture:.1%}")

# ── feature importance ─────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("Top-6 feature drivers (XGBoost gain importance):")
print(f"{'Rank':>4}  {'Feature':<24}  {'Importance':>10}")
print(SEP)

xgb_clf     = model.calibrated_classifiers_[0].estimator.estimator
importances = xgb_clf.feature_importances_
ranked      = sorted(zip(FEATURES, importances), key=lambda t: t[1], reverse=True)

for rank, (feat, imp) in enumerate(ranked[:6], 1):
    bar = "#" * int(imp * 200)
    print(f"{rank:>4}  {feat:<24}  {imp:>10.4f}  {bar}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. NARRATIVE EVAL (LLM-as-judge)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 68}")
print("  NARRATIVE EVALUATION  (LLM-as-judge)")
print(f"{'=' * 68}\n")

model_id_path = Path("data/fine_tuned_model.txt")
if not model_id_path.exists():
    print("data/fine_tuned_model.txt not found — skipping narrative eval.")
    print("Run finetune.py first to produce a fine-tuned model ID.")
    raise SystemExit(0)

ft_model_id = model_id_path.read_text().strip()
print(f"Fine-tuned model : {ft_model_id}")

# ── sample 20 held-out members ─────────────────────────────────────────────────
# use the same test split; stratify across risk tiers for coverage
test_df = df.iloc[
    train_test_split(range(len(df)), test_size=0.20, stratify=y, random_state=42)[1]
].copy().reset_index(drop=True)

test_proba = model.predict_proba(test_df[FEATURES].values)[:, 1]
test_df["churn_score"] = test_proba

rng = np.random.default_rng(7)
high_idx   = np.where(test_proba >  0.65)[0]
low_idx    = np.where(test_proba <  0.20)[0]
medium_idx = np.where((test_proba >= 0.20) & (test_proba <= 0.65))[0]

chosen = np.concatenate([
    rng.choice(high_idx,   size=min(8,  len(high_idx)),   replace=False),
    rng.choice(low_idx,    size=min(8,  len(low_idx)),    replace=False),
    rng.choice(medium_idx, size=min(4,  len(medium_idx)), replace=False),
])
eval_df = test_df.iloc[chosen].reset_index(drop=True)
print(f"Eval members : {len(eval_df)} "
      f"({(eval_df.churn_score > 0.65).sum()} high / "
      f"{((eval_df.churn_score >= 0.20) & (eval_df.churn_score <= 0.65)).sum()} medium / "
      f"{(eval_df.churn_score < 0.20).sum()} low)\n")

# ── generate narratives with fine-tuned model ──────────────────────────────────
oai_client = openai.OpenAI()
ant_client = anthropic.Anthropic()

print(f"{'Mbr':>4}  {'Score':>6}  {'Acc':>4}  {'Act':>4}  {'Tone':>5}  Narrative snippet")
print(SEP)

scores: list[dict] = []

for i, row in eval_df.iterrows():
    profile_text = format_profile(row, row.churn_score)

    # -- generate narrative (fine-tuned model) --
    completion = oai_client.chat.completions.create(
        model=ft_model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": profile_text},
        ],
        max_tokens=200,
        temperature=0.3,
    )
    narrative = completion.choices[0].message.content.strip()

    # -- judge with Claude --
    judge_user = (
        f"Member profile:\n{profile_text}\n\n"
        f"Narrative written by the model:\n{narrative}"
    )
    judge_resp = ant_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=64,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": judge_user}],
    )
    raw_json = judge_resp.content[0].text.strip()

    try:
        dims = json.loads(raw_json)
        acc  = int(dims.get("accuracy",      0))
        act  = int(dims.get("actionability", 0))
        tone = int(dims.get("tone",          0))
    except (json.JSONDecodeError, ValueError):
        acc = act = tone = 0

    scores.append({"accuracy": acc, "actionability": act, "tone": tone})
    snippet = narrative[:55].replace("\n", " ")
    print(f"{i+1:>4}  {row.churn_score:>6.3f}  {acc:>4}  {act:>4}  {tone:>5}  {snippet}...")

# ── aggregate ──────────────────────────────────────────────────────────────────
score_df = pd.DataFrame(scores)
print(f"\n{SEP}")
print("Mean scores across 20 narratives:")
print(f"  Accuracy      : {score_df.accuracy.mean():.2f} / 5")
print(f"  Actionability : {score_df.actionability.mean():.2f} / 5")
print(f"  Tone          : {score_df.tone.mean():.2f} / 5")
print(f"  Overall mean  : {score_df.values.mean():.2f} / 5")
print(f"\nDistribution:")
print(score_df.describe().round(2).to_string())
