# cu_churn — Credit Union Member Churn Classifier

A two-layer system that scores member churn risk with XGBoost and generates plain-language risk narratives with a fine-tuned LLM, so member services reps get both a probability and an explanation in one place.

---

## Architecture

```
member_features.csv (5,000 rows)
        │
        ▼
┌───────────────────┐
│   data_gen.py     │  Synthetic data — logistic DGP with realistic CU features
└───────┬───────────┘
        │ member_features.csv
        ▼
┌───────────────────┐
│   train_xgb.py    │  XGBoost (300 trees, depth 5) + isotonic calibration
│                   │  Tracks run with MLflow
└───────┬───────────┘
        │ churn_model.pkl
        ├─────────────────────────────────────────┐
        ▼                                         ▼
┌──────────────────────┐              ┌───────────────────────┐
│  generate_finetune.py│              │     evaluate.py       │
│                      │              │  ┌─────────────────┐  │
│  Scores all members  │              │  │ Classifier eval │  │
│  Samples 200:        │              │  │  AUC-ROC, AP,   │  │
│   80 high (>0.65)    │              │  │  Brier, MACE,   │  │
│   80 low  (<0.20)    │              │  │  decile lift,   │  │
│   40 medium          │              │  │  feature imp.   │  │
│                      │              │  └─────────────────┘  │
│  Calls claude-       │              │  ┌─────────────────┐  │
│  sonnet-4-6 to write │              │  │ Narrative eval  │  │
│  2-3 sentence        │              │  │ (LLM-as-judge)  │  │
│  risk narratives     │              │  │ Fine-tuned model│  │
└──────┬───────────────┘              │  │ → Claude scores │  │
       │                              │  │ acc/act/tone    │  │
       │ finetune_train.jsonl (160)   │  └─────────────────┘  │
       │ finetune_val.jsonl   (40)    └───────────────────────┘
       ▼
┌───────────────────┐
│   finetune.py     │  Uploads JSONL, launches OpenAI fine-tune job,
│                   │  polls until done, saves model ID
└───────┬───────────┘
        │ fine_tuned_model.txt
        ▼
┌───────────────────┐
│   predict.py      │  Inference: XGBoost score + fine-tuned narrative
│                   │  for a single member (CLI or imported)
└───────────────────┘
```

### Layer 1 — Calibrated XGBoost classifier

`train_xgb.py` trains an `XGBClassifier` on an 80/20 stratified split, then wraps it in `CalibratedClassifierCV(method="isotonic")` so predicted probabilities are reliable enough to act on. `scale_pos_weight` handles the ~15% class imbalance. All params, AUC-ROC, average precision, and Brier score are logged to MLflow.

### Layer 2 — Fine-tuned narrative LLM

`generate_finetune.py` uses the calibrated model to score all 5,000 members, then calls `claude-sonnet-4-6` to write a 2-3 sentence risk narrative for each of 200 sampled members (80 high / 80 low / 40 medium risk). The resulting JSONL in OpenAI chat format is used to fine-tune `gpt-4o-mini-2024-07-18` via `finetune.py`.

---

## Features

| Feature | Description |
|---|---|
| `tenure_years` | How long the member has been with the CU |
| `product_count` | Number of products held (checking, savings, loan, etc.) |
| `has_direct_deposit` | Whether payroll DD is active |
| `login_freq_90d` | Online/mobile logins in the last 90 days |
| `avg_balance_trend` | $ change in average balance over 90 days |
| `nsf_count_6m` | Non-sufficient funds events in the last 6 months |
| `complaints_6m` | Formal complaints logged in the last 6 months |
| `cd_maturing_90d` | Whether a CD matures within 90 days |
| `debit_swipe_delta` | % change in debit card transaction count |

**Churn label:** `1` = account closed or dormant 180+ days. Synthetic base rate ≈ 15%.

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and fill in API keys
cp .env.example .env

# 3. Generate synthetic data
python data_gen.py

# 4. Train + calibrate the classifier
python train_xgb.py
mlflow ui          # inspect the run at http://localhost:5000

# 5. Build fine-tuning pairs (requires ANTHROPIC_API_KEY)
python generate_finetune.py

# 6. Launch OpenAI fine-tune job (requires OPENAI_API_KEY)
python finetune.py

# 7. Evaluate both layers
python evaluate.py

# 8. Score a single member
python predict.py
```

---

## Evaluation results (example run)

### Classifier

| Metric | Value |
|---|---|
| AUC-ROC | 0.683 |
| Average Precision | 0.292 |
| Brier Score | 0.116 |
| MACE (calibration) | 0.000 — well-calibrated |
| Top-decile lift | 2.43x (36% churn rate vs 14.8% base) |

**Top feature drivers** (XGBoost gain importance):

```
1. complaints_6m        0.178
2. has_direct_deposit   0.153
3. product_count        0.132
4. nsf_count_6m         0.127
5. tenure_years         0.101
6. debit_swipe_delta    0.083
```

### Narratives (LLM-as-judge via Claude)

| Dimension | Mean score (/ 5) |
|---|---|
| Accuracy | 4.85 |
| Actionability | 4.90 |
| Tone | 5.00 |

---

## Project layout

```
cu_churn/
├── data_gen.py            # Synthetic dataset (5,000 members)
├── train_xgb.py           # XGBoost + isotonic calibration, MLflow
├── generate_finetune.py   # Claude-generated JSONL training pairs
├── finetune.py            # OpenAI fine-tune job launcher + poller
├── evaluate.py            # Classifier metrics + LLM-as-judge narrative eval
├── predict.py             # Single-member inference
├── requirements.txt
├── .env.example
└── data/
    ├── member_features.csv
    ├── churn_model.pkl        # not committed
    ├── finetune_train.jsonl   # not committed
    ├── finetune_val.jsonl     # not committed
    └── fine_tuned_model.txt   # not committed
```

---

## Environment variables

| Variable | Used by |
|---|---|
| `ANTHROPIC_API_KEY` | `generate_finetune.py`, `evaluate.py` |
| `OPENAI_API_KEY` | `finetune.py`, `evaluate.py`, `predict.py` |
