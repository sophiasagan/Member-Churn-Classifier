import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# -- data ----------------------------------------------------------------------
df = pd.read_csv("data/member_features.csv")

FEATURES = [
    "tenure_years", "product_count", "has_direct_deposit", "login_freq_90d",
    "avg_balance_trend", "nsf_count_6m", "complaints_6m", "cd_maturing_90d",
    "debit_swipe_delta",
]
X = df[FEATURES].values
y = df["churn"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
spw = neg / pos

# -- model params --------------------------------------------------------------
XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=spw,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    early_stopping_rounds=20,
    random_state=42,
    n_jobs=-1,
)

# -- train --------------------------------------------------------------------─
base_clf = XGBClassifier(**XGB_PARAMS)
base_clf.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)

# CalibratedClassifierCV on test set (cv="prefit" = use already-fitted estimator)
calibrated_clf = CalibratedClassifierCV(estimator=FrozenEstimator(base_clf), method="isotonic")
calibrated_clf.fit(X_test, y_test)

# -- evaluate ------------------------------------------------------------------
proba = calibrated_clf.predict_proba(X_test)[:, 1]
preds = (proba >= 0.5).astype(int)

auc_roc    = roc_auc_score(y_test, proba)
avg_prec   = average_precision_score(y_test, proba)
brier      = brier_score_loss(y_test, proba)

# top-decile lift: fraction of churners in top 10% of scores vs. base rate
n_top = max(1, len(proba) // 10)
top_idx = np.argsort(proba)[::-1][:n_top]
top_churn_rate = y_test[top_idx].mean()
base_rate = y_test.mean()
top_decile_lift = top_churn_rate / base_rate

print("\n-- Classification Report (threshold=0.5) -------------------------------")
print(classification_report(y_test, preds, target_names=["retained", "churned"]))

print("-- Key Metrics ----------------------------------------------------------")
print(f"  AUC-ROC          : {auc_roc:.4f}")
print(f"  Avg Precision    : {avg_prec:.4f}")
print(f"  Brier Score      : {brier:.4f}")
print(f"  Top-Decile Lift  : {top_decile_lift:.2f}x  "
      f"(top-10% capture rate: {top_churn_rate:.1%} vs base {base_rate:.1%})")
print(f"  Best iteration   : {base_clf.best_iteration}")
print(f"  scale_pos_weight : {spw:.2f}")

# -- MLflow --------------------------------------------------------------------
mlflow.set_experiment("cu_churn_xgb")
with mlflow.start_run(run_name="xgb_isotonic"):
    log_params = {k: v for k, v in XGB_PARAMS.items() if k != "early_stopping_rounds"}
    log_params["early_stopping_rounds"] = XGB_PARAMS["early_stopping_rounds"]
    log_params["best_iteration"] = base_clf.best_iteration
    log_params["calibration_method"] = "isotonic"
    log_params["train_size"] = len(X_train)
    log_params["test_size"] = len(X_test)
    mlflow.log_params(log_params)

    mlflow.log_metrics({
        "auc_roc": auc_roc,
        "avg_precision": avg_prec,
        "brier_score": brier,
        "top_decile_lift": top_decile_lift,
    })

    mlflow.sklearn.log_model(calibrated_clf, artifact_path="model")
    run_id = mlflow.active_run().info.run_id

print(f"\n-- MLflow ---------------------------------------------------------------")
print(f"  Run ID: {run_id}")
print(f"  Run: mlflow ui  to inspect")

# -- persist ------------------------------------------------------------------─
out_path = Path("data/churn_model.pkl")
with open(out_path, "wb") as f:
    pickle.dump({"model": calibrated_clf, "features": FEATURES}, f)

print(f"\nModel saved to {out_path}")
