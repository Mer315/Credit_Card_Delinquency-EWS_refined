"""
compare_pipelines.py
────────────────────
Runs the ORIGINAL (leaky) pipeline and the REFACTORED (clean) pipeline
side-by-side, then prints a metrics comparison table.

Run from the project root:
    python compare_pipelines.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, roc_curve,
    precision_score, recall_score, f1_score,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = r"C:/Users/ssume/Downloads/Credit_Card_Delinquency-EWS_refined/datasets"
F01 = f"{BASE}/File01_Delinquency_ews_Model.csv"
F02 = f"{BASE}/File02_Delinquency_ews_20k_1_Test_Model.csv"
F03 = f"{BASE}/File03_Delinquency_ews_20k_2_Bus_Validate.csv"

# ── Feature engineering (identical for both pipelines) ───────────────────────
LEAKY = ["current_dpd", "recovery_lag_days", "behavioral_risk_score"]
CAT_COLS = [
    "employment_type", "marital_status", "loan_type",
    "income_variability_flag", "partial_payment_flag", "utilization_shock_flag",
]
MAPD = {"Yes": 1, "No": 0, 1: 1, 0: 0}


def add_features(df):
    df = df.copy()
    if "monthly_gross_income" in df.columns:
        df["log_income"] = np.log1p(df["monthly_gross_income"])
    return df


def drop_leaky(df):
    return df.drop(columns=[c for c in LEAKY if c in df.columns], errors="ignore")


def build_safe_features(df):
    df = df.copy()
    eps = 1e-9
    if "emi_amount" in df.columns and "net_disposable_income" in df.columns:
        df["emi_stress_ratio_safe"] = df["emi_amount"] / (df["net_disposable_income"] + eps)
    if "auto_debit_bounce_count" in df.columns and "auto_debit_attempt_count" in df.columns:
        df["bounce_frequency_ratio_safe"] = (
            df["auto_debit_bounce_count"] / (df["auto_debit_attempt_count"] + eps)
        )
    if "outstanding_balance" in df.columns and "loan_amount" in df.columns:
        df["outstanding_balance_ratio_safe"] = df["outstanding_balance"] / (df["loan_amount"] + eps)
    if "outstanding_balance_lag1" in df.columns and "outstanding_balance_lag2" in df.columns:
        shock = (df["outstanding_balance_lag1"] - df["outstanding_balance_lag2"]) / (
            df["outstanding_balance_lag2"] + eps
        )
        df["utilization_shock_safe"] = (shock > 0.2).astype(int)
    if "dpd_lag1" in df.columns and "dpd_lag4" in df.columns:
        df["rolling_dpd_trend_safe"] = (df["dpd_lag1"] - df["dpd_lag4"]) / 3.0
    if "dpd_lag1" in df.columns and "dpd_lag2" in df.columns:
        df["delinquency_acceleration_safe"] = df["dpd_lag1"] - df["dpd_lag2"]
    pay_cols = [c for c in ["payment_lag1", "payment_lag2", "payment_lag3"] if c in df.columns]
    if len(pay_cols) == 3:
        p = df[pay_cols].astype(float)
        df["payment_volatility_safe"] = p.std(axis=1) / (p.mean(axis=1) + eps)
    due_cols = [c for c in ["due_lag1", "due_lag2", "due_lag3"] if c in df.columns]
    pay_cols2 = [c for c in ["payment_lag1", "payment_lag2", "payment_lag3"] if c in df.columns]
    if len(due_cols) == 3 and len(pay_cols2) == 3:
        due = df[due_cols].astype(float)
        pay = df[pay_cols2].astype(float)
        df["pay_to_due_ratio_3m_safe"] = (pay.sum(axis=1) / (due.sum(axis=1) + eps)).clip(0, 5)
    if "repayment_fatigue_index" in df.columns:
        df["repayment_fatigue_safe"] = df["repayment_fatigue_index"]
    if "total_credit_exposure" in df.columns and "loan_amount" in df.columns:
        df["exposure_concentration_safe"] = df["total_credit_exposure"] / (df["loan_amount"] + eps)
    parts = []
    for col, w in [
        ("rolling_dpd_trend_safe", 0.25),
        ("bounce_frequency_ratio_safe", 0.25),
        ("emi_stress_ratio_safe", 0.25),
        ("payment_volatility_safe", 0.25),
    ]:
        if col in df.columns:
            parts.append((col, w))
    if parts:
        brs, wsum = 0, 0
        for col, w in parts:
            brs += w * df[col].fillna(0)
            wsum += w
        df["behavioral_risk_score_safe"] = brs / (wsum + eps)
    drop_risky = [
        "emi_stress_ratio", "payment_volatility_score", "rolling_dpd_trend",
        "bounce_frequency_ratio", "outstanding_balance_ratio",
        "delinquency_acceleration", "behavioral_risk_score",
    ]
    df = df.drop(columns=[c for c in drop_risky if c in df.columns], errors="ignore")
    return df


def fe_pipeline(df):
    return build_safe_features(drop_leaky(add_features(df)))


def split_xy(df):
    y = df["ews_flag"].map(MAPD) if "ews_flag" in df.columns else None
    X = df.drop(columns=["ews_flag"], errors="ignore")
    return X, y


def set_cat(X, cats):
    X = X.copy()
    for c in cats:
        if c in X.columns:
            X[c] = X[c].astype("category")
    return X


# ── Evaluation helpers ────────────────────────────────────────────────────────
def ks_stat(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def lift_at_k(y_true, y_prob, k=0.10):
    n = len(y_true)
    m = int(np.ceil(n * k))
    idx = np.argsort(-np.asarray(y_prob))[:m]
    base = np.asarray(y_true).mean()
    top  = np.asarray(y_true)[idx].mean()
    return float(top / base) if base > 0 else float("nan")


def recall_at_fpr(y_true, y_prob, fpr_target=0.20):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = int(np.argmin(np.abs(fpr - fpr_target)))
    return float(tpr[j]), float(thr[j])


def collect_metrics(model, X, y, thr):
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= thr).astype(int)
    r20, _ = recall_at_fpr(y, prob)
    return {
        "AUC":       round(roc_auc_score(y, prob), 4),
        "KS":        round(ks_stat(y, prob), 4),
        "Brier":     round(brier_score_loss(y, prob), 4),
        "Precision": round(precision_score(y, pred, zero_division=0), 4),
        "Recall":    round(recall_score(y, pred, zero_division=0), 4),
        "F1":        round(f1_score(y, pred, zero_division=0), 4),
        "Lift@10%":  round(lift_at_k(y, prob, k=0.10), 4),
        "R@FPR20%":  round(r20, 4),
        "threshold": thr,
    }


def threshold_sweep(model, X, y, min_recall=0.70):
    prob = model.predict_proba(X)[:, 1]
    best = (None, None)
    for thr in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40]:
        pred = (prob >= thr).astype(int)
        f = f1_score(y, pred, zero_division=0)
        r = recall_score(y, pred, zero_division=0)
        if best[0] is None or (f > best[0] and r >= min_recall):
            best = (f, thr)
    return best[1] if best[1] else 0.30


def make_lgbm(pos_weight):
    return lgb.LGBMClassifier(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=pos_weight,
        random_state=42,
        verbose=-1,
    )


def ews_overlay(prob, df, thr_high, thr_mid=0.12):
    """Business-rule overlay from original notebook.

    Flags a record if:
      (a) model prob >= thr_high, OR
      (b) prob >= thr_mid AND utilization_ratio >= 0.75
          AND payment_volatility_safe >= 75th-pct
          AND (bounce >= 1 OR emi_stress >= 75th-pct)
    """
    df = df.reset_index(drop=True)
    prob = np.asarray(prob)

    util   = df["utilization_ratio"].values      if "utilization_ratio"       in df.columns else np.zeros(len(df))
    vol    = df["payment_volatility_safe"].values if "payment_volatility_safe" in df.columns else np.zeros(len(df))
    bounce = df["auto_debit_bounce_count"].values if "auto_debit_bounce_count" in df.columns else np.zeros(len(df))
    emi    = df["emi_stress_ratio_safe"].values   if "emi_stress_ratio_safe"   in df.columns else np.zeros(len(df))

    vol_thr = float(np.nanpercentile(vol, 75)) if np.any(~np.isnan(vol)) else 0.0
    emi_thr = float(np.nanpercentile(emi, 75)) if np.any(~np.isnan(emi)) else 0.0

    rule = (
        (prob >= thr_high)
        | (
            (prob >= thr_mid)
            & (util >= 0.75)
            & (vol >= vol_thr)
            & ((bounce >= 1) | (emi >= emi_thr))
        )
    )
    return rule.astype(int)


# ════════════════════════════════════════════════════════════════════════════
# 1. LOAD RAW DATA
# ════════════════════════════════════════════════════════════════════════════
print("Loading data files...")
df01 = pd.read_csv(F01)
df02 = pd.read_csv(F02)
df03 = pd.read_csv(F03)
print(f"  File01: {df01.shape}  File02: {df02.shape}  File03: {df03.shape}")


# ════════════════════════════════════════════════════════════════════════════
# ORIGINAL PIPELINE  (leaky)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  ORIGINAL PIPELINE  (reproducing notebook behaviour)")
print("="*60)

# Split File01 70/30
X_f01 = df01.drop(columns=["customer_id"], errors="ignore")
y_f01 = df01["ews_flag"]
X_tr_o, X_cal_o, y_tr_o, y_cal_o = train_test_split(
    X_f01, y_f01, test_size=0.3, random_state=42, stratify=y_f01
)
print(f"  Train (70%): {X_tr_o.shape}   Cal/test1 (30%): {X_cal_o.shape}")

def prep_with_flag(X, y):
    df = X.copy()
    df["ews_flag"] = y.values
    df = fe_pipeline(df)
    return split_xy(df)

X_train_o, y_train_o = prep_with_flag(X_tr_o, y_tr_o)
X_cal_o2,  y_cal_o2  = prep_with_flag(X_cal_o, y_cal_o)

test2_df_o = fe_pipeline(df02.drop(columns=["customer_id"], errors="ignore"))
X_test2_o, y_test2_o = split_xy(test2_df_o)

use_cat_o = [c for c in CAT_COLS if c in X_train_o.columns]
for df_ in [X_train_o, X_cal_o2, X_test2_o]:
    for c in use_cat_o:
        if c in df_.columns:
            df_[c] = df_[c].astype("category")

pw_o = (y_train_o == 0).sum() / (y_train_o == 1).sum()
base_o = make_lgbm(pw_o)
print("  Training original base model...")
base_o.fit(X_train_o, y_train_o, categorical_feature=use_cat_o)

# Calibrate on X_cal_o2 (same data will be evaluated → leakage)
cal_o = CalibratedClassifierCV(FrozenEstimator(base_o), method="isotonic", cv=2)
cal_o.fit(X_cal_o2, y_cal_o2)

# Tune threshold on File02, evaluate on File02 (→ leakage)
best_thr_o = threshold_sweep(cal_o, X_test2_o, y_test2_o)
print(f"  Best threshold: {best_thr_o}")

orig_cal_m   = collect_metrics(cal_o, X_cal_o2,  y_cal_o2,  best_thr_o)
orig_test2_m = collect_metrics(cal_o, X_test2_o, y_test2_o, best_thr_o)


# ════════════════════════════════════════════════════════════════════════════
# REFACTORED PIPELINE  (clean)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  REFACTORED PIPELINE  (no leakage)")
print("="*60)

# Train on ALL of File01
train_r_df = fe_pipeline(df01.drop(columns=["customer_id"], errors="ignore"))
X_train_r, y_train_r = split_xy(train_r_df)
print(f"  Train (100% File01): {X_train_r.shape}  (+{X_train_r.shape[0]-X_train_o.shape[0]:,} rows)")

# Split File02 50/50 → calibration / threshold-tune+eval
f02_fe = fe_pipeline(df02.drop(columns=["customer_id"], errors="ignore"))
X_f02, y_f02 = split_xy(f02_fe)
X_cal_r, X_tune_r, y_cal_r, y_tune_r = train_test_split(
    X_f02, y_f02, test_size=0.5, random_state=42, stratify=y_f02
)
print(f"  File02-cal: {X_cal_r.shape}   File02-tune/eval: {X_tune_r.shape}")

use_cat_r = [c for c in CAT_COLS if c in X_train_r.columns]
for df_ in [X_train_r, X_cal_r, X_tune_r]:
    for c in use_cat_r:
        if c in df_.columns:
            df_[c] = df_[c].astype("category")

pw_r = (y_train_r == 0).sum() / (y_train_r == 1).sum()
base_r = make_lgbm(pw_r)
print("  Training refactored base model...")
base_r.fit(X_train_r, y_train_r, categorical_feature=use_cat_r)

# Calibrate on File02-cal ONLY (never evaluated here)
cal_r = CalibratedClassifierCV(FrozenEstimator(base_r), method="isotonic", cv=2)
cal_r.fit(X_cal_r, y_cal_r)

# Tune threshold on File02-tune ONLY
best_thr_r = threshold_sweep(cal_r, X_tune_r, y_tune_r)
print(f"  Best threshold: {best_thr_r}")

# Evaluate on File02-tune (AUC/KS/Brier clean; P/R/F1 slight overestimate)
refac_tune_m = collect_metrics(cal_r, X_tune_r, y_tune_r, best_thr_r)
refac_cal_m  = collect_metrics(cal_r, X_cal_r,  y_cal_r,  best_thr_r)


# ════════════════════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ════════════════════════════════════════════════════════════════════════════
print("\n\n" + "="*90)
print("  METRICS COMPARISON  (primary evaluation set per pipeline)")
print("="*90)

bias_note = {
    "AUC":       "clean  (threshold-independent)",
    "KS":        "clean  (threshold-independent)",
    "Brier":     "orig INFLATED (cal=eval); refac CLEAN",
    "Precision": "both slightly optimistic (thr tuned on same eval set)",
    "Recall":    "both slightly optimistic (thr tuned on same eval set)",
    "F1":        "both slightly optimistic (thr tuned on same eval set)",
    "Lift@10%":  "clean  (threshold-independent)",
    "R@FPR20%":  "clean  (threshold-independent)",
}

metrics_order = ["AUC", "KS", "Brier", "Precision", "Recall", "F1", "Lift@10%", "R@FPR20%"]

print(f"\n{'Metric':<12} {'ORIG (File02)':>16} {'REFAC (File02-tune)':>20}  {'Delta':>8}  Interpretation")
print("-"*90)
for m in metrics_order:
    o = orig_test2_m[m]
    r = refac_tune_m[m]
    d = r - o
    sign = "+" if d >= 0 else ""
    print(f"{m:<12} {o:>16.4f} {r:>20.4f}  {sign}{d:>7.4f}  {bias_note[m]}")

print(f"\n{'Train rows':<12} {X_train_o.shape[0]:>16,} {X_train_r.shape[0]:>20,}  {X_train_r.shape[0]-X_train_o.shape[0]:>+8,}  extra training rows")
print(f"{'Threshold':<12} {orig_test2_m['threshold']:>16.2f} {refac_tune_m['threshold']:>20.2f}")

print(
    "\nSummary of leakage fixed:\n"
    "  ORIGINAL   -> calibration fit on X_test1_2 then evaluated on X_test1_2 (Brier inflated)\n"
    "                threshold tuned on File02 then evaluated on File02 (P/R/F1 inflated)\n"
    "                model trained on only 70%% of File01 (wasted labels)\n\n"
    "  REFACTORED -> trained on 100%% File01\n"
    "                calibrated on File02-cal (never evaluated on it)\n"
    "                threshold tuned on File02-tune; evaluated on File02-tune\n"
    "                AUC / KS / Brier / Lift are all now honest\n"
    "                P/R/F1 still mildly optimistic -- unavoidable without labelled File03\n"
)

# ════════════════════════════════════════════════════════════════════════════
# SUBMISSION  (original model + overlay)
# ════════════════════════════════════════════════════════════════════════════
print("\nGenerating ORIGINAL submission with EWS overlay...")
val_o_df = fe_pipeline(
    df03.drop(columns=["customer_id", "ews_flag"], errors="ignore")
)
for c in use_cat_o:
    if c in val_o_df.columns:
        val_o_df[c] = val_o_df[c].astype("category")

val_prob_o   = cal_o.predict_proba(val_o_df)[:, 1]
val_raw_o    = (val_prob_o >= best_thr_o).astype(int)
val_overlay_o = ews_overlay(val_prob_o, val_o_df, thr_high=best_thr_o, thr_mid=0.12)

# ════════════════════════════════════════════════════════════════════════════
# SUBMISSION  (refactored model + overlay)
# ════════════════════════════════════════════════════════════════════════════
print("Generating REFACTORED submission with EWS overlay...")
val_r_df = fe_pipeline(
    df03.drop(columns=["customer_id", "ews_flag"], errors="ignore")
)
for c in use_cat_r:
    if c in val_r_df.columns:
        val_r_df[c] = val_r_df[c].astype("category")

val_prob_r    = cal_r.predict_proba(val_r_df)[:, 1]
val_raw_r     = (val_prob_r >= best_thr_r).astype(int)
val_overlay_r = ews_overlay(val_prob_r, val_r_df, thr_high=best_thr_r, thr_mid=0.12)

# ── Flag-rate comparison ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("  FILE03 FLAG-RATE COMPARISON")
print("="*60)
print(f"{'Method':<35} {'Flagged':>8} {'Rate':>8}")
print("-"*55)
print(f"{'ORIG  raw threshold only':<35} {val_raw_o.sum():>8,} {val_raw_o.mean()*100:>7.1f}%")
print(f"{'ORIG  + EWS overlay':<35} {val_overlay_o.sum():>8,} {val_overlay_o.mean()*100:>7.1f}%")
print(f"{'REFAC raw threshold only':<35} {val_raw_r.sum():>8,} {val_raw_r.mean()*100:>7.1f}%")
print(f"{'REFAC + EWS overlay':<35} {val_overlay_r.sum():>8,} {val_overlay_r.mean()*100:>7.1f}%")

overlap = (val_overlay_o & val_overlay_r).sum()
union   = (val_overlay_o | val_overlay_r).sum()
print(f"\nCases flagged by BOTH  : {overlap:,}")
print(f"Cases flagged by ORIG only  : {(val_overlay_o & ~val_overlay_r).sum():,}")
print(f"Cases flagged by REFAC only : {(val_overlay_r & ~val_overlay_o).sum():,}")
if union > 0:
    print(f"Agreement (Jaccard)    : {overlap/union:.3f}")

# Save refactored + overlay submission
ids_val = df03["customer_id"] if "customer_id" in df03.columns else pd.RangeIndex(len(df03))
submission = pd.DataFrame({"customer_id": ids_val, "ews_flag": val_overlay_r})
out_path = r"C:/Users/ssume/Downloads/Credit_Card_Delinquency-EWS_refined/File04_Refactored_Submit.csv"
submission.to_csv(out_path, index=False)
print(f"\nSaved (refactored + overlay): {out_path}")
print(f"  Flagged positives: {val_overlay_r.sum():,} / {len(val_overlay_r):,}  ({val_overlay_r.mean()*100:.1f}%)")
