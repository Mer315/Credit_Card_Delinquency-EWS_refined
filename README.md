# Credit Delinquency Early Warning System (EWS)

End-to-end pipeline to predict delinquency risk using **LightGBM**, calibrated probabilities, threshold tuning, and **SHAP explainability** with reason codes.

## ✨ Core Features
- Leakage-safe training (drops known leaky columns)
- Safe feature engineering (repayment stress + behavior signals)
- LightGBM champion model + class imbalance handling (`scale_pos_weight`)
- Probability calibration (Isotonic)
- Threshold sweep for best precision/recall operating point
- Metrics: ROC-AUC, KS, Brier, Lift@Top10%, Recall@FPR=20%
- Challenger baseline: Logistic Regression (scaled + OHE)
- Leakage sanity check: no-lag model
- Explainability: SHAP summary + top reason codes per customer

---

## 🧠 Feature Engineering

> `eps = 1e-9` added to avoid division-by-zero.

| Feature | Formula | Meaning |
|--------|---------|---------|
| `log_income` | `log(1 + monthly_gross_income)` | Normalizes income scale and reduces outlier impact |
| `emi_stress_ratio_safe` | `emi_amount / (net_disposable_income + eps)` | Measures repayment burden vs disposable income |
| `bounce_frequency_ratio_safe` | `auto_debit_bounce_count / (auto_debit_attempt_count + eps)` | Captures payment discipline / bounce intensity |
| `outstanding_balance_ratio_safe` | `outstanding_balance / (loan_amount + eps)` | Measures how much of the loan is still outstanding |
| `utilization_shock_safe` | `1[(out_bal_lag1 - out_bal_lag2)/(out_bal_lag2+eps) > 0.2]` | Flags sudden jump in utilization/outstanding balance |
| `rolling_dpd_trend_safe` | `(dpd_lag1 - dpd_lag4) / 3` | 3-month delinquency trend (improving vs worsening) |
| `delinquency_acceleration_safe` | `dpd_lag1 - dpd_lag2` | Detects recent acceleration in delinquency |
| `payment_volatility_safe` | `std(pay_lag1,pay_lag2,pay_lag3) / (mean(...) + eps)` | Measures instability in repayment amounts |
| `pay_to_due_ratio_3m_safe` | `(pay1+pay2+pay3) / (due1+due2+due3 + eps)` | Tracks repayment coverage vs obligations (3 months) |
| `repayment_fatigue_safe` | `repayment_fatigue_index` | Uses fatigue index directly as a stress signal |
| `exposure_concentration_safe` | `total_credit_exposure / (loan_amount + eps)` | Measures external exposure pressure vs this loan |
| `behavioral_risk_score_safe` | `avg(trend, bounce, emi_stress, volatility)` | Composite behavioral risk indicator from safe signals |

---

## 📁 Inputs
- `File01_Delinquency_ews_Model.csv` (training)
- `File02_Delinquency_ews_20k_1_Test_Model.csv` (testing + threshold tuning)
- `File03_Delinquency_ews_20k_2_Bus_Validate.csv` (final scoring)

## 📦 Output
Generates:
- `File04_Delinquency_Results_Submit_Final.csv` with `customer_id` and `ews_flag`
