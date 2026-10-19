# Databricks notebook source
# MAGIC %md
# MAGIC # 08 - Risk tiers, explanations (SHAP), and cost-based decisions
# MAGIC Turns the late-risk score into actions: who is risky, why, and when speeding up pays off.

# COMMAND ----------

# MAGIC %pip install xgboost shap

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.abspath(".."))
from src.late_model import FEATURES, CAT_FEATURES, TARGET, time_split, make_xgb_pipeline
from src.risk import assign_tier, expedite_decision

with open("../config/config.yaml") as f:
    cfg = yaml.safe_load(f)

C     = cfg["catalog"]
G     = f"{C}.gold"
TIERS = cfg["risk_tiers"]
COST  = cfg["cost_assumptions"]

# COMMAND ----------

# MAGIC %md ## Step 1: Train the model (same features, split, and seed as notebook 07)

# COMMAND ----------

df = spark.table(f"{G}.order_features").toPandas()
train, test = time_split(df)
test = test.reset_index(drop=True)

pipe = make_xgb_pipeline().fit(train[FEATURES], train[TARGET])
p = pipe.predict_proba(test[FEATURES])[:, 1]
print("Test ROC-AUC:", round(roc_auc_score(test[TARGET], p), 4), "(should match notebook 07, ~0.775)")

# COMMAND ----------

# MAGIC %md ## Step 2 (US-9): Risk tiers

# COMMAND ----------

scores = test[["order_id", "order_ts", "shipping_mode", "order_sales", TARGET]].copy()
scores["late_risk"] = p
scores["risk_tier"] = assign_tier(p, TIERS["high"], TIERS["medium"])

total_late = scores[TARGET].sum()
tier_summary = (
    scores.groupby("risk_tier")
    .agg(orders=("order_id", "count"),
         actual_late_rate=(TARGET, "mean"),
         avg_predicted_risk=("late_risk", "mean"),
         late_orders=(TARGET, "sum"),
         revenue=("order_sales", "sum"))
    .reindex(["High", "Medium", "Low"])
)
tier_summary["share_of_orders"] = (tier_summary.orders / len(scores)).round(3)
tier_summary["share_of_all_late"] = (tier_summary.late_orders / total_late).round(3)
display(tier_summary.round(3).reset_index())

print("Tiers by shipping mode:")
display(pd.crosstab(scores.shipping_mode, scores.risk_tier).reset_index())

# COMMAND ----------

# MAGIC %md ## Step 3 (US-10): Why is each order risky? (SHAP)

# COMMAND ----------

pre, model = pipe.named_steps["pre"], pipe.named_steps["model"]
X_t = pre.transform(test[FEATURES])
X_t = X_t.toarray() if hasattr(X_t, "toarray") else X_t
names = pre.get_feature_names_out()

sv = shap.TreeExplainer(model).shap_values(X_t)

def original_feature(name):
    """'cat__shipping_mode_First Class' -> 'shipping_mode'; 'num__order_hour' -> 'order_hour'"""
    raw = name.split("__", 1)[1]
    if name.startswith("num__"):
        return raw
    for f in CAT_FEATURES:
        if raw.startswith(f + "_"):
            return f
    return raw

# Add up one-hot columns back into their original feature
groups = [original_feature(n) for n in names]
sv_by_feature = pd.DataFrame(sv, columns=names).T.groupby(groups).sum().T

# Global view: which features matter most overall?
global_imp = sv_by_feature.abs().mean().sort_values()
ax = global_imp.plot.barh(figsize=(8, 5))
ax.set_title("Average impact on late risk (mean |SHAP|)")
plt.tight_layout()
plt.show()

# COMMAND ----------

# Local view: top 3 reasons per order
feature_values = test[FEATURES]

def top_reasons(i, k=3):
    row = sv_by_feature.iloc[i]
    top = row.abs().sort_values(ascending=False).head(k).index
    return "; ".join(
        f"{f}={feature_values.at[i, f]} ({'+' if row[f] > 0 else ''}{row[f]:.2f})" for f in top
    )

scores["top_reasons"] = [top_reasons(i) for i in range(len(scores))]

print("Example High-risk orders and why:")
display(scores[scores.risk_tier == "High"]
        .sort_values("late_risk", ascending=False)
        [["order_id", "shipping_mode", "late_risk", "top_reasons", TARGET]].head(10))

print("Example Same Day orders (high vs low risk):")
display(scores[scores.shipping_mode == "Same Day"]
        .sort_values("late_risk")
        [["order_id", "late_risk", "top_reasons", TARGET]].iloc[[0, 1, -2, -1]])

# COMMAND ----------

# MAGIC %md ## Step 4 (US-11): When is speeding up an order worth it?
# MAGIC Expedite only if  risk × late_cost × effectiveness  >  expedite_cost.
# MAGIC **All costs are illustrative assumptions** (see config.yaml).

# COMMAND ----------

L, E, eff = COST["late_order_cost"], COST["expedite_cost"], COST["expedite_effectiveness"]
break_even = E / (L * eff)
print(f"Break-even risk: {break_even:.3f} → expedite orders above this risk")

scores["expedite"] = expedite_decision(p, L, E, eff)

def policy_outcome(name, mask):
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    late_caught = int(scores.loc[mask, TARGET].sum())
    avoided = late_caught * eff
    return {
        "policy": name,
        "orders_expedited": n,
        "late_orders_caught": late_caught,
        "late_orders_avoided": round(avoided),
        "expedite_spend": n * E,
        "late_cost_saved": round(avoided * L),
        "net_savings": round(avoided * L - n * E),
    }

policies = pd.DataFrame([
    policy_outcome("Expedite none", np.zeros(len(scores))),
    policy_outcome("Expedite all", np.ones(len(scores))),
    policy_outcome("Expedite First Class only (rule)", scores.shipping_mode == "First Class"),
    policy_outcome("Expedite High tier", scores.risk_tier == "High"),
    policy_outcome("Expedite by cost rule (model)", scores.expedite),
])
display(policies)

# COMMAND ----------

# MAGIC %md ### Sensitivity: how the decision changes if a late order costs more or less

# COMMAND ----------

rows = []
for late_cost in [5, 10, 15, 25, 40]:
    mask = expedite_decision(p, late_cost, E, eff)
    out = policy_outcome(f"late_cost=${late_cost}", mask)
    out["break_even_risk"] = round(E / (late_cost * eff), 3)
    out["net_savings"] = round(out["late_orders_avoided"] * late_cost - out["orders_expedited"] * E)
    rows.append(out)
display(pd.DataFrame(rows)[["policy", "break_even_risk", "orders_expedited", "late_orders_avoided", "net_savings"]])

# COMMAND ----------

# MAGIC %md ## Step 5: What if we simply fix the delivery promises?
# MAGIC Recalculate lateness if each mode promised days closer to what it actually delivers.

# COMMAND ----------

promise = spark.sql(f"""
    SELECT f.order_id,
           MAX(sm.shipping_mode)  AS shipping_mode,
           MAX(f.scheduled_days)  AS scheduled_days,
           MAX(f.actual_days)     AS actual_days
    FROM {G}.fact_order_items f
    JOIN {G}.dim_shipping_mode sm ON f.shipping_mode_key = sm.shipping_mode_key
    WHERE f.delivery_status <> 'Shipping canceled'
    GROUP BY f.order_id
""").toPandas()

NEW_PROMISE = {"First Class": 2, "Second Class": 4, "Same Day": 0, "Standard Class": 4}
promise["new_scheduled"] = promise.shipping_mode.map(NEW_PROMISE)
promise["late_now"] = (promise.actual_days > promise.scheduled_days).astype(int)
promise["late_after"] = (promise.actual_days > promise.new_scheduled).astype(int)

what_if = promise.groupby("shipping_mode").agg(
    orders=("order_id", "count"),
    promised_now=("scheduled_days", "max"),
    promised_new=("new_scheduled", "max"),
    late_rate_now=("late_now", "mean"),
    late_rate_after=("late_after", "mean"),
).round(3).reset_index()
display(what_if)

print(f"Overall late rate: {promise.late_now.mean():.2%} → {promise.late_after.mean():.2%}")

# COMMAND ----------

# MAGIC %md ## Step 6: Save scores to Gold (for the dashboard)

# COMMAND ----------

spark.createDataFrame(scores).write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(f"{G}.order_risk_scores")
spark.createDataFrame(policies).write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(f"{G}.expedite_policy_comparison")
spark.createDataFrame(what_if).write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(f"{G}.promise_what_if")
print("Saved: order_risk_scores, expedite_policy_comparison, promise_what_if")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusions
# MAGIC | Question | Answer |
# MAGIC |---|---|
# MAGIC | High tier: share of orders / actual late rate | 35.9% of orders, 90.0% actually late, capturing 56.5% of all late orders |
# MAGIC | Low tier: actual late rate | 37.6%. Mostly Standard Class, whose lateness isn't predictable from order-time data |
# MAGIC | Top SHAP drivers | shipping_mode (dominant), scheduled_days, order_hour. All other features near zero |
# MAGIC | Same Day finding | Late-evening orders (~7–9 PM) are late; early-morning orders (~2–3 AM) on time → cutoff-time pattern |
# MAGIC | Best expedite policy (net savings) | Cost rule: +$4,120 (4,156 orders, ≈ First Class + late-evening Same Day). Expedite-all loses $65,496 |
# MAGIC | Break-even risk | 0.889 at $15/late order. Expediting never pays at ≤ $10; ~$46K at $25; ~$135K at $40 |
# MAGIC | Late rate if promises are fixed | 57.3% → ~34% (First Class 1→2 days, Second Class 2→4 days) |
# MAGIC | **Recommendation** | 1) **Reset delivery promises** to match actual performance (biggest impact, no extra cost). 2) Add a **Same Day order cutoff time**. 3) Expedite only above the break-even risk, once the real cost of a late order is known. All costs here are illustrative |