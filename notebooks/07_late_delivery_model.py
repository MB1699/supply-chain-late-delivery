# Databricks notebook source
# MAGIC %md
# MAGIC # 07 - Late-delivery model
# MAGIC Predicts late risk per order using ONLY information known when the order is placed.
# MAGIC Compares: always-late baseline, shipping-mode rule, logistic regression, XGBoost.

# COMMAND ----------

# MAGIC %pip install xgboost

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import yaml
import numpy as np
import pandas as pd
import mlflow
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_score, recall_score, f1_score, brier_score_loss)
from xgboost import XGBClassifier

with open("../config/config.yaml") as f:
    cfg = yaml.safe_load(f)

C = cfg["catalog"]
G = f"{C}.gold"
SEED = 42

# COMMAND ----------

# MAGIC %md ## Step 1: Order-level feature table (no leakage)

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {G}.order_features
COMMENT 'One row per delivered order. Features known at order time + target is_late.'
AS
WITH items AS (
    SELECT f.*, p.category_name
    FROM {G}.fact_order_items f
    JOIN {G}.dim_product p ON f.product_key = p.product_key
    WHERE f.delivery_status <> 'Shipping canceled'
),
main_cat AS (   -- category with the most units in each order
    SELECT order_id, max_by(category_name, quantity) AS main_category
    FROM items GROUP BY order_id
)
SELECT
    i.order_id,
    MIN(i.order_ts)                         AS order_ts,
    -- Features (known at order time)
    MAX(sm.shipping_mode)                   AS shipping_mode,
    MAX(i.scheduled_days)                   AS scheduled_days,
    MAX(l.market)                           AS market,
    MAX(l.order_region)                     AS order_region,
    MAX(c.customer_segment)                 AS customer_segment,
    MAX(i.payment_type)                     AS payment_type,
    MAX(mc.main_category)                   AS main_category,
    date_format(MIN(i.order_ts), 'E')       AS order_weekday,
    month(MIN(i.order_ts))                  AS order_month,
    hour(MIN(i.order_ts))                   AS order_hour,
    SUM(i.quantity)                         AS order_qty,
    COUNT(*)                                AS n_items,
    COUNT(DISTINCT i.category_name)         AS n_categories,
    ROUND(SUM(i.net_sales), 2)              AS order_sales,
    ROUND(AVG(i.discount_rate), 4)          AS avg_discount_rate,
    -- Target
    MAX(i.is_late)                          AS is_late
FROM items i
JOIN {G}.dim_shipping_mode sm ON i.shipping_mode_key = sm.shipping_mode_key
JOIN {G}.dim_location      l  ON i.location_key      = l.location_key
JOIN {G}.dim_customer      c  ON i.customer_key      = c.customer_key
JOIN main_cat             mc  ON i.order_id          = mc.order_id
GROUP BY i.order_id
""")

df = spark.table(f"{G}.order_features").toPandas()
print(f"Orders: {len(df):,} | Late rate: {df.is_late.mean():.2%}")

# COMMAND ----------

CAT_FEATURES = ["shipping_mode", "market", "order_region", "customer_segment",
                "payment_type", "main_category", "order_weekday"]
NUM_FEATURES = ["scheduled_days", "order_month", "order_hour", "order_qty",
                "n_items", "n_categories", "order_sales", "avg_discount_rate"]
FEATURES = CAT_FEATURES + NUM_FEATURES
TARGET = "is_late"

# Leakage guard: these must NEVER be features
LEAKAGE = {"delivery_status", "actual_days", "delay_days", "shipping_ts", "order_status"}
assert not LEAKAGE & set(FEATURES), "Leakage column found in features!"

# COMMAND ----------

# MAGIC %md ## Step 2: Time-based split (train on the past, test on the future)

# COMMAND ----------

split_date = pd.Timestamp("2017-01-01")
train = df[df.order_ts < split_date]
test  = df[df.order_ts >= split_date]

X_train, y_train = train[FEATURES], train[TARGET]
X_test,  y_test  = test[FEATURES],  test[TARGET]

print(f"Train: {len(train):,} orders ({train.order_ts.min():%Y-%m} to {train.order_ts.max():%Y-%m}) | late {y_train.mean():.2%}")
print(f"Test:  {len(test):,} orders ({test.order_ts.min():%Y-%m} to {test.order_ts.max():%Y-%m}) | late {y_test.mean():.2%}")

# COMMAND ----------

# MAGIC %md ## Step 3: Evaluation helper

# COMMAND ----------

def evaluate(name, y_true, p, threshold=0.5):
    y_pred = (p >= threshold).astype(int)
    return {
        "model": name,
        "roc_auc": round(roc_auc_score(y_true, p), 4),
        "pr_auc": round(average_precision_score(y_true, p), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "brier": round(brier_score_loss(y_true, p), 4),   # lower = better probabilities
    }

results, preds = [], {}

# COMMAND ----------

# MAGIC %md ## Step 4: Baselines

# COMMAND ----------

# Baseline 1: always predict the training late rate
p_const = np.full(len(test), y_train.mean())
preds["always_late"] = p_const
results.append(evaluate("1_always_late", y_test, p_const))

# Baseline 2: shipping-mode rule = late rate of that mode in training data
mode_rate = train.groupby("shipping_mode")[TARGET].mean()
p_rule = test["shipping_mode"].map(mode_rate).fillna(y_train.mean()).values
preds["mode_rule"] = p_rule
results.append(evaluate("2_shipping_mode_rule", y_test, p_rule))

print("Training late rate by mode:\n", mode_rate.round(3))

# COMMAND ----------

# MAGIC %md ## Step 5: Logistic regression and XGBoost (tracked in MLflow)

# COMMAND ----------

def make_pipeline(model):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ("num", "passthrough", NUM_FEATURES),
    ])
    return Pipeline([("pre", pre), ("model", model)])

models = {
    "3_logistic_regression": LogisticRegression(max_iter=2000),
    "4_xgboost": XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=SEED,
    ),
}

fitted = {}
for name, model in models.items():
    with mlflow.start_run(run_name=name):
        pipe = make_pipeline(model).fit(X_train, y_train)
        p = pipe.predict_proba(X_test)[:, 1]
        metrics = evaluate(name, y_test, p)

        mlflow.log_params({"model": name, "split_date": str(split_date.date()),
                           "n_features": len(FEATURES)})
        mlflow.log_metrics({k: v for k, v in metrics.items() if k != "model"})

        results.append(metrics)
        preds[name] = p
        fitted[name] = pipe

# COMMAND ----------

# MAGIC %md ## Step 6: Compare all models

# COMMAND ----------

comparison = pd.DataFrame(results)
display(comparison)

# COMMAND ----------

# MAGIC %md ## Step 7: Honesty test: does the model add anything WITHIN a shipping mode?
# MAGIC The rule gives every order in a mode the same score, so its AUC within a mode is 0.5.
# MAGIC If XGBoost is also ~0.5 within each mode, then it only learned the shipping mode.

# COMMAND ----------

rows = []
for mode in ["Second Class", "Same Day", "Standard Class"]:
    mask = (test["shipping_mode"] == mode).values
    if y_test[mask].nunique() < 2:
        continue
    rows.append({
        "shipping_mode": mode,
        "orders": int(mask.sum()),
        "late_rate": round(y_test[mask].mean(), 3),
        "xgb_auc_within_mode": round(roc_auc_score(y_test[mask], preds["4_xgboost"][mask]), 4),
        "logreg_auc_within_mode": round(roc_auc_score(y_test[mask], preds["3_logistic_regression"][mask]), 4),
    })
display(pd.DataFrame(rows))

# COMMAND ----------

# MAGIC %md ## Step 8: Top features (XGBoost)

# COMMAND ----------

xgb_pipe = fitted["4_xgboost"]
names = xgb_pipe.named_steps["pre"].get_feature_names_out()
imp = pd.Series(xgb_pipe.named_steps["model"].feature_importances_, index=names)
display(imp.sort_values(ascending=False).head(15).rename("importance").reset_index())

# COMMAND ----------

# MAGIC %md ## Step 9: Save test-set predictions to Gold (used later for risk tiers + dashboard)

# COMMAND ----------

out = test[["order_id", "order_ts", "shipping_mode", "order_sales", TARGET]].copy()
out["p_mode_rule"] = preds["mode_rule"]
out["p_logreg"]    = preds["3_logistic_regression"]
out["p_xgb"]       = preds["4_xgboost"]

spark.createDataFrame(out).write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(f"{G}.late_risk_predictions")
print("Saved:", f"{G}.late_risk_predictions", f"({len(out):,} rows)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusions (fill in after running)
# MAGIC | Question | Answer |
# MAGIC |---|---|
# MAGIC | Best model overall (ROC-AUC / PR-AUC) | ___ |
# MAGIC | Gain over shipping-mode rule | ___ |
# MAGIC | AUC within each mode | ___ (≈0.5 means no extra signal) |
# MAGIC | Top features | ___ |
# MAGIC | So what? | ___ |