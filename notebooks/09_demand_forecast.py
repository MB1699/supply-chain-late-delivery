# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 09 - Demand forecasting + forecast-based inventory
# MAGIC 1. Backtest 5 weekly forecasting methods per category (forecast 2017 using only past data).
# MAGIC 2. Rerun the inventory simulation: naive restocking vs forecast-based restocking, same weeks.

# COMMAND ----------

import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(".."))
from src.forecasting import (naive_next, moving_average_next, ses_next,
                             seasonal_naive_next, one_step_errors, tune_alpha)
from src.inventory import simulate_inventory

with open("../config/config.yaml") as f:
    cfg = yaml.safe_load(f)

C, G = cfg["catalog"], f"{cfg['catalog']}.gold"
FC, INV = cfg["forecast"], cfg["inventory"]
DATA_START = pd.Timestamp(FC["data_start"])
DATA_END   = pd.Timestamp(FC["data_end"])
TEST_START = pd.Timestamp(FC["test_start"])

# COMMAND ----------

# MAGIC %md ## Step 1: Weekly demand per category (full weeks, before the Oct-2017 drop)

# COMMAND ----------

weekly = spark.sql(f"""
    SELECT p.category_id, p.category_name,
           CAST(date_trunc('week', f.order_ts) AS DATE) AS week_start,
           SUM(f.quantity) AS demand
    FROM {G}.fact_order_items f
    JOIN {G}.dim_product p ON f.product_key = p.product_key
    WHERE f.order_ts >= TIMESTAMP '{DATA_START.date()}'
      AND f.order_ts <  TIMESTAMP '{(DATA_END + pd.Timedelta(days=1)).date()}'
    GROUP BY ALL
""").toPandas()
weekly["week_start"] = pd.to_datetime(weekly["week_start"])

series = {}
for (cid, cname), g in weekly.groupby(["category_id", "category_name"]):
    weeks = pd.date_range(g.week_start.min(), g.week_start.max(), freq="W-MON")
    s = g.set_index("week_start")["demand"].reindex(weeks, fill_value=0).astype(float)
    if (s.index < TEST_START).sum() >= FC["min_train_weeks"] and (s.index >= TEST_START).sum() > 0:
        series[(int(cid), cname)] = s

print(f"Categories with enough history: {len(series)} of {weekly.category_id.nunique()}")

# COMMAND ----------

# MAGIC %md ## Step 2: Backtest all methods

# COMMAND ----------

METHODS = ["naive", "ma_4", "ma_8", "ses", "seasonal_naive_52"]

def all_forecasts(s):
    d = s.values
    alpha = tune_alpha(d[s.index < TEST_START])     # tuned on training weeks only
    return {
        "naive": naive_next(d),
        "ma_4": moving_average_next(d, 4),
        "ma_8": moving_average_next(d, 8),
        "ses": ses_next(d, alpha),
        "seasonal_naive_52": seasonal_naive_next(d, 52),
    }, alpha

cache, metric_rows, fc_frames = {}, [], []
for key, s in series.items():
    fcs, alpha = all_forecasts(s)
    start = int((s.index < TEST_START).sum())
    cache[key] = (fcs, alpha, start)

    for m in METHODS:
        act, f = one_step_errors(s.values, fcs[m], start)
        metric_rows.append({"category_id": key[0], "category_name": key[1], "method": m,
                            "actual": act.sum(), "abs_err": np.abs(act - f).sum(), "err": (f - act).sum()})

    idx = np.arange(start, len(s))
    rec = pd.DataFrame({"category_id": key[0], "category_name": key[1],
                        "week_start": s.index[idx], "actual": s.values[idx], "ses_alpha": alpha})
    for m in METHODS:
        rec[f"fc_{m}"] = fcs[m][idx - 1]
    fc_frames.append(rec)

metrics = pd.DataFrame(metric_rows)
backtest = pd.concat(fc_frames, ignore_index=True)

overall = (metrics.groupby("method")[["actual", "abs_err", "err"]].sum()
           .assign(wape=lambda x: (x.abs_err / x.actual).round(4),
                   bias=lambda x: (x.err / x.actual).round(4))
           .sort_values("wape"))
best_method = overall.index[0]
naive_wape = overall.loc["naive", "wape"]
overall["improvement_vs_naive"] = ((naive_wape - overall.wape) / naive_wape).round(4)

print(f"Best method: {best_method}")
display(overall.reset_index())

# COMMAND ----------

# Which method wins most often per category?
per_cat = metrics.assign(wape=metrics.abs_err / metrics.actual)
winners = per_cat.loc[per_cat.groupby("category_id")["wape"].idxmin()]
display(winners.method.value_counts().rename("categories_won").reset_index())

# COMMAND ----------

# MAGIC %md ## Step 3 (US-14): Forecast vs actual for the 3 biggest categories

# COMMAND ----------

top3 = backtest.groupby("category_name").actual.sum().nlargest(3).index
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
for ax, cat in zip(axes, top3):
    b = backtest[backtest.category_name == cat]
    ax.plot(b.week_start, b.actual, label="actual", marker="o", markersize=3)
    ax.plot(b.week_start, b.fc_naive, label="naive", alpha=0.7)
    ax.plot(b.week_start, b[f"fc_{best_method}"], label=best_method, linewidth=2)
    ax.set_title(cat)
    ax.legend(loc="upper right")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md ## Step 4 (US-15b): Does the forecast improve inventory? Same weeks, same assumptions.

# COMMAND ----------

POLICIES = {"backtest_naive": "naive", f"backtest_{best_method}": best_method}

sim_frames = []
for key, s in series.items():
    fcs, _, start = cache[key]
    d_test = s.values[start:]
    for policy, m in POLICIES.items():
        sim = simulate_inventory(
            d_test, fcs[m][start:],
            lead_time=INV["lead_time_weeks"], safety_weeks=INV["safety_weeks"],
            initial_cover_weeks=INV["initial_cover_weeks"], avg_window=INV["avg_window_weeks"],
        )
        sim.insert(0, "week_start", s.index[start:])
        sim.insert(0, "category_name", key[1])
        sim.insert(0, "category_id", key[0])
        sim["policy"] = policy
        sim["is_simulated"] = True
        sim_frames.append(sim)

sims = pd.concat(sim_frames, ignore_index=True)

comparison = sims.groupby("policy").agg(
    category_weeks=("demand", "size"),
    total_demand=("demand", "sum"),
    total_sold=("units_sold", "sum"),
    lost_units=("lost_sales", "sum"),
    stockout_week_rate=("stockout_flag", "mean"),
    avg_stock=("closing_stock", "mean"),
    avg_weekly_demand=("demand", "mean"),
)
comparison["fill_rate"] = (comparison.total_sold / comparison.total_demand).round(4)
comparison["weeks_of_cover"] = (comparison.avg_stock / comparison.avg_weekly_demand).round(2)
comparison = comparison.round(4).reset_index()
display(comparison)

# COMMAND ----------

# 1) Order stability: how much do weekly orders swing? (lower CV = steadier orders for suppliers)
display(
    sims.groupby("policy").order_qty.agg(["mean", "std"])
        .assign(cv=lambda x: (x["std"] / x["mean"]).round(3))
        .round(1).reset_index()
)

# 2) Can the better forecast run on LESS safety stock?
rows = []
for sw in [0.0, 0.25, 0.5]:
    for policy, m in POLICIES.items():
        lost = dem = stock = n = 0
        for key, s in series.items():
            fcs, _, start = cache[key]
            sim = simulate_inventory(
                s.values[start:], fcs[m][start:],
                lead_time=INV["lead_time_weeks"], safety_weeks=sw,
                initial_cover_weeks=INV["initial_cover_weeks"], avg_window=INV["avg_window_weeks"],
            )
            lost += sim.lost_sales.sum(); dem += sim.demand.sum()
            stock += sim.closing_stock.sum(); n += len(sim)
        rows.append({"safety_weeks": sw, "policy": policy,
                     "fill_rate": round(1 - lost / dem, 4),
                     "lost_units": int(lost),
                     "avg_stock": round(stock / n, 1)})
display(pd.DataFrame(rows).sort_values(["safety_weeks", "policy"]))

# COMMAND ----------

# MAGIC %md ## Step 5: Future forecast (next ~3 months) for active categories

# COMMAND ----------

last_week = max(s.index[-1] for s in series.values())
future_rows = []
for key, s in series.items():
    if s.index[-1] != last_week:
        continue    # category no longer sold
    fcs, _, _ = cache[key]
    d = s.values
    for h in range(1, FC["horizon_weeks"] + 1):
        value = d[len(d) - 52 + h - 1] if best_method == "seasonal_naive_52" else fcs[best_method][-1]
        future_rows.append({"category_id": key[0], "category_name": key[1],
                            "week_start": last_week + pd.Timedelta(weeks=h),
                            "forecast_units": round(float(value), 1), "method": best_method})
future = pd.DataFrame(future_rows)
display(future.groupby("category_name").forecast_units.first().sort_values(ascending=False).head(10).reset_index())

# COMMAND ----------

# MAGIC %md ## Step 6: Save to Gold

# COMMAND ----------

def save(pdf, name):
    spark.createDataFrame(pdf).write.mode("overwrite").option("overwriteSchema", "true") \
        .saveAsTable(f"{G}.{name}")
    print("Saved:", f"{G}.{name}")

save(backtest, "demand_forecast_backtest")
save(overall.reset_index(), "forecast_metrics")
save(comparison, "inventory_policy_comparison")
save(future, "demand_forecast_future")

# Add backtest policies to the inventory table (side by side with the original naive run)
from pyspark.sql import functions as F
sims_df = spark.createDataFrame(sims).withColumn("week_start", F.col("week_start").cast("date"))
(sims_df.write.format("delta").mode("overwrite")
    .option("replaceWhere", "policy LIKE 'backtest_%'")
    .saveAsTable(f"{G}.inventory_sim_weekly"))
print("Updated:", f"{G}.inventory_sim_weekly")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusions
# MAGIC | Question | Answer |
# MAGIC |---|---|
# MAGIC | Best forecasting method (WAPE) | **SES**: 10.5% WAPE, wins 14 of 25 categories |
# MAGIC | Improvement vs naive | 22% lower error (13.5% → 10.5%). Same-week-last-year is worse (no seasonality) |
# MAGIC | Bias | +0.5%, essentially unbiased |
# MAGIC | Inventory: fill rate naive vs forecast | 99.97% → 99.97% (no change) |
# MAGIC | Inventory: lost units naive vs forecast | 27 → 32 (no meaningful difference) |
# MAGIC | Inventory: avg stock naive vs forecast | 184.8 → 182.4 (−1%); orders ~7% steadier |
# MAGIC | Safety stock test | Cutting the buffer from 0.5 to 0 weeks reduces stock **32%** with fill rate 99.97% → 99.89%, for both policies |
# MAGIC | **So what?** | Demand is flat with small noise, so a better forecast barely changes inventory outcomes. **The biggest saving is reducing the safety buffer** (illustrative, simulated). Forecasting would matter more with trends, seasonality, or promotions |