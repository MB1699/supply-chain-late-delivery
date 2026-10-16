# Databricks notebook source
# MAGIC %md
# MAGIC # 06 - Root-cause analysis (statistical tests)
# MAGIC For each factor: is its link to lateness real (p-value) and how strong is it (effect size)?
# MAGIC Delivered orders only (cancelled excluded). Analysis notebook, not part of the pipeline.

# COMMAND ----------

import yaml
import numpy as np
import pandas as pd
from scipy import stats

with open("../config/config.yaml") as f:
    cfg = yaml.safe_load(f)

C = cfg["catalog"]
G = f"{C}.gold"

# COMMAND ----------

# MAGIC %md ## Load order-level data

# COMMAND ----------

pdf = spark.sql(f"""
    SELECT
        f.order_id,
        MAX(f.is_late)                      AS is_late,
        MAX(f.scheduled_days)               AS scheduled_days,
        MAX(f.actual_days)                  AS actual_days,
        MAX(sm.shipping_mode)               AS shipping_mode,
        MAX(l.market)                       AS market,
        MAX(l.order_region)                 AS order_region,
        MAX(c.customer_segment)             AS customer_segment,
        MAX(f.payment_type)                 AS payment_type,
        date_format(MIN(f.order_ts), 'E')   AS order_weekday,
        month(MIN(f.order_ts))              AS order_month,
        SUM(f.quantity)                     AS order_qty,
        COUNT(*)                            AS n_items,
        SUM(f.net_sales)                    AS order_sales
    FROM {G}.fact_order_items f
    JOIN {G}.dim_shipping_mode sm ON f.shipping_mode_key = sm.shipping_mode_key
    JOIN {G}.dim_location      l  ON f.location_key      = l.location_key
    JOIN {G}.dim_customer      c  ON f.customer_key      = c.customer_key
    WHERE f.delivery_status <> 'Shipping canceled'
    GROUP BY f.order_id
""").toPandas()

print(f"Delivered orders: {len(pdf):,}")

# COMMAND ----------

# MAGIC %md ## Helper functions

# COMMAND ----------

def chi_square(df, factor, target="is_late"):
    """Chi-square test of independence + Cramér's V effect size."""
    ct = pd.crosstab(df[factor], df[target])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    n = ct.values.sum()
    k = min(ct.shape) - 1
    v = np.sqrt(chi2 / (n * k)) if k > 0 else 0.0
    return chi2, p, dof, v


def strength(v):
    if v < 0.10: return "Negligible"
    if v < 0.30: return "Small"
    if v < 0.50: return "Medium"
    return "Large"

# COMMAND ----------

# MAGIC %md ## Test 1: Which factors are linked to lateness?
# MAGIC Bonferroni correction: we run several tests, so the significance threshold is divided by the number of tests.

# COMMAND ----------

factors = ["shipping_mode", "market", "order_region", "customer_segment",
           "payment_type", "order_weekday", "order_month"]
alpha = 0.05 / len(factors)

rows = []
for f in factors:
    chi2, p, dof, v = chi_square(pdf, f)
    rows.append({
        "test": "factor_vs_late",
        "factor": f,
        "chi2": round(chi2, 1),
        "dof": dof,
        "p_value": p,
        "significant": p < alpha,
        "cramers_v": round(v, 4),
        "strength": strength(v),
    })

test1 = pd.DataFrame(rows).sort_values("cramers_v", ascending=False)
display(test1)

# COMMAND ----------

# MAGIC %md ## Test 2: Do other factors matter once shipping mode is fixed?
# MAGIC Run the same tests *within* each shipping mode. First Class is skipped (it's always late, so nothing to test).

# COMMAND ----------

rows = []
for mode in ["Second Class", "Same Day", "Standard Class"]:
    sub = pdf[pdf.shipping_mode == mode]
    for f in [x for x in factors if x != "shipping_mode"]:
        chi2, p, dof, v = chi_square(sub, f)
        rows.append({
            "test": "within_mode",
            "shipping_mode": mode,
            "factor": f,
            "p_value": p,
            "significant": p < alpha,
            "cramers_v": round(v, 4),
            "strength": strength(v),
        })

test2 = pd.DataFrame(rows)
display(test2.pivot(index="factor", columns="shipping_mode", values="cramers_v"))

# COMMAND ----------

# MAGIC %md ## Test 3: Do Second Class and Standard Class really deliver in the same number of days?
# MAGIC If p-value is high, there's no evidence their actual delivery days differ → only the *promise* differs.

# COMMAND ----------

two = pdf[pdf.shipping_mode.isin(["Second Class", "Standard Class"])]
ct = pd.crosstab(two.shipping_mode, two.actual_days)
display(ct)

chi2, p, dof, _ = stats.chi2_contingency(ct)
ks = stats.ks_2samp(
    two[two.shipping_mode == "Second Class"].actual_days,
    two[two.shipping_mode == "Standard Class"].actual_days,
)
print(f"Chi-square on actual days: p = {p:.4f}")
print(f"KS test on actual days:    p = {ks.pvalue:.4f}")
print("→ Same delivery-time distribution" if min(p, ks.pvalue) > 0.05 else "→ Distributions differ")

test3 = pd.DataFrame([
    {"test": "second_vs_standard_actual_days", "factor": "chi_square", "p_value": p},
    {"test": "second_vs_standard_actual_days", "factor": "ks_test",    "p_value": ks.pvalue},
])

# COMMAND ----------

# MAGIC %md ## Test 4: Do order size or value differ between late and on-time orders?
# MAGIC Mann-Whitney U test (no normality assumption) + rank-biserial effect size (|r| < 0.1 = negligible).

# COMMAND ----------

rows = []
late, on_time = pdf[pdf.is_late == 1], pdf[pdf.is_late == 0]
for col in ["order_qty", "n_items", "order_sales"]:
    u, p = stats.mannwhitneyu(late[col], on_time[col], alternative="two-sided")
    r = 1 - (2 * u) / (len(late) * len(on_time))
    rows.append({
        "test": "numeric_late_vs_ontime",
        "factor": col,
        "median_late": late[col].median(),
        "median_on_time": on_time[col].median(),
        "p_value": p,
        "effect_size_r": round(r, 4),
        "strength": "Negligible" if abs(r) < 0.1 else "Small+",
    })

test4 = pd.DataFrame(rows)
display(test4)

# COMMAND ----------

# MAGIC %md ## Save all results to Gold

# COMMAND ----------

all_results = pd.concat([test1, test2, test3, test4], ignore_index=True)
all_results = all_results.astype(object).where(pd.notnull(all_results), None).astype(str)

spark.createDataFrame(all_results).write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(f"{G}.root_cause_tests")
print("Saved:", f"{G}.root_cause_tests")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusions (fill in after running)
# MAGIC | Question | Answer |
# MAGIC |---|---|
# MAGIC | Strongest driver of lateness | ___ (Cramér's V = ___) |
# MAGIC | Other factors | ___ |
# MAGIC | Do other factors matter within a shipping mode? | ___ |
# MAGIC | Second vs Standard actual days the same? | ___ (p = ___) |
# MAGIC | Order size / value effect | ___ |
# MAGIC | **Business recommendation** | ___ |