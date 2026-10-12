# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Simulated inventory (Gold)
# MAGIC Weekly demand per category is REAL (from fact_order_items). Stock levels are SIMULATED
# MAGIC using the restocking assumptions in `config.yaml`. Policy here: "naive" (expect next week = this week).

# COMMAND ----------

import os
import sys
import yaml
import pandas as pd
from pyspark.sql import functions as F

sys.path.append(os.path.abspath(".."))
from src.inventory import simulate_inventory

with open("../config/config.yaml") as f:
    cfg = yaml.safe_load(f)

C      = cfg["catalog"]
G      = f"{C}.gold"
INV    = cfg["inventory"]
POLICY = "naive_last_week"
TARGET = f"{G}.inventory_sim_weekly"

# COMMAND ----------

# MAGIC %md ## Step 1: Real weekly demand per category

# COMMAND ----------

weekly = spark.sql(f"""
    SELECT p.category_id,
           p.category_name,
           CAST(date_trunc('week', f.order_ts) AS DATE) AS week_start,
           SUM(f.quantity)                              AS demand
    FROM {G}.fact_order_items f
    JOIN {G}.dim_product p ON f.product_key = p.product_key
    GROUP BY ALL
""").toPandas()

weekly["week_start"] = pd.to_datetime(weekly["week_start"])
print(f"{weekly['category_id'].nunique()} categories, {len(weekly):,} category-weeks with orders")

# COMMAND ----------

# MAGIC %md ## Step 2: Run the simulation for each category

# COMMAND ----------

results = []
for (cat_id, cat_name), g in weekly.groupby(["category_id", "category_name"]):
    # Fill weeks with no orders as 0 demand (only between the category's first and last week)
    weeks = pd.date_range(g["week_start"].min(), g["week_start"].max(), freq="W-MON")
    g = g.set_index("week_start").reindex(weeks, fill_value=0).rename_axis("week_start").reset_index()

    demand = g["demand"].values
    expected = demand            # naive policy: expect next week to look like this week

    sim = simulate_inventory(
        demand,
        expected,
        lead_time=INV["lead_time_weeks"],
        safety_weeks=INV["safety_weeks"],
        initial_cover_weeks=INV["initial_cover_weeks"],
        avg_window=INV["avg_window_weeks"],
    )
    sim.insert(0, "week_start", g["week_start"].values)
    sim.insert(0, "category_name", cat_name)
    sim.insert(0, "category_id", int(cat_id))
    results.append(sim)

sim_pdf = pd.concat(results, ignore_index=True)
sim_pdf["policy"] = POLICY
sim_pdf["is_simulated"] = True

# COMMAND ----------

# MAGIC %md ## Step 3: Save to Gold (only this policy's rows are replaced)

# COMMAND ----------

sim_df = spark.createDataFrame(sim_pdf).withColumn("week_start", F.col("week_start").cast("date"))

if spark.catalog.tableExists(TARGET):
    (sim_df.write.format("delta").mode("overwrite")
        .option("replaceWhere", f"policy = '{POLICY}'")
        .saveAsTable(TARGET))
else:
    sim_df.write.format("delta").saveAsTable(TARGET)
    spark.sql(f"COMMENT ON TABLE {TARGET} IS 'SIMULATED inventory. Demand is real; stock levels follow assumptions in config.yaml.'")

print("Rows written:", sim_df.count())

# COMMAND ----------

# MAGIC %md ## Step 4: Inventory KPIs per policy and category

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {G}.inventory_kpis
COMMENT 'KPIs from SIMULATED inventory, per policy and category.'
AS
SELECT
    policy,
    category_id,
    category_name,
    COUNT(*)                                        AS weeks,
    SUM(demand)                                     AS total_demand,
    SUM(units_sold)                                 AS total_sold,
    SUM(lost_sales)                                 AS total_lost,
    ROUND(SUM(units_sold) / NULLIF(SUM(demand), 0), 4)   AS fill_rate,
    ROUND(AVG(stockout_flag), 4)                    AS stockout_week_rate,
    ROUND(AVG(closing_stock), 1)                    AS avg_stock,
    ROUND(SUM(units_sold) / NULLIF(AVG(closing_stock), 0), 2) AS inventory_turnover
FROM {TARGET}
GROUP BY policy, category_id, category_name
""")

display(spark.sql(f"""
    SELECT policy,
           ROUND(SUM(total_sold) / SUM(total_demand), 4) AS overall_fill_rate,
           ROUND(AVG(stockout_week_rate), 4)             AS avg_stockout_week_rate,
           SUM(total_lost)                               AS total_lost_units
    FROM {G}.inventory_kpis
    GROUP BY policy
"""))

# COMMAND ----------

# MAGIC %md ## Step 5: Checks

# COMMAND ----------

sim_t = spark.table(TARGET).filter(F.col("policy") == POLICY)

real_total = spark.table(f"{G}.fact_order_items").agg(F.sum("quantity")).first()[0]
sim_total  = sim_t.agg(F.sum("demand")).first()[0]

checks = {
    "demand_matches_fact":   abs(real_total - sim_total),
    "no_negative_stock":     sim_t.filter("closing_stock < 0").count(),
    "sold_plus_lost_equals_demand": sim_t.filter("ABS(units_sold + lost_sales - demand) > 0.001").count(),
    "unique_category_week":  sim_t.count() - sim_t.select("category_id", "week_start").distinct().count(),
}

for name, failed in checks.items():
    print(f"{'PASS' if failed == 0 else 'FAIL'}  {name:30} {failed}")

failed = [n for n, v in checks.items() if v != 0]
if failed:
    raise Exception(f"Inventory checks failed: {failed}")
print("\nAll inventory checks passed ✅")