# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Silver cleaning
# MAGIC Reads only NEW Bronze rows, cleans them, and upserts (MERGE) into `silver.orders`.
# MAGIC Then runs data quality checks and saves the results to `silver.dq_results`.

# COMMAND ----------

import os
import sys
import yaml
from datetime import datetime
from delta.tables import DeltaTable
from pyspark.sql import functions as F

sys.path.append(os.path.abspath(".."))
from src.utils import clean_orders, dedupe_latest, PII_COLS, DUPLICATE_PAIRS

with open("../config/config.yaml") as f:
    cfg = yaml.safe_load(f)
with open("../config/country_map.yaml", encoding="utf-8") as f:
    COUNTRY_MAP = yaml.safe_load(f)

CATALOG    = cfg["catalog"]
LANDING    = cfg["landing_path"]
BRONZE     = f"{CATALOG}.bronze.orders_raw"
TARGET     = f"{CATALOG}.silver.orders"
DQ_TABLE   = f"{CATALOG}.silver.dq_results"
CHECKPOINT = f"{LANDING}/_checkpoints/silver_orders"

# COMMAND ----------

# MAGIC %md ## Step 1: Verify the "duplicate" columns really are duplicates before dropping them

# COMMAND ----------

bronze_df = spark.table(BRONZE)
for a, b in DUPLICATE_PAIRS:
    mismatches = bronze_df.filter(
        F.expr(f"NOT (try_cast({a} AS DOUBLE) <=> try_cast({b} AS DOUBLE))")
    ).count()
    print(f"{a:28} vs {b:24} -> mismatches: {mismatches}")

print("Rows with rescued (unparsed) data:", bronze_df.filter("_rescued_data IS NOT NULL").count())

# COMMAND ----------

# MAGIC %md ## Step 2: Clean new Bronze rows and MERGE into Silver

# COMMAND ----------

def upsert_to_silver(batch_df, batch_id):
    session = batch_df.sparkSession
    cleaned = dedupe_latest(clean_orders(batch_df, COUNTRY_MAP))

    if not session.catalog.tableExists(TARGET):
        cleaned.write.format("delta").saveAsTable(TARGET)
        return

    (
        DeltaTable.forName(session, TARGET).alias("t")
        .merge(cleaned.alias("s"), "t.order_item_id = s.order_item_id")
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )

query = (
    spark.readStream.table(BRONZE)
    .writeStream
    .foreachBatch(upsert_to_silver)
    .option("checkpointLocation", CHECKPOINT)
    .trigger(availableNow=True)
    .start()
)
query.awaitTermination()

print("Silver rows:", spark.table(TARGET).count())

# COMMAND ----------

# MAGIC %md ## Step 3: Data quality checks

# COMMAND ----------

silver = spark.table(TARGET)
silver_rows = silver.count()
bronze_keys = spark.table(BRONZE).select("order_item_id").distinct().count()

checks = [
    # (check name, severity, number of failing rows)
    ("pk_not_null",            "critical", silver.filter("order_item_id IS NULL").count()),
    ("pk_unique",              "critical", silver_rows - silver.select("order_item_id").distinct().count()),
    ("row_count_matches_bronze","critical", abs(silver_rows - bronze_keys)),
    ("order_ts_parsed",        "critical", silver.filter("order_ts IS NULL").count()),
    ("late_flag_valid",        "critical", silver.filter("late_delivery_risk IS NULL OR late_delivery_risk NOT IN (0, 1)").count()),
    ("no_pii_columns",         "critical", len([c for c in PII_COLS if c in silver.columns])),
    ("ship_after_order",       "warning",  silver.filter("shipping_ts < order_ts").count()),
    ("late_flag_consistent",   "warning",  silver.filter("late_delivery_risk != CAST(days_for_shipping_real > days_for_shipment_scheduled AS INT)").count()),
    ("quantity_positive",      "warning",  silver.filter("order_item_quantity <= 0").count()),
    ("customer_state_nulled",  "info",     silver.filter("customer_state IS NULL").count()),
]

run_ts = datetime.now()
dq_df = spark.createDataFrame(
    [(run_ts, name, sev, int(failed), "PASS" if failed == 0 else "FAIL") for name, sev, failed in checks],
    "run_ts TIMESTAMP, check_name STRING, severity STRING, failed_rows BIGINT, status STRING",
)
dq_df.write.mode("append").saveAsTable(DQ_TABLE)
display(dq_df)

critical_fails = [n for n, s, f in checks if s == "critical" and f > 0]
if critical_fails:
    raise Exception(f"Critical data quality checks failed: {critical_fails}")
print("All critical checks passed ✅")

# COMMAND ----------

# MAGIC %md ## Step 4: Countries still untranslated (review and add to country_map.yaml)

# COMMAND ----------

mapped_values = set(COUNTRY_MAP.values())
display(
    silver.groupBy("order_country").count()
    .filter(~F.col("order_country").isin(list(mapped_values)))
    .orderBy(F.desc("count"))
)

# COMMAND ----------

# MAGIC %md ## Reset (dev only - uncomment to start over)

# COMMAND ----------

# spark.sql(f"DROP TABLE IF EXISTS {TARGET}")
# spark.sql(f"DROP TABLE IF EXISTS {DQ_TABLE}")
# dbutils.fs.rm(CHECKPOINT, True)