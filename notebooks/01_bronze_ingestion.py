# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Bronze ingestion
# MAGIC Picks up only NEW files from `incoming/` using Auto Loader and appends them
# MAGIC to `bronze.orders_raw`. Data is kept as-is (all strings) plus audit columns.

# COMMAND ----------

import re
import yaml
from pyspark.sql import functions as F

with open("../config/config.yaml") as f:
    cfg = yaml.safe_load(f)

CATALOG    = cfg["catalog"]
LANDING    = cfg["landing_path"]
INCOMING   = f"{LANDING}/incoming"
CHECKPOINT = f"{LANDING}/_checkpoints/bronze_orders"
SCHEMA_LOC = f"{LANDING}/_schemas/bronze_orders"
TARGET     = f"{CATALOG}.bronze.orders_raw"

# COMMAND ----------

def clean_col(name: str) -> str:
    """'Days for shipping (real)' -> 'days_for_shipping_real'.
    Delta tables don't allow spaces or brackets in column names."""
    if name.startswith("_"):
        return name
    return re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()

# COMMAND ----------

raw = (
    spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "csv")
    .option("header", "true")
    .option("cloudFiles.inferColumnTypes", "false")   # Bronze = raw strings
    .option("cloudFiles.schemaLocation", SCHEMA_LOC)
    .load(INCOMING)
)

new_names = [clean_col(c) for c in raw.columns]
assert len(new_names) == len(set(new_names)), "Duplicate column names after cleaning!"

bronze = raw.select(
    *[F.col(f"`{c}`").alias(clean_col(c)) for c in raw.columns],
    F.col("_metadata.file_path").alias("_source_file"),
    F.current_timestamp().alias("_ingested_at"),
)

# COMMAND ----------

query = (
    bronze.writeStream.format("delta")
    .option("checkpointLocation", CHECKPOINT)
    .trigger(availableNow=True)    # process new files, then stop
    .toTable(TARGET)
)
query.awaitTermination()

# COMMAND ----------

# MAGIC %md ## Validation

# COMMAND ----------

display(spark.sql(f"""
    SELECT _source_file, COUNT(*) AS row_count, MAX(_ingested_at) AS loaded_at
    FROM {TARGET}
    GROUP BY _source_file
    ORDER BY _source_file
"""))

print("Total Bronze rows:", spark.table(TARGET).count())

# COMMAND ----------

# MAGIC %md ## Reset (dev only - uncomment to start over)

# COMMAND ----------

# spark.sql(f"DROP TABLE IF EXISTS {TARGET}")
# dbutils.fs.rm(CHECKPOINT, True)
# dbutils.fs.rm(SCHEMA_LOC, True)