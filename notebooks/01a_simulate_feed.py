# Databricks notebook source
# MAGIC %md
# MAGIC # 01a - Simulate live data feed
# MAGIC Splits the source CSV into batches (sorted by order date) and releases
# MAGIC them one at a time into `incoming/`, like new files arriving daily.

# COMMAND ----------

import os
import yaml
import pandas as pd

with open("../config/config.yaml") as f:
    cfg = yaml.safe_load(f)

LANDING = cfg["landing_path"]
SOURCE = f"{LANDING}/source/{cfg['source_file']}"
STAGING = f"{LANDING}/staging"
INCOMING = f"{LANDING}/incoming"

os.makedirs(STAGING, exist_ok=True)
os.makedirs(INCOMING, exist_ok=True)

dbutils.widgets.text("batches_to_release", "1")   # "1" = one batch, "all" = everything

# COMMAND ----------

# Split the source file into batches ONCE (only if nothing exists yet)
if not os.listdir(STAGING) and not os.listdir(INCOMING):
    df = pd.read_csv(SOURCE, encoding=cfg["encoding"])

    # Sort by order date so batches arrive in time order, like real data
    df["_sort_dt"] =  pd.to_datetime(df["order date (DateOrders)"])
    df = df.sort_values("_sort_dt").drop(columns="_sort_dt")

    size = cfg["batch_size"]
    for i, start in enumerate(range(0, len(df), size)):
        df.iloc[start:start + size].to_csv(
            f"{STAGING}/batch_{i:03d}.csv", index=False, encoding="utf-8"
        )
    print(f"Source rows: {len(df):,} -> split into {i + 1} batches")
else:
    print("Batches already created. Skipping split.")

# COMMAND ----------

# Release the next batch(es) into incoming/
pending = sorted(os.listdir(STAGING))
n = dbutils.widgets.get("batches_to_release")
to_release = pending if n == "all" else pending[: int(n)]

for name in to_release:
    dbutils.fs.mv(f"{STAGING}/{name}", f"{INCOMING}/{name}")
    print(f"Released: {name}")

print(f"Pending: {len(pending) - len(ti_release)} | Delivered so far: {len(os.listdir(INCOMING))}")