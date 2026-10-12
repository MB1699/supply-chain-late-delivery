# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Gold star schema
# MAGIC Builds dimension tables and the fact table from `silver.orders`.
# MAGIC Full refresh each run (small data, simple and safe), then integrity checks.

# COMMAND ----------

import yaml

with open("../config/config.yaml") as f:
    cfg = yaml.safe_load(f)

C      = cfg["catalog"]
SILVER = f"{C}.silver.orders"
G      = f"{C}.gold"

# COMMAND ----------

# MAGIC %md ## dim_date: one row per calendar day

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {G}.dim_date
COMMENT 'Calendar dimension. Grain: one row per day.'
AS
WITH bounds AS (
    SELECT MIN(order_date) AS start_d,
           GREATEST(MAX(order_date), MAX(to_date(shipping_ts))) AS end_d
    FROM {SILVER}
)
SELECT
    CAST(date_format(d, 'yyyyMMdd') AS INT) AS date_key,
    d                                       AS date,
    year(d)                                 AS year,
    quarter(d)                              AS quarter,
    month(d)                                AS month,
    date_format(d, 'MMMM')                  AS month_name,
    weekofyear(d)                           AS week_of_year,
    dayofweek(d)                            AS day_of_week,
    date_format(d, 'EEEE')                  AS day_name,
    dayofweek(d) IN (1, 7)                  AS is_weekend
FROM bounds
LATERAL VIEW explode(sequence(start_d, end_d, INTERVAL 1 DAY)) t AS d
""")

# COMMAND ----------

# MAGIC %md ## dim_product: one row per product

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {G}.dim_product
COMMENT 'Product dimension. Grain: one row per product_card_id.'
AS
SELECT
    product_card_id                       AS product_key,
    max_by(product_name, order_ts)        AS product_name,
    max_by(category_id, order_ts)         AS category_id,
    max_by(category_name, order_ts)       AS category_name,
    max_by(department_id, order_ts)       AS department_id,
    max_by(department_name, order_ts)     AS department_name,
    max_by(product_price, order_ts)       AS product_price
FROM {SILVER}
GROUP BY product_card_id
""")

# COMMAND ----------

# MAGIC %md ## dim_customer: one row per customer (latest known details)

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {G}.dim_customer
COMMENT 'Customer dimension (no PII). Grain: one row per customer_id.'
AS
SELECT
    customer_id                           AS customer_key,
    max_by(customer_segment, order_ts)    AS customer_segment,
    max_by(customer_city, order_ts)       AS customer_city,
    max_by(customer_state, order_ts)      AS customer_state,
    max_by(customer_country, order_ts)    AS customer_country
FROM {SILVER}
GROUP BY customer_id
""")

# COMMAND ----------

# MAGIC %md ## dim_location: where the order was shipped to

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {G}.dim_location
COMMENT 'Order destination. Grain: one row per market/region/country/state/city.'
AS
SELECT DISTINCT
    xxhash64(market, order_region, order_country, order_state, order_city) AS location_key,
    market,
    order_region,
    order_country,
    order_state,
    order_city
FROM {SILVER}
""")

# COMMAND ----------

# MAGIC %md ## dim_shipping_mode

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {G}.dim_shipping_mode
COMMENT 'Shipping options. Grain: one row per shipping mode.'
AS
SELECT
    xxhash64(shipping_mode)               AS shipping_mode_key,
    shipping_mode,
    MIN(days_for_shipment_scheduled)      AS min_scheduled_days,
    MAX(days_for_shipment_scheduled)      AS max_scheduled_days
FROM {SILVER}
GROUP BY shipping_mode
""")

# COMMAND ----------

# MAGIC %md ## fact_order_items: one row per order item

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {G}.fact_order_items
COMMENT 'Order item facts. Grain: one row per order_item_id.'
AS
SELECT
    s.order_item_id,
    s.order_id,

    -- Keys to dimensions
    CAST(date_format(s.order_date, 'yyyyMMdd') AS INT)            AS order_date_key,
    CAST(date_format(to_date(s.shipping_ts), 'yyyyMMdd') AS INT)  AS shipping_date_key,
    s.product_card_id                                             AS product_key,
    s.customer_id                                                 AS customer_key,
    xxhash64(s.market, s.order_region, s.order_country,
             s.order_state, s.order_city)                         AS location_key,
    xxhash64(s.shipping_mode)                                     AS shipping_mode_key,

    -- Order attributes
    s.type             AS payment_type,
    s.order_status,
    s.delivery_status,          -- post-delivery info: KPIs only, never a model feature

    -- Measures
    s.order_item_quantity       AS quantity,
    s.product_price             AS unit_price,
    s.order_item_discount       AS discount_amount,
    s.order_item_discount_rate  AS discount_rate,
    s.sales                     AS gross_sales,
    s.order_item_total          AS net_sales,
    s.order_profit_per_order    AS profit,
    s.order_item_profit_ratio   AS profit_ratio,

    -- Delivery
    s.days_for_shipment_scheduled AS scheduled_days,
    s.days_for_shipping_real      AS actual_days,      -- post-delivery: KPIs only
    s.shipping_delay_days         AS delay_days,       -- post-delivery: KPIs only
    s.late_delivery_risk          AS is_late,          -- target label

    s.order_ts,
    s.shipping_ts
FROM {SILVER} s
""")

# COMMAND ----------

# MAGIC %md ## Integrity checks

# COMMAND ----------

fact = f"{G}.fact_order_items"

checks = {
    "fact_rows_match_silver": abs(spark.table(fact).count() - spark.table(SILVER).count()),
    "fact_pk_unique":         spark.sql(f"SELECT COUNT(*) - COUNT(DISTINCT order_item_id) AS n FROM {fact}").first().n,
}

# Every key in the fact must exist in its dimension (no "orphans")
orphan_joins = {
    "orphan_order_date":    ("order_date_key",    f"{G}.dim_date",          "date_key"),
    "orphan_shipping_date": ("shipping_date_key", f"{G}.dim_date",          "date_key"),
    "orphan_product":       ("product_key",       f"{G}.dim_product",       "product_key"),
    "orphan_customer":      ("customer_key",      f"{G}.dim_customer",      "customer_key"),
    "orphan_location":      ("location_key",      f"{G}.dim_location",      "location_key"),
    "orphan_shipping_mode": ("shipping_mode_key", f"{G}.dim_shipping_mode", "shipping_mode_key"),
}
for name, (fk, dim, pk) in orphan_joins.items():
    checks[name] = spark.sql(f"""
        SELECT COUNT(*) AS n FROM {fact} f
        LEFT ANTI JOIN {dim} d ON f.{fk} = d.{pk}
    """).first().n

for name, failed in checks.items():
    print(f"{'PASS' if failed == 0 else 'FAIL'}  {name:24} {failed}")

failed = [n for n, v in checks.items() if v != 0]
if failed:
    raise Exception(f"Gold integrity checks failed: {failed}")
print("\nAll Gold checks passed ✅")

# COMMAND ----------

# MAGIC %md ## Table sizes

# COMMAND ----------

for t in ["dim_date", "dim_product", "dim_customer", "dim_location", "dim_shipping_mode", "fact_order_items"]:
    print(f"{t:20} {spark.table(f'{G}.{t}').count():>8,}")