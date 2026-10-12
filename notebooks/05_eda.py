# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Exploratory Data Analysis (EDA)
# MAGIC Goals: (1) check the order-level grain, (2) set baseline KPIs, (3) confirm leakage,
# MAGIC (4) investigate DQ warnings, (5) see where lateness happens, (6) check data readiness for forecasting.
# MAGIC
# MAGIC *Analysis notebook. Not part of the scheduled pipeline.*

# COMMAND ----------

import yaml
import matplotlib.pyplot as plt

with open("../config/config.yaml") as f:
    cfg = yaml.safe_load(f)

C = cfg["catalog"]
G = f"{C}.gold"

# COMMAND ----------

# MAGIC %md ## 1. Build an order-level view (one row per order)

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TEMP VIEW orders AS
SELECT
    f.order_id,
    MIN(f.order_ts)                    AS order_ts,
    MAX(f.is_late)                     AS is_late,
    MIN(f.is_late)                     AS is_late_min,      -- to check consistency
    MAX(f.delay_days)                  AS delay_days,
    MAX(f.scheduled_days)              AS scheduled_days,
    MAX(f.actual_days)                 AS actual_days,
    MAX(sm.shipping_mode)              AS shipping_mode,
    COUNT(DISTINCT sm.shipping_mode)   AS n_shipping_modes,
    MAX(l.market)                      AS market,
    MAX(l.order_region)                AS order_region,
    MAX(c.customer_segment)            AS customer_segment,
    MAX(f.payment_type)                AS payment_type,
    MAX(f.delivery_status)             AS delivery_status,
    MAX(f.order_status)                AS order_status,
    SUM(f.net_sales)                   AS order_sales,
    SUM(f.quantity)                    AS order_qty,
    COUNT(*)                           AS n_items
FROM {G}.fact_order_items f
JOIN {G}.dim_shipping_mode sm ON f.shipping_mode_key = sm.shipping_mode_key
JOIN {G}.dim_location      l  ON f.location_key      = l.location_key
JOIN {G}.dim_customer      c  ON f.customer_key      = c.customer_key
GROUP BY f.order_id
""")

display(spark.sql("""
    SELECT COUNT(*)                                              AS orders,
           SUM(CASE WHEN n_shipping_modes > 1 THEN 1 ELSE 0 END) AS orders_with_mixed_modes,
           SUM(CASE WHEN is_late <> is_late_min THEN 1 ELSE 0 END) AS orders_with_mixed_late_flag,
           ROUND(AVG(n_items), 2)                                AS avg_items_per_order
    FROM orders
"""))

# COMMAND ----------

# MAGIC %md ## 2. Delivery status breakdown (do we exclude cancelled orders?)

# COMMAND ----------

display(spark.sql("""
    SELECT delivery_status, COUNT(*) AS orders,
           ROUND(100 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
    FROM orders
    GROUP BY delivery_status
    ORDER BY orders DESC
"""))

# COMMAND ----------

# MAGIC %md ## 3. Leakage check: `delivery_status` vs `is_late`
# MAGIC If one status maps perfectly to late/on-time, it gives away the answer, so it must never be a feature.

# COMMAND ----------

display(spark.sql("""
    SELECT delivery_status,
           SUM(CASE WHEN is_late = 1 THEN 1 ELSE 0 END) AS late_orders,
           SUM(CASE WHEN is_late = 0 THEN 1 ELSE 0 END) AS on_time_orders
    FROM orders
    GROUP BY delivery_status
    ORDER BY delivery_status
"""))

# COMMAND ----------

# MAGIC %md ## 4. Investigate the DQ warnings from Silver

# COMMAND ----------

# Rows where the late flag doesn't match actual vs scheduled days
display(spark.sql(f"""
    SELECT delivery_status, is_late, actual_days, scheduled_days, COUNT(*) AS item_rows
    FROM {G}.fact_order_items
    WHERE is_late <> CAST(actual_days > scheduled_days AS INT)
    GROUP BY ALL
    ORDER BY item_rows DESC
"""))

# Rows where shipping happened before the order
display(spark.sql(f"""
    SELECT order_id, order_ts, shipping_ts, actual_days, delivery_status
    FROM {G}.fact_order_items
    WHERE shipping_ts < order_ts
"""))

# COMMAND ----------

# MAGIC %md ## 5. Baseline KPIs (excluding cancelled orders, which were never delivered)

# COMMAND ----------

baseline = spark.sql("""
    SELECT
        COUNT(*)                                                         AS orders,
        ROUND(100 * AVG(1 - is_late), 2)                                 AS on_time_pct,
        ROUND(100 * AVG(is_late), 2)                                     AS late_rate_pct,
        ROUND(AVG(CASE WHEN is_late = 1 THEN delay_days END), 2)         AS avg_delay_days_when_late,
        ROUND(SUM(CASE WHEN is_late = 1 THEN order_sales END), 0)        AS late_order_revenue,
        ROUND(100 * SUM(CASE WHEN is_late = 1 THEN order_sales END)
                  / SUM(order_sales), 2)                                 AS late_revenue_pct
    FROM orders
    WHERE delivery_status <> 'Shipping canceled'
""")
display(baseline)

baseline.write.mode("overwrite").saveAsTable(f"{G}.baseline_kpis")
overall_late = baseline.first().late_rate_pct

# COMMAND ----------

# MAGIC %md ## 6. Where does lateness happen? (order level)

# COMMAND ----------

def late_by(col, min_orders=100):
    pdf = spark.sql(f"""
        SELECT {col},
               COUNT(*)                       AS orders,
               ROUND(100 * AVG(is_late), 1)   AS late_rate_pct,
               ROUND(AVG(delay_days), 2)      AS avg_delay_days
        FROM orders
        WHERE delivery_status <> 'Shipping canceled'
        GROUP BY {col}
        HAVING COUNT(*) >= {min_orders}
        ORDER BY late_rate_pct DESC
    """).toPandas()
    display(pdf)

    ax = pdf.sort_values("late_rate_pct").plot.barh(
        x=col, y="late_rate_pct", legend=False, figsize=(8, max(2.5, len(pdf) * 0.35))
    )
    ax.axvline(overall_late, linestyle="--", color="gray", label=f"overall {overall_late}%")
    ax.set_xlabel("Late rate (%)")
    ax.set_title(f"Late rate by {col}")
    ax.legend()
    plt.tight_layout()
    plt.show()

for col in ["shipping_mode", "market", "order_region", "customer_segment", "payment_type"]:
    late_by(col)

# COMMAND ----------

# MAGIC %md ## 7. Scheduled vs actual days by shipping mode

# COMMAND ----------

display(spark.sql("""
    SELECT shipping_mode, scheduled_days, actual_days, COUNT(*) AS orders
    FROM orders
    WHERE delivery_status <> 'Shipping canceled'
    GROUP BY ALL
    ORDER BY shipping_mode, scheduled_days, actual_days
"""))

# COMMAND ----------

# MAGIC %md ## 8. Late rate by product category (item level, categories with 500+ items)

# COMMAND ----------

display(spark.sql(f"""
    SELECT p.category_name,
           COUNT(*)                      AS items,
           ROUND(100 * AVG(f.is_late), 1) AS late_rate_pct
    FROM {G}.fact_order_items f
    JOIN {G}.dim_product p ON f.product_key = p.product_key
    WHERE f.delivery_status <> 'Shipping canceled'
    GROUP BY p.category_name
    HAVING COUNT(*) >= 500
    ORDER BY late_rate_pct DESC
"""))

# COMMAND ----------

# MAGIC %md ## 9. Does time matter? Late rate by weekday and month

# COMMAND ----------

for expr, label in [("date_format(order_ts, 'E')", "weekday"), ("month(order_ts)", "month")]:
    pdf = spark.sql(f"""
        SELECT {expr} AS {label}, COUNT(*) AS orders, ROUND(100 * AVG(is_late), 1) AS late_rate_pct
        FROM orders
        WHERE delivery_status <> 'Shipping canceled'
        GROUP BY 1 ORDER BY 1
    """).toPandas()
    display(pdf)

# COMMAND ----------

# MAGIC %md ## 10. Delay distribution

# COMMAND ----------

pdf = spark.sql("""
    SELECT delay_days FROM orders WHERE delivery_status <> 'Shipping canceled'
""").toPandas()

ax = pdf["delay_days"].plot.hist(bins=range(int(pdf.delay_days.min()), int(pdf.delay_days.max()) + 2),
                                  figsize=(8, 4), edgecolor="black")
ax.set_xlabel("Delay (actual − scheduled days)")
ax.set_title("How late are orders? (negative = early)")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md ## 11. Is the data ready for forecasting? Monthly volume

# COMMAND ----------

pdf = spark.sql(f"""
    SELECT date_trunc('month', order_ts) AS month,
           COUNT(DISTINCT order_id)      AS orders,
           SUM(quantity)                 AS units
    FROM {G}.fact_order_items
    GROUP BY 1 ORDER BY 1
""").toPandas()
display(pdf)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(pdf["month"], pdf["units"], marker="o")
ax.set_title("Monthly units ordered (look for sudden drops or odd months)")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Findings (fill in after running)
# MAGIC | # | Finding | So what? |
# MAGIC |---|---|---|
# MAGIC | 1 | Baseline late rate = ___% | The number the model must help reduce |
# MAGIC | 2 | Highest late rate: shipping mode ___ | ... |
# MAGIC | 3 | Region / market effect: ___ | ... |
# MAGIC | 4 | DQ warning explained: ___ | ... |
# MAGIC | 5 | Monthly volume: ___ | Affects how we forecast |