# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 10 - Dashboard views
# MAGIC Clean, chart-ready views on top of Gold for the AI/BI dashboard.

# COMMAND ----------

import yaml

with open("../config/config.yaml") as f:
    cfg = yaml.safe_load(f)

G = f"{cfg['catalog']}.gold"

# COMMAND ----------

# Delivered orders, one row per order, with risk scores where available (2017+)
spark.sql(f"""
CREATE OR REPLACE VIEW {G}.vw_orders AS
WITH orders AS (
    SELECT
        f.order_id,
        MIN(f.order_ts)              AS order_ts,
        MAX(sm.shipping_mode)        AS shipping_mode,
        MAX(f.scheduled_days)        AS scheduled_days,
        MAX(f.actual_days)           AS actual_days,
        MAX(f.delay_days)            AS delay_days,
        MAX(f.is_late)               AS is_late,
        MAX(f.delivery_status)       AS delivery_status,
        MAX(l.market)                AS market,
        MAX(l.order_region)          AS order_region,
        MAX(c.customer_segment)      AS customer_segment,
        ROUND(SUM(f.net_sales), 2)   AS order_sales,
        SUM(f.quantity)              AS order_qty
    FROM {G}.fact_order_items f
    JOIN {G}.dim_shipping_mode sm ON f.shipping_mode_key = sm.shipping_mode_key
    JOIN {G}.dim_location      l  ON f.location_key      = l.location_key
    JOIN {G}.dim_customer      c  ON f.customer_key      = c.customer_key
    GROUP BY f.order_id
)
SELECT
    o.*,
    CAST(o.order_ts AS DATE)                              AS order_date,
    ofe.main_category,
    r.late_risk,
    r.risk_tier,
    r.top_reasons,
    r.expedite,
    CASE WHEN o.is_late = 1 THEN o.delay_days  END        AS delay_if_late,
    CASE WHEN o.is_late = 1 THEN o.order_sales END        AS late_sales
FROM orders o
LEFT JOIN {G}.order_features    ofe ON o.order_id = ofe.order_id
LEFT JOIN {G}.order_risk_scores r   ON o.order_id = r.order_id
WHERE o.delivery_status <> 'Shipping canceled'
""")

# COMMAND ----------

# Promise what-if in "long" format (one row per mode per scenario) for a grouped bar chart
spark.sql(f"""
CREATE OR REPLACE VIEW {G}.vw_promise_what_if AS
SELECT shipping_mode, 'Now'         AS scenario, promised_now AS promised_days, late_rate_now   AS late_rate FROM {G}.promise_what_if
UNION ALL
SELECT shipping_mode, 'After reset' AS scenario, promised_new AS promised_days, late_rate_after AS late_rate FROM {G}.promise_what_if
""")

# COMMAND ----------

# Forecast backtest in long format (one line per series) for a multi-line chart
spark.sql(f"""
CREATE OR REPLACE VIEW {G}.vw_forecast_backtest AS
SELECT category_name, week_start, 'Actual'         AS series, actual   AS units FROM {G}.demand_forecast_backtest
UNION ALL
SELECT category_name, week_start, 'Naive forecast' AS series, fc_naive AS units FROM {G}.demand_forecast_backtest
UNION ALL
SELECT category_name, week_start, 'SES forecast'   AS series, fc_ses   AS units FROM {G}.demand_forecast_backtest
""")

# COMMAND ----------

for v in ["vw_orders", "vw_promise_what_if", "vw_forecast_backtest"]:
    print(f"{v:22} {spark.table(f'{G}.{v}').count():>8,} rows")