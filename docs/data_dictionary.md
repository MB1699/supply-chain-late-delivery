# Data Dictionary (Gold layer)

Catalog: `supply_chain` · Schema: `gold`
⚠️ = post-delivery information: used for KPIs only, **never** as a model feature.

## Star schema

### fact_order_items
**Grain:** one row per order item (180,519 rows)

| Column | Type | Description |
|---|---|---|
| order_item_id | int | Primary key |
| order_id | int | Order the item belongs to |
| order_date_key / shipping_date_key | int | Keys to `dim_date` (yyyyMMdd) |
| product_key | int | Key to `dim_product` |
| customer_key | int | Key to `dim_customer` |
| location_key | bigint | Key to `dim_location` |
| shipping_mode_key | bigint | Key to `dim_shipping_mode` |
| payment_type | string | DEBIT, TRANSFER, CASH, PAYMENT |
| order_status | string | Order processing status |
| delivery_status | string | ⚠️ Late delivery / On time / Advance / Canceled |
| quantity | int | Units ordered |
| unit_price | double | Product price |
| discount_amount / discount_rate | double | Discount applied |
| gross_sales / net_sales | double | Sales before / after discount |
| profit / profit_ratio | double | Profit on the item |
| scheduled_days | int | Promised shipping days |
| actual_days | int | ⚠️ Actual shipping days |
| delay_days | int | ⚠️ actual_days − scheduled_days |
| is_late | int | **Target:** 1 = late, 0 = on time |
| order_ts / shipping_ts | timestamp | Order and shipping time |

### dim_date
**Grain:** one row per day (1,133 rows) · date_key, date, year, quarter, month, month_name, week_of_year, day_of_week, day_name, is_weekend

### dim_product
**Grain:** one row per product (118 rows) · product_key, product_name, category_id, category_name, department_id, department_name, product_price

### dim_customer
**Grain:** one row per customer (20,652 rows), **no PII** · customer_key, customer_segment, customer_city, customer_state, customer_country

### dim_location
**Grain:** one row per destination (3,772 rows) · location_key, market, order_region, order_country, order_state, order_city

### dim_shipping_mode
**Grain:** one row per mode (4 rows) · shipping_mode_key, shipping_mode, min_scheduled_days, max_scheduled_days

## Analytics tables

| Table | Grain | Purpose |
|---|---|---|
| baseline_kpis | 1 row | Baseline delivery KPIs (delivered orders) |
| order_features | 1 row per delivered order | Model features known at order time + target |
| late_risk_predictions | 1 row per 2017+ order | Scores from rule, logistic regression, XGBoost |
| order_risk_scores | 1 row per 2017+ order | Late risk, risk tier, top 3 SHAP reasons, expedite decision |
| expedite_policy_comparison | 1 row per policy | Net savings of each expedite policy (illustrative costs) |
| promise_what_if | 1 row per shipping mode | Late rate now vs after resetting promises |
| root_cause_tests | 1 row per test | Statistical test results (p-values, effect sizes) |
| demand_forecast_backtest | 1 row per category-week (2017) | Actual demand + forecast from each method |
| forecast_metrics | 1 row per method | WAPE, bias, improvement vs naive |
| demand_forecast_future | 1 row per category-week (next 13 weeks) | Future forecast from the best method |
| inventory_sim_weekly | 1 row per policy-category-week | **Simulated** inventory (demand is real) |
| inventory_kpis | 1 row per policy-category | Fill rate, stockouts, turnover (simulated) |
| inventory_policy_comparison | 1 row per policy | Naive vs forecast restocking (simulated, 2017) |

## Dashboard views

| View | Purpose |
|---|---|
| vw_orders | Delivered orders, one row per order, with risk scores (2017+) |
| vw_promise_what_if | Promise what-if in long format for grouped bars |
| vw_forecast_backtest | Actual vs forecast series in long format for line charts |

## Silver
| Table | Purpose |
|---|---|
| silver.orders | Cleaned order items (types fixed, PII removed, countries translated, deduplicated) |
| silver.dq_results | Data quality check results, appended every pipeline run |