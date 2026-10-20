# User Acceptance Testing (UAT)

**Project:** Late Delivery Prediction & Demand Forecasting
**Tester:** Muskan · **Environment:** Databricks Free Edition (serverless), branch `dev`
**Result:** ✅ All "Must" stories passed

| Status key | Meaning |
|---|---|
| ✅ Pass | Works as the acceptance criteria describe |
| ⚠️ Pass with note | Works, with a known limitation documented |
| ❌ Fail | Does not meet the criteria |

## Epic 1: Data Foundation

| ID | Story | Test steps | Expected result | Actual result | Status |
|---|---|---|---|---|---|
| UAT-01 | US-1: Raw data saved in Bronze | Run `01a_simulate_feed` + `01_bronze_ingestion` with all batches | Bronze rows = source rows; each file logged once | 180,519 rows in Bronze across 10 files, matching the source; rerun loaded 0 duplicates | ✅ |
| UAT-02 | US-5: Data arrives in batches | Release 1 batch, ingest, release another, ingest | Only the new batch is loaded each time | 20K → 40K → 60K rows; earlier batches kept their original load time | ✅ |
| UAT-03 | US-2: Clean data in Silver | Run `02_silver_cleaning`, check `silver.dq_results` | No duplicates, dates parsed, PII removed, critical checks pass | All 6 critical checks pass; 2 warnings explained (4,423 = cancelled orders; 3 minor rows) | ✅ |
| UAT-04 | US-3: Star schema in Gold | Run `03_gold_star_schema` | Fact + 5 dimensions; no orphan keys; rows match Silver | 180,519 fact rows; all 8 integrity checks pass | ✅ |
| UAT-05 | US-4: Pipeline runs as one job | Run job `supply_chain_daily_pipeline` | All tasks succeed in order; rerun is safe | 5/5 tasks green; rerun left row counts unchanged | ✅ |
| UAT-06 | US-19: Automatic quality checks | Open a PR | Unit tests run automatically; DQ checks run on every load | GitHub Actions runs all unit tests on every PR (all pass); DQ results appended each run | ✅ |

## Epic 2: Why Orders Are Late

| ID | Story | Test steps | Expected result | Actual result | Status |
|---|---|---|---|---|---|
| UAT-07 | US-6: Factors linked to late orders | Run `06_root_cause` | Tests on shipping mode, region, category, with effect sizes, explained in plain words | Shipping mode is the dominant driver; region, segment, payment, category negligible; Second and Standard Class have the same actual delivery days | ✅ |

## Epic 3: Predicting Late Orders

| ID | Story | Test steps | Expected result | Actual result | Status |
|---|---|---|---|---|---|
| UAT-08 | US-8: Late-risk score per order | Run `07_late_delivery_model`; check leakage guard | Probability per order; no leakage columns; beats a simple baseline | XGBoost ROC-AUC 0.775 vs 0.743 for the shipping-mode rule; leakage guard passes | ⚠️ Gain over the rule is small; within Second/Standard Class, lateness isn't predictable |
| UAT-09 | US-9: Risk tiers | Run `08_risk_scoring`; check tier summary | Low/Medium/High with documented cut-offs; late rate rises with tier | High 90.0% late · Medium 56.1% · Low 37.6%; cut-offs in `config.yaml` | ✅ |
| UAT-10 | US-10: Why an order is risky | Check `top_reasons` in `order_risk_scores` | Top 3 reasons per order | Every scored order has 3 SHAP reasons (e.g., shipping mode, scheduled days, order hour) | ✅ |
| UAT-11 | US-11: Cost-based expedite advice | Check policy comparison + sensitivity | Rule compares late cost vs expedite cost; assumptions written down | Cost rule gives the best net savings (+$4,120); break-even risk 0.889; sensitivity from $5 to $40 shown | ⚠️ Costs are illustrative assumptions |

## Epic 4: Demand Forecasting & Inventory

| ID | Story | Test steps | Expected result | Actual result | Status |
|---|---|---|---|---|---|
| UAT-12 | US-13: Forecast per category | Run `09_demand_forecast` | 1–3 month forecast; error compared with naive | SES: 10.5% WAPE vs 13.5% naive (22% better); 13-week forecast saved | ✅ (WAPE used instead of MAPE because of zero-demand weeks) |
| UAT-13 | US-14: Forecast vs actual | Check backtest chart + dashboard | Both shown on a held-back period | 2017 backtest chart for each category | ✅ |
| UAT-14 | US-15: Forecast's effect on stockouts | Check inventory policy comparison | Old rule vs forecast-based restocking compared | Fill rate 99.97% for both; safety-stock test shows a 32% stock reduction possible | ⚠️ Inventory is simulated |

## Epic 5: Dashboard

| ID | Story | Test steps | Expected result | Actual result | Status |
|---|---|---|---|---|---|
| UAT-15 | US-16: Executive page | Open published dashboard → Executive | KPIs with trends | 4 KPI counters match baseline (62,897 orders · 57.3% late · 1.62 days · $18.08M); trend, mode, and what-if charts | ✅ |
| UAT-16 | US-17: Operations page | Open Operations; filter by region and mode | Risky orders with filters | Risk tiers, order list with SHAP reasons, filters work | ✅ |
| UAT-17 | US-18: Forecast page | Open Forecast & Inventory; change category | Forecast by category with filters | Category filter updates the forecast chart | ✅ |

## Known limitations (accepted)
1. DataCo is an academic dataset that appears partly simulated; results show the **method**, not real business conclusions.
2. Inventory levels and cost figures are **simulated / illustrative**.
3. Order volume drops ~80% from Oct 2017; forecasting uses Jan 2015 to Sep 2017.
4. Tableau was replaced by Databricks AI/BI (live connection to Gold, no export needed).

## Sign-off
| Role | Name | Decision | Date |
|---|---|---|---|
| Business Analyst / Product Owner | Muskan | Accepted | |
| Logistics Manager *(persona)* | — | Accepted | |
| Inventory Planner *(persona)* | — | Accepted | |