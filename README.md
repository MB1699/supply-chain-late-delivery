# Supply Chain: Late Delivery Prediction & Demand Forecasting

End-to-end data project built the way a real data team works: Agile sprints, Git feature branches, Pull Requests, automated tests, and an orchestrated pipeline.

## Business problem
Late deliveries hurt customer trust, and stock is planned without a reliable forecast. This project:
1. **Predicts which orders are likely to be late**, so the team can act before shipping.
2. **Forecasts category demand** to support inventory planning.

📄 Full brief: [docs/BRD.md](docs/BRD.md)

## Architecture
```
Mendeley CSV → simulated daily feed → Bronze (Auto Loader) → Silver (clean + DQ) → Gold (star schema + inventory sim)
                                   orchestrated as one Databricks job
```

| Layer | Table(s) | What happens |
|---|---|---|
| Bronze | `bronze.orders_raw` | Raw data as-is + audit columns; each file loaded exactly once |
| Silver | `silver.orders`, `silver.dq_results` | Types fixed, PII removed, countries translated, dedupe via MERGE, data quality checks |
| Gold | `fact_order_items`, `dim_date`, `dim_product`, `dim_customer`, `dim_location`, `dim_shipping_mode` | Star schema with integrity checks |
| Gold | `inventory_sim_weekly`, `inventory_kpis` | **Simulated** weekly inventory using real demand |
| Gold | `baseline_kpis` | Baseline delivery KPIs from EDA |

## Tech stack
Databricks (Free Edition, serverless) · PySpark · Delta Lake · Auto Loader · Databricks Jobs · Python · pandas · matplotlib · pytest · GitHub Actions

## Repo structure
```
config/      settings + business assumptions (config.yaml, country_map.yaml)
docs/        project brief (BRD)
notebooks/   pipeline steps 00–04, EDA (05)
src/         reusable, unit-tested logic (cleaning, inventory simulation)
tests/       pytest unit tests (run automatically on every PR)
jobs/        pipeline job definition
```

## How to run
1. Run `notebooks/00_setup` to create the catalog, schemas, and volume.
2. Upload `DataCoSupplyChainDataset.csv` to `/Volumes/supply_chain/landing/raw_files/source/`.
3. Run the job `supply_chain_daily_pipeline` (see `jobs/pipeline_job.json`).
4. Run `notebooks/05_eda` for exploratory analysis and baseline KPIs.

## Data
**DataCo Smart Supply Chain for Big Data Analysis**, Mendeley Data, DOI: 10.17632/8gx2fvg2k6.5 (~180K order-item rows, ~65K orders, 2015–2018).

**Limitations:** a published academic dataset that appears partly simulated (e.g., First Class is always late, and actual delivery days look evenly spread). Results demonstrate the **method**, not real business conclusions. Inventory levels are **simulated**; order quantity is used as a proxy for demand.

## Data quality findings (Silver)
- All critical checks pass.
- Duplicate columns were verified as identical before being dropped.
- **4,423 rows flagged "late flag inconsistent"** are all *cancelled* orders: they are marked not-late even when actual days exceed scheduled. Cancelled orders were never delivered, so they are **excluded** from late-delivery analysis and modeling.
- **3 rows** with shipping date before order date: minor, noted.
- `delivery_status` maps perfectly to the late flag (data leakage), so it is used for KPIs only, never as a model feature.
- Order volume drops ~80% from Oct 2017 (likely a data artifact); forecasting uses Jan 2015 to Sep 2017.

## Baseline KPIs (delivered orders, cancelled excluded)
| KPI | Value |
|---|---|
| Delivered orders | 62,897 |
| Late rate | 57.31% |
| On-time rate | 42.69% |
| Avg delay when late | 1.62 days |
| Revenue on late orders | $18.08M (57.14% of revenue) |

## EDA key findings
| # | Finding | So what? |
|---|---|---|
| 1 | Shipping mode drives lateness: First Class 100%, Second Class 80%, Same Day 48%, Standard 40% | Main driver to test and model |
| 2 | Second and Standard Class have the **same actual delivery days** (evenly 2–6); only the promised days differ | Lateness is mainly a **promise** problem |
| 3 | First Class is promised 1 day but always delivered in 2 | Reset the promise to match reality |
| 4 | Market, region, segment, payment type, and category all sit near 57% ± a few points | Weak signals |
| 5 | Order-level grain is clean: no mixed shipping modes or late flags within an order | Order-level analysis is safe |

## Inventory baseline (simulated, naive policy)
Fill rate 99.57% · Stockout weeks 3.81% · Lost units 1,666

## Status
- [x] Sprint 1: Requirements (BRD)
- [x] Sprint 2: Data pipeline (Bronze, Silver, Gold, inventory sim, orchestration, tests)
- [ ] Sprint 3: EDA ✅ · root-cause tests · late-delivery model
- [ ] Sprint 4: Forecasting + dashboard
- [ ] Sprint 5: Testing + stakeholder presentation