"""Reusable cleaning functions for the Silver layer.
Kept outside the notebook so they can be unit-tested with pytest."""

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

# Personal data: not needed, never kept past Bronze
PII_COLS = [
    "customer_email", "customer_password", "customer_fname",
    "customer_lname", "customer_street", "customer_zipcode",
]

# Empty/unused columns + exact duplicates of other columns
DROP_COLS = [
    "product_description", "product_image", "order_zipcode",
    "benefit_per_order",        # = order_profit_per_order
    "sales_per_customer",       # = order_item_total
    "order_customer_id",        # = customer_id
    "order_item_cardprod_id",   # = product_card_id
    "product_category_id",      # = category_id
    "order_item_product_price", # = product_price
    "_rescued_data",
]

# Pairs we claim are duplicates (verified in the notebook before dropping)
DUPLICATE_PAIRS = [
    ("benefit_per_order", "order_profit_per_order"),
    ("sales_per_customer", "order_item_total"),
    ("order_customer_id", "customer_id"),
    ("order_item_cardprod_id", "product_card_id"),
    ("product_category_id", "category_id"),
    ("order_item_product_price", "product_price"),
]

INT_COLS = [
    "days_for_shipping_real", "days_for_shipment_scheduled", "late_delivery_risk",
    "category_id", "customer_id", "department_id", "order_id", "order_item_id",
    "order_item_quantity", "product_card_id", "product_status",
]

DOUBLE_COLS = [
    "latitude", "longitude", "order_item_discount", "order_item_discount_rate",
    "order_item_profit_ratio", "sales", "order_item_total",
    "order_profit_per_order", "product_price",
]

DATE_FMT = "M/d/yyyy H:mm"   # e.g. 1/31/2018 22:56


def clean_orders(df: DataFrame, country_map: dict) -> DataFrame:
    """Bronze -> Silver cleaning. Each step is one simple, explainable rule."""

    # 1. Drop PII, empty, and duplicate columns
    df = df.drop(*[c for c in PII_COLS + DROP_COLS if c in df.columns])

    # 2. Trim spaces from all text columns
    for c, dtype in df.dtypes:
        if dtype == "string":
            df = df.withColumn(c, F.trim(F.col(c)))

    # 3. Fix data types (try_cast = bad values become NULL instead of crashing)
    for c in INT_COLS:
        df = df.withColumn(c, F.expr(f"try_cast({c} AS INT)"))
    for c in DOUBLE_COLS:
        df = df.withColumn(c, F.expr(f"try_cast({c} AS DOUBLE)"))

    # 4. Parse dates
    df = (
        df.withColumn("order_ts", F.expr(f"try_to_timestamp(order_date_dateorders, '{DATE_FMT}')"))
          .withColumn("shipping_ts", F.expr(f"try_to_timestamp(shipping_date_dateorders, '{DATE_FMT}')"))
          .withColumn("order_date", F.to_date("order_ts"))
          .drop("order_date_dateorders", "shipping_date_dateorders")
    )

    # 5. Translate Spanish country names to English
    if country_map:
        mapping = F.create_map(*[F.lit(x) for kv in country_map.items() for x in kv])
        for c in ["order_country", "customer_country"]:
            df = df.withColumn(c, F.coalesce(mapping[F.col(c)], F.col(c)))

    # 6. Customer state must be a 2-letter code; anything else becomes NULL
    df = df.withColumn(
        "customer_state",
        F.when(F.col("customer_state").rlike("^[A-Z]{2}$"), F.col("customer_state")),
    )

    # 7. Helpful derived columns
    df = (
        df.withColumn("shipping_delay_days",
                      F.col("days_for_shipping_real") - F.col("days_for_shipment_scheduled"))
          .withColumn("is_loss", F.col("order_profit_per_order") < 0)
    )
    return df


def dedupe_latest(df: DataFrame, key: str = "order_item_id") -> DataFrame:
    """Keep only the most recently ingested row per key."""
    w = Window.partitionBy(key).orderBy(F.col("_ingested_at").desc())
    return df.withColumn("_rn", F.row_number().over(w)).filter("_rn = 1").drop("_rn")