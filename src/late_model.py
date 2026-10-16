"""Shared definitions for the late-delivery model."""

import pandas as pd

CAT_FEATURES = ["shipping_mode", "market", "order_region", "customer_segment",
                "payment_type", "main_category", "order_weekday"]
NUM_FEATURES = ["scheduled_days", "order_month", "order_hour", "order_qty",
                "n_items", "n_categories", "order_sales", "avg_discount_rate"]
FEATURES = CAT_FEATURES + NUM_FEATURES
TARGET = "is_late"
LEAKAGE = {"delivery_status", "actual_days", "delay_days", "shipping_ts", "order_status"}
SPLIT_DATE = pd.Timestamp("2017-01-01")


def time_split(df, split_date=SPLIT_DATE):
    """Train on the past, test on the future."""
    return df[df.order_ts < split_date], df[df.order_ts >= split_date]


def make_xgb_pipeline(seed=42):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ("num", "passthrough", NUM_FEATURES),
    ])
    model = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=seed,
    )
    return Pipeline([("pre", pre), ("model", model)])