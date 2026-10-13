
import numpy as np
import pandas as pd


FEATURES = [
    "quarter",
    "month",
    "year",
    "dayofyear",
    "weekofyear",
    "sin_month",
    "cos_month",
    "lag_1",
    "lag_2",
    "lag_4",
    "lag_8",
    "lag_13",
    "lag_26",
    "lag_52",
    "rolling_mean_4",
    "rolling_mean_8",
    "rolling_mean_13",
]


def extract_weekly_features(weekly_df):
    weekly_df = weekly_df.copy()

    weekly_df["Date"] = pd.to_datetime(
        weekly_df["Date"], errors="coerce"
    )
    weekly_df["Amount"] = pd.to_numeric(
        weekly_df["Amount"], errors="coerce"
    )

    weekly_df = weekly_df.dropna(
        subset=["Date", "Amount"]
    )

    weekly_df = (
        weekly_df.sort_values("Date")
        .reset_index(drop=True)
    )

    # Calendar features
    weekly_df["quarter"] = weekly_df["Date"].dt.quarter
    weekly_df["month"] = weekly_df["Date"].dt.month
    weekly_df["year"] = weekly_df["Date"].dt.year
    weekly_df["dayofyear"] = weekly_df["Date"].dt.dayofyear

    weekly_df["weekofyear"] = (
        weekly_df["Date"].dt.isocalendar().week.astype(int)
    )

    # Cyclical month features
    weekly_df["sin_month"] = np.sin(
        2 * np.pi * weekly_df["month"] / 12
    )
    weekly_df["cos_month"] = np.cos(
        2 * np.pi * weekly_df["month"] / 12
    )

    # Lag features
    for lag in [1, 2, 4, 8, 13, 26, 52]:
        weekly_df[f"lag_{lag}"] = (
            weekly_df["Amount"].shift(lag)
        )

    # Rolling features: previous weeks only
    for window in [4, 8, 13]:
        weekly_df[f"rolling_mean_{window}"] = (
            weekly_df["Amount"]
            .shift(1)
            .rolling(window=window)
            .mean()
        )

    data = weekly_df.dropna(
        subset=FEATURES + ["Amount"]
    ).copy()

    X = data[FEATURES]
    y = data["Amount"]

    return X, y, data


def make_future_features(history, future_date):
    history = list(history)

    if len(history) < 52:
        raise ValueError(
            "At least 52 historical weekly values are required."
        )

    future_date = pd.Timestamp(future_date)

    month = future_date.month

    row = {
        "quarter": future_date.quarter,
        "month": month,
        "year": future_date.year,
        "dayofyear": future_date.dayofyear,
        "weekofyear": int(
            future_date.isocalendar().week
        ),
        "sin_month": np.sin(2 * np.pi * month / 12),
        "cos_month": np.cos(2 * np.pi * month / 12),
    }

    for lag in [1, 2, 4, 8, 13, 26, 52]:
        row[f"lag_{lag}"] = history[-lag]

    for window in [4, 8, 13]:
        row[f"rolling_mean_{window}"] = float(
            np.mean(history[-window:])
        )

    return pd.DataFrame([row], columns=FEATURES)
