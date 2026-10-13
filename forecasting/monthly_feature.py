
import numpy as np
import pandas as pd


FEATURES = [
    "quarter",
    "month",
    "year",
    "sin_month",
    "cos_month",
    "time_index",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "rolling_mean_3",
    "rolling_mean_6",
    "rolling_mean_12",
]


def extract_monthly_features(monthly_df):
    """
    Generate model-ready monthly features.

    Expected columns:
        Date, Amount

    Returns:
        X, y, data
    """
    data = monthly_df.copy()

    data["Date"] = pd.to_datetime(
        data["Date"], errors="coerce"
    )
    data["Amount"] = pd.to_numeric(
        data["Amount"], errors="coerce"
    )

    data = data.dropna(
        subset=["Date", "Amount"]
    )

    data = data.sort_values("Date").reset_index(drop=True)

    # Ensure one row per calendar month.
    data = (
        data.set_index("Date")["Amount"]
        .resample("ME")
        .sum()
        .reset_index()
    )

    data["quarter"] = data["Date"].dt.quarter
    data["month"] = data["Date"].dt.month
    data["year"] = data["Date"].dt.year

    data["sin_month"] = np.sin(
        2 * np.pi * data["month"] / 12
    )
    data["cos_month"] = np.cos(
        2 * np.pi * data["month"] / 12
    )

    # Trend feature
    data["time_index"] = np.arange(len(data))

    # Lag features
    for lag in [1, 2, 3, 6, 12]:
        data[f"lag_{lag}"] = data["Amount"].shift(lag)

    # Rolling averages use only previous months
    for window in [3, 6, 12]:
        data[f"rolling_mean_{window}"] = (
            data["Amount"]
            .shift(1)
            .rolling(window=window)
            .mean()
        )

    # Remove rows without sufficient historical features
    data = data.dropna(
        subset=FEATURES + ["Amount"]
    ).copy()

    X = data[FEATURES]
    y = data["Amount"]

    return X, y, data


def make_monthly_future_features(history, future_date, time_index):
    """
    Build one future month's feature row.

    history must be ordered oldest to newest and contain
    actual amounts followed by prior predictions.
    """
    history = list(history)

    if len(history) < 12:
        raise ValueError(
            "At least 12 months of history are required."
        )

    future_date = pd.Timestamp(future_date)
    month = future_date.month

    row = {
        "quarter": future_date.quarter,
        "month": month,
        "year": future_date.year,
        "sin_month": np.sin(2 * np.pi * month / 12),
        "cos_month": np.cos(2 * np.pi * month / 12),
        "time_index": time_index,
    }

    for lag in [1, 2, 3, 6, 12]:
        row[f"lag_{lag}"] = float(history[-lag])

    for window in [3, 6, 12]:
        row[f"rolling_mean_{window}"] = float(
            np.mean(history[-window:])
        )

    return pd.DataFrame([row], columns=FEATURES)
