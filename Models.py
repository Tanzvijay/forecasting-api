import numpy as np
import pandas as pd

from typing import Literal

from fastapi import HTTPException

import repo

from forecasting.holtwinters import forecast_holt_winters
from forecasting.holt_linear import forecast_holt_linear
from forecasting.arima import forecast_arima


# ============================================================
# DETECT FREQUENCY FROM DATE
# ============================================================

def detect_frequency(
    df: pd.DataFrame,
    date_column: str = "Date"
) -> str:

    if date_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Date column '{date_column}' not found."
        )

    dates = (
        pd.to_datetime(df[date_column], errors="coerce")
        .dropna()
        .sort_values()
        .drop_duplicates()
    )

    if len(dates) < 2:
        raise HTTPException(
            status_code=400,
            detail="Not enough date values to detect frequency."
        )

    differences = dates.diff().dropna().dt.total_seconds() / 86400
    median_days = differences.median()

    if median_days <= 2:
        return "daily"
    elif median_days <= 8:
        return "weekly"
    elif median_days <= 31:
        return "monthly"

    return "unknown"


# ============================================================
# PREPARE FORECAST DATA
#
# This is the ONLY place where weekly / monthly aggregation
# happens. It returns a dataframe with "Date" and "Amount"
# columns (Date is a regular column, not the index).
# ============================================================

def prepare_forecast_df(
    df: pd.DataFrame,
    frequency: Literal["raw", "weekly", "monthly"]
) -> pd.DataFrame:

    df = df.copy()

    if "Date" not in df.columns:
        raise HTTPException(
            status_code=400,
            detail="Date column 'Date' not found."
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="No valid date values available."
        )

    # RAW: do not aggregate here.
    if frequency == "raw":
        return df.sort_values("Date").reset_index(drop=True)

    elif frequency == "weekly":
        df = (
            df.set_index("Date")
              .resample("W-SUN")
              .sum(numeric_only=True)
              .reset_index()
        )

    elif frequency == "monthly":
        df = (
            df.set_index("Date")
              .resample("ME")
              .sum(numeric_only=True)
              .reset_index()
        )

    else:
        raise HTTPException(
            status_code=400,
            detail="Frequency must be 'raw', 'weekly' or 'monthly'."
        )

    if "Amount" in df.columns:
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").abs()
        df = df[df["Amount"] != 0]

    return df.sort_values("Date").reset_index(drop=True)


# ============================================================
# METRICS
# ============================================================

def calculate_wape(actual, forecast):

    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    valid = np.isfinite(actual) & np.isfinite(forecast)

    actual = actual[valid]
    forecast = forecast[valid]

    if len(actual) == 0:
        return np.inf

    denominator = np.sum(np.abs(actual))

    if denominator == 0:
        return np.inf

    return float(
        (np.sum(np.abs(actual - forecast)) / denominator) * 100
    )


def calculate_accuracy(actual, forecast):

    wape = calculate_wape(actual, forecast)

    if not np.isfinite(wape):
        return np.nan

    return max(0.0, 100.0 - wape)


# ============================================================
# FORECAST OUTPUT
# ============================================================

def generate_forecast_output(
    frequency: Literal["raw", "weekly", "monthly"],
    count: int
):

    # ========================================================
    # 1. CHECK FILTERED DATA
    # ========================================================

    if repo.latest_filtered_df is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "No filtered data available. "
                "Please call the table data endpoint first."
            )
        )

    df = repo.latest_filtered_df.copy()

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="Filtered dataframe is empty."
        )

    # ========================================================
    # 2. GET SELECTED DATE / TARGET COLUMNS
    # ========================================================

    date_column = getattr(repo, "latest_date_column", None)
    target_column = getattr(repo, "latest_target_column", None)

    if not date_column:
        raise HTTPException(
            status_code=400,
            detail=(
                "Date column information is not available. "
                "Please call /tables/{table_name} first."
            )
        )

    if not target_column:
        raise HTTPException(
            status_code=400,
            detail=(
                "Target column information is not available. "
                "Please call /tables/{table_name} first."
            )
        )

    if date_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    f"Date column '{date_column}' "
                    f"not found in filtered dataframe."
                ),
                "available_columns": df.columns.tolist()
            }
        )

    if target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    f"Target column '{target_column}' "
                    f"not found in filtered dataframe."
                ),
                "available_columns": df.columns.tolist()
            }
        )

    # ========================================================
    # 3. STANDARDIZE INTERNAL COLUMN NAMES
    #
    # Forecasting code always uses "Date" and "Amount" from
    # here on. date_column / target_column are not used again.
    # ========================================================

    df["Date"] = pd.to_datetime(df[date_column], errors="coerce")
    df["Amount"] = pd.to_numeric(df[target_column], errors="coerce")

    df = df.dropna(subset=["Date", "Amount"])

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="No valid date/target data available for forecasting."
        )

    # Make target positive
    df["Amount"] = df["Amount"].abs()

    # ========================================================
    # 4. RAW -> DETECT ACTUAL FREQUENCY
    # ========================================================

    if frequency == "raw":

        frequency = detect_frequency(df, "Date")

        if frequency == "unknown":
            raise HTTPException(
                status_code=400,
                detail=(
                    "Unable to detect forecasting frequency "
                    "from the selected date column."
                )
            )

        if frequency == "daily":
            raise HTTPException(
                status_code=400,
                detail=(
                    "Daily frequency was detected, but forecasting "
                    "currently supports only weekly and monthly data. "
                    "Please choose 'weekly' or 'monthly' explicitly to "
                    "aggregate the data."
                )
            )

    # ========================================================
    # 5. VALIDATE FREQUENCY
    # ========================================================

    if frequency not in ["weekly", "monthly"]:
        raise HTTPException(
            status_code=400,
            detail="Frequency must be 'raw', 'weekly' or 'monthly'."
        )

    # ========================================================
    # 6. PREPARE DATA (aggregation happens exactly once, here)
    # ========================================================

    df = prepare_forecast_df(df=df, frequency=frequency)

    if df.empty or "Date" not in df.columns or "Amount" not in df.columns:
        raise HTTPException(
            status_code=400,
            detail="No valid data available for forecasting."
        )

    # ========================================================
    # 7. SORT + INDEX BY DATE
    # ========================================================

    df = (
        df.sort_values("Date")
          .drop_duplicates(subset="Date", keep="last")
          .set_index("Date")
    )

    # ========================================================
    # 8. TARGET SERIES
    # ========================================================

    y = df["Amount"].astype(float)

    if y.empty:
        raise HTTPException(
            status_code=400,
            detail="No valid data available for forecasting."
        )

    # ========================================================
    # 9. TRAIN / TEST SPLIT
    # ========================================================

    train_ratio = 0.80
    initial_train_size = int(len(y) * train_ratio)

    # ========================================================
    # 10. SEASONAL PERIOD
    # ========================================================

    seasonal_period = 12 if frequency == "monthly" else 52

    # ========================================================
    # 11. MINIMUM DATA CHECK
    # ========================================================

    if initial_train_size <= seasonal_period:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Not enough data for {frequency} forecasting. "
                f"At least {seasonal_period + 1} observations "
                f"are required in the initial training period."
            )
        )

    # ========================================================
    # 12. TEST DATA + HISTORIES
    # ========================================================

    test = y.iloc[initial_train_size:]

    hw_history = y.iloc[:initial_train_size].copy()
    holt_linear_history = list(y.iloc[:initial_train_size].values)

    hw_actuals = []
    hw_forecasts = []

    holt_linear_actuals = []
    holt_linear_forecasts = []

    # ========================================================
    # 13. WALK-FORWARD VALIDATION
    # ========================================================

    for date, actual_value in test.items():

        # ---------------- Holt-Winters ----------------
        try:
            hw_prediction = float(
                forecast_holt_winters(
                    hw_history,
                    seasonal_period,
                    1
                ).iloc[0]
            )
        except Exception:
            hw_prediction = np.nan

        hw_actuals.append(actual_value)
        hw_forecasts.append(hw_prediction)

        # ---------------- Holt Linear ----------------
        try:
            holt_linear_prediction = float(
                np.asarray(
                    forecast_holt_linear(
                        holt_linear_history,
                        1
                    )
                )[0]
            )
        except Exception as e:
            print(f"Holt Linear validation error: {repr(e)}")
            holt_linear_prediction = np.nan

        holt_linear_actuals.append(actual_value)
        holt_linear_forecasts.append(holt_linear_prediction)

        # ---------------- Expanding window ----------------
        hw_history.loc[date] = actual_value
        holt_linear_history.append(actual_value)

    # ========================================================
    # 14. HOLT-WINTERS / HOLT LINEAR METRICS
    # ========================================================

    hw_wape = calculate_wape(hw_actuals, hw_forecasts)
    hw_accuracy = calculate_accuracy(hw_actuals, hw_forecasts)

    holt_linear_wape = calculate_wape(
        holt_linear_actuals,
        holt_linear_forecasts
    )
    holt_linear_accuracy = calculate_accuracy(
        holt_linear_actuals,
        holt_linear_forecasts
    )

    # ========================================================
    # 15. ARIMA VALIDATION
    # ========================================================

    arima_wape = np.inf
    arima_accuracy = np.nan
    arima_order = None

    try:
        arima_train = y.iloc[:initial_train_size].values

        arima_test_forecast, arima_order = forecast_arima(
            arima_train,
            len(test)
        )

        arima_test_forecast = np.asarray(arima_test_forecast, dtype=float)

        arima_wape = calculate_wape(test.values, arima_test_forecast)
        arima_accuracy = calculate_accuracy(test.values, arima_test_forecast)

    except Exception as e:
        print(f"ARIMA validation error: {e}")
        arima_wape = np.inf
        arima_accuracy = np.nan
        arima_order = None

    # ========================================================
    # 16. FINAL HOLT-WINTERS FORECAST
    # ========================================================

    try:
        hw_future = np.asarray(
            forecast_holt_winters(y, seasonal_period, count),
            dtype=float
        )
    except Exception as e:
        print(f"Holt-Winters forecast error: {e}")
        hw_future = np.array([np.nan] * count)

    # ========================================================
    # 17. FINAL HOLT LINEAR FORECAST
    # ========================================================

    try:
        holt_linear_future = np.asarray(
            forecast_holt_linear(y, count),
            dtype=float
        )
    except Exception as e:
        print(f"Holt Linear forecast error: {repr(e)}")
        holt_linear_future = np.array([np.nan] * count)

    # ========================================================
    # 18. FINAL ARIMA FORECAST
    # ========================================================

    try:
        arima_future, final_arima_order = forecast_arima(y.values, count)

        arima_future = np.asarray(arima_future, dtype=float)

        if final_arima_order is not None:
            arima_order = final_arima_order

    except Exception as e:
        print(f"ARIMA forecast error: {e}")
        arima_future = np.array([np.nan] * count)

    # ========================================================
    # 19. FUTURE DATES
    # ========================================================

    if frequency == "weekly":
        future_dates = pd.date_range(
            start=y.index[-1] + pd.Timedelta(weeks=1),
            periods=count,
            freq="W-SUN"
        )
    else:
        future_dates = pd.date_range(
            start=y.index[-1] + pd.offsets.MonthEnd(1),
            periods=count,
            freq="ME"
        )

    # ========================================================
    # 20. HISTORICAL RESPONSE
    # ========================================================

    historical = [
        {
            "Date": date.strftime("%Y-%m-%d"),
            "Actual": round(float(value), 2),
            "Type": "historical"
        }
        for date, value in y.items()
    ]

    # ========================================================
    # 21. MODEL RESPONSE BUILDER
    # ========================================================

    def build_model_response(
        model_name,
        accuracy,
        wape,
        forecast_values,
        order=None
    ):

        forecast_records = []

        for date, value in zip(future_dates, forecast_values):
            forecast_records.append({
                "Date": date.strftime("%Y-%m-%d"),
                "Forecast": (
                    None
                    if not np.isfinite(value)
                    else round(float(value), 2)
                )
            })

        response = {
            "model": model_name,
            "accuracy": (
                None
                if not np.isfinite(accuracy)
                else f"{accuracy:.2f}%"
            ),
            "wape": (
                None
                if not np.isfinite(wape)
                else f"{wape:.2f}%"
            ),
            "forecast": forecast_records
        }

        if order is not None:
            response["order"] = list(order)

        return response

    # ========================================================
    # 22. BUILD MODELS
    # ========================================================

    models = [
        build_model_response(
            "Holt-Winters",
            hw_accuracy,
            hw_wape,
            hw_future
        ),
        build_model_response(
            "Holt Linear",
            holt_linear_accuracy,
            holt_linear_wape,
            holt_linear_future
        ),
        build_model_response(
            "ARIMA",
            arima_accuracy,
            arima_wape,
            arima_future,
            arima_order
        ),
    ]

    # ========================================================
    # 23. SELECT TOP MODEL
    # ========================================================

    valid_models = [m for m in models if m["wape"] is not None]

    if valid_models:
        top_model = min(
            valid_models,
            key=lambda m: float(m["wape"].replace("%", ""))
        )
        top_model_name = top_model["model"]
        top_model_wape = top_model["wape"]
        top_model_accuracy = top_model["accuracy"]
    else:
        top_model_name = None
        top_model_wape = None
        top_model_accuracy = None

    # ========================================================
    # 24. RETURN RESPONSE
    # ========================================================

    return {
        "frequency": frequency,
        "forecast_count": count,
        "historical": historical,
        "models": models,
        "top_model": top_model_name,
        "top_model_wape": top_model_wape,
        "top_model_accuracy": top_model_accuracy,
    }
