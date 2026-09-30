import numpy as np
import pandas as pd

from typing import Literal

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from fastapi import HTTPException


# ============================================================
# PREPARE FORECAST DATA
# ============================================================

def prepare_forecast_df(
    df: pd.DataFrame,
    frequency: Literal["raw", "weekly", "monthly"]
) -> pd.DataFrame:

    df = df.copy()

    # --------------------------------------------------------
    # Convert Date
    # --------------------------------------------------------

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Date"]
    )

    # --------------------------------------------------------
    # Raw
    # --------------------------------------------------------

    if frequency == "raw":

        return (
            df.sort_values("Date")
              .reset_index(drop=True)
        )

    # --------------------------------------------------------
    # Weekly
    # --------------------------------------------------------

    elif frequency == "weekly":

        df = (
            df.set_index("Date")
              .resample("W-SUN")
              .sum(numeric_only=True)
              .reset_index()
        )

        # Make Amount positive
        if "Amount" in df.columns:

            df["Amount"] = (
                df["Amount"]
                .abs()
            )

        # IMPORTANT:
        # Do NOT remove zero weeks.
        # Keeping them maintains a continuous time series.

    # --------------------------------------------------------
    # Monthly
    # --------------------------------------------------------

    elif frequency == "monthly":

        df = (
            df.set_index("Date")
              .resample("ME")
              .sum(numeric_only=True)
              .reset_index()
        )

        # Make Amount positive
        if "Amount" in df.columns:

            df["Amount"] = (
                df["Amount"]
                .abs()
            )

    # --------------------------------------------------------
    # Final sorting
    # --------------------------------------------------------

    return (
        df.sort_values("Date")
          .reset_index(drop=True)
    )


# ============================================================
# FORECAST OUTPUT
# ============================================================

def generate_forecast_output(
    df: pd.DataFrame,
    date_column: str,
    target_column: str,
    frequency: Literal["weekly", "monthly"],
    count: int
):

    # ========================================================
    # 1. PREPARE DATA
    # ========================================================

    df = df.copy()

    if date_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Date column '{date_column}' not found."
        )

    if target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_column}' not found."
        )

    df[date_column] = pd.to_datetime(
        df[date_column],
        errors="coerce"
    )

    df[target_column] = pd.to_numeric(
        df[target_column],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            date_column,
            target_column
        ]
    )

    df[target_column] = (
        df[target_column]
        .abs()
    )

    df = df.sort_values(date_column)

    # ========================================================
    # 2. AGGREGATE TIME SERIES
    # ========================================================

    if frequency == "weekly":

        df = (
            df.set_index(date_column)
              .resample("W-SUN")[target_column]
              .sum()
              .to_frame()
        )
        df = df[df[target_column] != 0]

    elif frequency == "monthly":

        df = (
            df.set_index(date_column)
              .resample("ME")[target_column]
              .sum()
              .to_frame()
        )
        df = df[df[target_column] != 0]

    y = df[target_column].astype(float)

    if y.empty:
        raise HTTPException(
            status_code=400,
            detail="No valid data available for forecasting."
        )

    # ========================================================
    # 3. VALIDATION SETTINGS
    # ========================================================

    if frequency == "monthly":
        seasonal_period = 12
        initial_train_size = 24
    else:
        seasonal_period = 52
        initial_train_size = 104

    if len(y) <= initial_train_size:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Not enough data for {frequency} forecasting. "
                f"Minimum required observations: "
                f"{initial_train_size + 1}. "
                f"Available: {len(y)}."
            )
        )

    # ========================================================
    # 4. WAPE / ACCURACY
    # Accuracy = 100 - WAPE
    # ========================================================

    def calculate_wape(actual, forecast):

        actual = np.asarray(
            actual,
            dtype=float
        )

        forecast = np.asarray(
            forecast,
            dtype=float
        )

        valid = (
            np.isfinite(actual)
            &
            np.isfinite(forecast)
        )

        actual = actual[valid]
        forecast = forecast[valid]

        if len(actual) == 0:
            return np.inf

        denominator = np.sum(
            np.abs(actual)
        )

        if denominator == 0:
            return np.inf

        return float(
            (
                np.sum(
                    np.abs(
                        actual - forecast
                    )
                )
                / denominator
            ) * 100
        )

    def calculate_accuracy(actual, forecast):

        wape = calculate_wape(
            actual,
            forecast
        )

        if not np.isfinite(wape):
            return np.nan

        return max(
            0.0,
            100.0 - wape
        )

    # ========================================================
    # 5. WALK-FORWARD VALIDATION
    # ========================================================

    test = y.iloc[
        initial_train_size:
    ]

    hw_history = y.iloc[
        :initial_train_size
    ].copy()

    sarima_history = list(
        y.iloc[
            :initial_train_size
        ].values
    )

    hw_actuals = []
    hw_forecasts = []

    sarima_actuals = []
    sarima_forecasts = []

    for date, actual_value in test.items():

        # ----------------------------------------------------
        # HOLT-WINTERS
        # ----------------------------------------------------

        try:

            hw_model = ExponentialSmoothing(
                hw_history,
                trend="add",
                seasonal="add",
                seasonal_periods=seasonal_period,
                initialization_method="estimated"
            )

            hw_fitted = hw_model.fit(
                optimized=True,
                use_brute=True
            )

            hw_prediction = float(
                hw_fitted
                .forecast(steps=1)
                .iloc[0]
            )

        except Exception:

            hw_prediction = np.nan

        hw_actuals.append(
            actual_value
        )

        hw_forecasts.append(
            hw_prediction
        )

        # ----------------------------------------------------
        # SARIMA
        # ----------------------------------------------------

        try:

            sarima_model = SARIMAX(
                sarima_history,
                order=(1, 1, 1),
                seasonal_order=(
                    0,
                    0,
                    0,
                    seasonal_period
                ),
                enforce_stationarity=False,
                enforce_invertibility=False
            )

            sarima_fitted = sarima_model.fit(
                disp=False
            )

            sarima_prediction = float(
                sarima_fitted
                .forecast(steps=1)[0]
            )

        except Exception:

            sarima_prediction = np.nan

        sarima_actuals.append(
            actual_value
        )

        sarima_forecasts.append(
            sarima_prediction
        )

        # ----------------------------------------------------
        # EXPANDING WINDOW
        # ----------------------------------------------------

        hw_history.loc[date] = actual_value

        sarima_history.append(
            actual_value
        )

    # ========================================================
    # 6. CALCULATE EACH MODEL ACCURACY
    # ========================================================

    hw_wape = calculate_wape(
        hw_actuals,
        hw_forecasts
    )

    hw_accuracy = calculate_accuracy(
        hw_actuals,
        hw_forecasts
    )

    sarima_wape = calculate_wape(
        sarima_actuals,
        sarima_forecasts
    )

    sarima_accuracy = calculate_accuracy(
        sarima_actuals,
        sarima_forecasts
    )

    # ========================================================
    # 7. FINAL HOLT-WINTERS FORECAST
    # ========================================================

    try:

        final_hw_model = ExponentialSmoothing(
            y,
            trend="add",
            seasonal="add",
            seasonal_periods=seasonal_period,
            initialization_method="estimated"
        )

        final_hw_fitted = final_hw_model.fit(
            optimized=True,
            use_brute=True
        )

        hw_future = final_hw_fitted.forecast(
            steps=count
        )

    except Exception:

        hw_future = [
            np.nan
        ] * count

    # ========================================================
    # 8. FINAL SARIMA FORECAST
    # ========================================================

    try:

        final_sarima_model = SARIMAX(
            y.values,
            order=(1, 1, 1),
            seasonal_order=(
                0,
                0,
                0,
                seasonal_period
            ),
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        final_sarima_fitted = (
            final_sarima_model.fit(
                disp=False
            )
        )

        sarima_future = (
            final_sarima_fitted
            .forecast(
                steps=count
            )
        )

    except Exception:

        sarima_future = [
            np.nan
        ] * count

    # ========================================================
    # 9. FUTURE DATES
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
    # 10. HISTORICAL RESPONSE
    # ========================================================

    historical = []

    for date, value in y.items():

        historical.append({
            "Date": date.strftime(
                "%Y-%m-%d"
            ),
            "Actual": round(
                float(value),
                2
            ),
            "Type": "historical"
        })

    # ========================================================
    # 11. MODEL RESPONSE BUILDER
    # ========================================================

    def build_model_response(
        model_name,
        accuracy,
        wape,
        forecast_values
    ):

        forecast_records = []

        for date, value in zip(
            future_dates,
            forecast_values
        ):

            forecast_records.append({
                "Date": date.strftime(
                    "%Y-%m-%d"
                ),
                "Forecast": (
                    None
                    if not np.isfinite(value)
                    else round(
                        float(value),
                        2
                    )
                )
            })

        return {
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

    # ========================================================
    # 12. RETURN ALL MODELS
    # ========================================================

    return {
        "frequency": frequency,
        "forecast_count": count,
        "historical": historical,
        "models": [
            build_model_response(
                "Holt-Winters",
                hw_accuracy,
                hw_wape,
                hw_future
            ),
            build_model_response(
                "SARIMA",
                sarima_accuracy,
                sarima_wape,
                sarima_future
            )
        ]
    }

