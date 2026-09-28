import numpy as np
import pandas as pd

from typing import Literal

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.ensemble import RandomForestRegressor

from fastapi import HTTPException


# ============================================================
# PREPARE FORECAST DATA
# ============================================================

def prepare_forecast_df(
    df: pd.DataFrame,
    frequency: Literal["raw", "weekly", "monthly"]
) -> pd.DataFrame:

    df = df.copy()

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df = df.dropna(subset=["Date"])

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

        if "Amount" in df.columns:

            df["Amount"] = (
                df["Amount"]
                .abs()
            )

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

        if "Amount" in df.columns:

            df["Amount"] = (
                df["Amount"]
                .abs()
            )

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

    # --------------------------------------------------------
    # CHECK DATE COLUMN
    # --------------------------------------------------------

    if date_column not in df.columns:

        raise HTTPException(
            status_code=400,
            detail=f"Date column '{date_column}' not found."
        )

    # --------------------------------------------------------
    # CHECK TARGET COLUMN
    # --------------------------------------------------------

    if target_column not in df.columns:

        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_column}' not found."
        )

    # --------------------------------------------------------
    # CONVERT DATE
    # --------------------------------------------------------

    df[date_column] = pd.to_datetime(
        df[date_column],
        errors="coerce"
    )

    # --------------------------------------------------------
    # CONVERT TARGET
    # --------------------------------------------------------

    df[target_column] = pd.to_numeric(
        df[target_column],
        errors="coerce"
    )

    # --------------------------------------------------------
    # REMOVE INVALID VALUES
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            date_column,
            target_column
        ]
    )

    # --------------------------------------------------------
    # MAKE TARGET POSITIVE
    # --------------------------------------------------------

    df[target_column] = (
        df[target_column]
        .abs()
    )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    df = df.sort_values(
        date_column
    )

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

    elif frequency == "monthly":

        df = (
            df.set_index(date_column)
              .resample("ME")[target_column]
              .sum()
              .to_frame()
        )

    # ========================================================
    # 3. CREATE TIME SERIES
    # ========================================================

    y = (
        df[target_column]
        .astype(float)
    )

    # ========================================================
    # 4. CHECK DATA
    # ========================================================

    if y.empty:

        raise HTTPException(
            status_code=400,
            detail="No valid data available for forecasting."
        )

    # ========================================================
    # 5. VALIDATION SETTINGS
    # ========================================================

    if frequency == "monthly":

        seasonal_period = 12
        initial_train_size = 24

    else:

        seasonal_period = 52
        initial_train_size = 104

    # ========================================================
    # 6. CHECK MINIMUM DATA
    # ========================================================

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
    # 7. WAPE FUNCTION
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

        wape = (
            np.sum(
                np.abs(
                    actual - forecast
                )
            )
            / denominator
        ) * 100

        return float(wape)

    # ========================================================
    # 8. ACCURACY FUNCTION
    # ========================================================

    def calculate_accuracy(
        actual,
        forecast
    ):

        wape = calculate_wape(
            actual,
            forecast
        )

        if not np.isfinite(wape):

            return np.nan

        accuracy = 100 - wape

        # Prevent negative accuracy
        accuracy = max(
            0,
            accuracy
        )

        return float(accuracy)

    # ========================================================
    # 9. MODEL RESULT STORAGE
    # ========================================================

    model_results = {

        "Seasonal Naive": {
            "actual": [],
            "forecast": []
        },

        "ETS": {
            "actual": [],
            "forecast": []
        },

        "SARIMA": {
            "actual": [],
            "forecast": []
        },

        "Random Forest": {
            "actual": [],
            "forecast": []
        }
    }

    # ========================================================
    # 10. WALK-FORWARD VALIDATION
    # ========================================================

    test = y.iloc[
        initial_train_size:
    ]

    for date, actual_value in test.items():

        # ====================================================
        # HISTORY
        # ====================================================

        history = y.loc[
            :date
        ].iloc[:-1]

        # ====================================================
        # SEASONAL NAIVE
        # ====================================================

        try:

            if len(history) >= seasonal_period:

                naive_prediction = float(
                    history.iloc[
                        -seasonal_period
                    ]
                )

            else:

                naive_prediction = float(
                    history.iloc[-1]
                )

        except Exception:

            naive_prediction = np.nan

        model_results[
            "Seasonal Naive"
        ]["actual"].append(
            actual_value
        )

        model_results[
            "Seasonal Naive"
        ]["forecast"].append(
            naive_prediction
        )

        # ====================================================
        # ETS
        # ====================================================

        try:

            ets_model = ExponentialSmoothing(
                history,
                trend="add",
                seasonal="add",
                seasonal_periods=seasonal_period,
                initialization_method="estimated"
            )

            ets_fitted = ets_model.fit(
                optimized=True,
                use_brute=True
            )

            ets_prediction = float(
                ets_fitted.forecast(
                    steps=1
                ).iloc[0]
            )

        except Exception:

            ets_prediction = np.nan

        model_results[
            "ETS"
        ]["actual"].append(
            actual_value
        )

        model_results[
            "ETS"
        ]["forecast"].append(
            ets_prediction
        )

        # ====================================================
        # SARIMA
        # ====================================================

        try:

            sarima_model = SARIMAX(
                history.values,
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
                sarima_fitted.forecast(
                    steps=1
                )[0]
            )

        except Exception:

            sarima_prediction = np.nan

        model_results[
            "SARIMA"
        ]["actual"].append(
            actual_value
        )

        model_results[
            "SARIMA"
        ]["forecast"].append(
            sarima_prediction
        )

        # ====================================================
        # RANDOM FOREST
        # ====================================================

        try:

            rf_history = history.copy()

            # Create lag features
            rf_df = pd.DataFrame({
                "y": rf_history.values
            })

            rf_df["lag_1"] = rf_df["y"].shift(1)
            rf_df["lag_2"] = rf_df["y"].shift(2)
            rf_df["lag_3"] = rf_df["y"].shift(3)

            rf_df["rolling_mean_3"] = (
                rf_df["y"]
                .shift(1)
                .rolling(3)
                .mean()
            )

            rf_df = rf_df.dropna()

            if len(rf_df) >= 10:

                X = rf_df[
                    [
                        "lag_1",
                        "lag_2",
                        "lag_3",
                        "rolling_mean_3"
                    ]
                ]

                target = rf_df["y"]

                rf_model = RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1
                )

                rf_model.fit(
                    X,
                    target
                )

                last_values = rf_history.values

                rf_input = pd.DataFrame({
                    "lag_1": [
                        last_values[-1]
                    ],
                    "lag_2": [
                        last_values[-2]
                    ],
                    "lag_3": [
                        last_values[-3]
                    ],
                    "rolling_mean_3": [
                        np.mean(
                            last_values[-3:]
                        )
                    ]
                })

                rf_prediction = float(
                    rf_model.predict(
                        rf_input
                    )[0]
                )

            else:

                rf_prediction = float(
                    history.iloc[-1]
                )

        except Exception:

            rf_prediction = np.nan

        model_results[
            "Random Forest"
        ]["actual"].append(
            actual_value
        )

        model_results[
            "Random Forest"
        ]["forecast"].append(
            rf_prediction
        )

    # ========================================================
    # 11. FINAL FORECASTS
    # ========================================================

    future_forecasts = {}

    # ========================================================
    # SEASONAL NAIVE
    # ========================================================

    try:

        seasonal_naive_forecast = []

        history_values = list(
            y.values
        )

        for i in range(count):

            if len(history_values) >= seasonal_period:

                prediction = history_values[
                    -seasonal_period
                ]

            else:

                prediction = history_values[-1]

            seasonal_naive_forecast.append(
                float(prediction)
            )

            history_values.append(
                prediction
            )

        future_forecasts[
            "Seasonal Naive"
        ] = seasonal_naive_forecast

    except Exception:

        future_forecasts[
            "Seasonal Naive"
        ] = [np.nan] * count

    # ========================================================
    # ETS
    # ========================================================

    try:

        ets_model = ExponentialSmoothing(
            y,
            trend="add",
            seasonal="add",
            seasonal_periods=seasonal_period,
            initialization_method="estimated"
        )

        ets_fitted = ets_model.fit(
            optimized=True,
            use_brute=True
        )

        ets_forecast = ets_fitted.forecast(
            steps=count
        )

        future_forecasts[
            "ETS"
        ] = [
            float(value)
            for value in ets_forecast
        ]

    except Exception:

        future_forecasts[
            "ETS"
        ] = [np.nan] * count

    # ========================================================
    # SARIMA
    # ========================================================

    try:

        sarima_model = SARIMAX(
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

        sarima_fitted = sarima_model.fit(
            disp=False
        )

        sarima_forecast = (
            sarima_fitted
            .forecast(
                steps=count
            )
        )

        future_forecasts[
            "SARIMA"
        ] = [
            float(value)
            for value in sarima_forecast
        ]

    except Exception:

        future_forecasts[
            "SARIMA"
        ] = [np.nan] * count

    # ========================================================
    # RANDOM FOREST
    # ========================================================

    try:

        rf_df = pd.DataFrame({
            "y": y.values
        })

        rf_df["lag_1"] = rf_df["y"].shift(1)
        rf_df["lag_2"] = rf_df["y"].shift(2)
        rf_df["lag_3"] = rf_df["y"].shift(3)

        rf_df["rolling_mean_3"] = (
            rf_df["y"]
            .shift(1)
            .rolling(3)
            .mean()
        )

        rf_df = rf_df.dropna()

        rf_model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )

        rf_model.fit(
            rf_df[
                [
                    "lag_1",
                    "lag_2",
                    "lag_3",
                    "rolling_mean_3"
                ]
            ],
            rf_df["y"]
        )

        rf_history = list(
            y.values
        )

        rf_forecasts = []

        for _ in range(count):

            rf_input = pd.DataFrame({
                "lag_1": [
                    rf_history[-1]
                ],
                "lag_2": [
                    rf_history[-2]
                ],
                "lag_3": [
                    rf_history[-3]
                ],
                "rolling_mean_3": [
                    np.mean(
                        rf_history[-3:]
                    )
                ]
            })

            prediction = float(
                rf_model.predict(
                    rf_input
                )[0]
            )

            rf_forecasts.append(
                prediction
            )

            rf_history.append(
                prediction
            )

        future_forecasts[
            "Random Forest"
        ] = rf_forecasts

    except Exception:

        future_forecasts[
            "Random Forest"
        ] = [np.nan] * count

    # ========================================================
    # 12. FUTURE DATES
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
    # 13. CREATE FINAL RESPONSE
    # ========================================================

    response = {

        "frequency": frequency,

        "forecast_count": count,

        "historical": [],

        "models": []
    }

    # ========================================================
    # 14. HISTORICAL DATA
    # ========================================================

    for date, value in y.items():

        response[
            "historical"
        ].append({

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
    # 15. MODEL OUTPUT
    # ========================================================

    for model_name in [
        "Seasonal Naive",
        "ETS",
        "SARIMA",
        "Random Forest"
    ]:

        actual_values = (
            model_results[
                model_name
            ]["actual"]
        )

        validation_forecasts = (
            model_results[
                model_name
            ]["forecast"]
        )

        # ----------------------------------------------------
        # WAPE
        # ----------------------------------------------------

        wape = calculate_wape(
            actual_values,
            validation_forecasts
        )

        # ----------------------------------------------------
        # ACCURACY
        # ----------------------------------------------------

        accuracy = calculate_accuracy(
            actual_values,
            validation_forecasts
        )

        # ----------------------------------------------------
        # FORECAST ARRAY
        # ----------------------------------------------------

        forecast_records = []

        for date, value in zip(
            future_dates,
            future_forecasts[
                model_name
            ]
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

        # ----------------------------------------------------
        # ADD MODEL
        # ----------------------------------------------------

        response[
            "models"
        ].append({

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
        })

    # ========================================================
    # 16. RETURN
    # ========================================================

    return response