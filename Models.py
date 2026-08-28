import warnings
import numpy as np
import pandas as pd
from typing import Literal, Optional
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.ensemble import RandomForestRegressor

import numpy as np
import pandas as pd

from fastapi import (

    Depends,
    HTTPException,

)
import repo

from sqlalchemy.orm import Session

WEEKLY_HORIZON = 12
MONTHLY_HORIZON = 12
N_ROLLING_WINDOWS = 3
RANDOM_STATE = 42


# XGBoost and CatBoost are intentionally removed.
MODELS = [
    "Seasonal Naive",
    "ETS",
    "SARIMA",
    "Random Forest",
]


def clean_date_and_target(df, DATE_COL, TARGET_COL):
    df = df.copy()

    df.columns = df.columns.str.strip()

    df[TARGET_COL] = pd.to_numeric(
        df[TARGET_COL]
        .astype(str)
        .str.replace(",", "", regex=False),
        errors="coerce",
    )

    df[DATE_COL] = pd.to_datetime(
        df[DATE_COL],
        errors="coerce",
    )

    return df


def create_weekly_data(df, DATE_COL, TARGET_COL):
    return create_weekly_series(df, DATE_COL, TARGET_COL)


def create_monthly_data(df, DATE_COL, TARGET_COL):
    return create_monthly_series(df, DATE_COL, TARGET_COL)


def create_weekly_series(df, DATE_COL, TARGET_COL):
    df = clean_date_and_target(df, DATE_COL, TARGET_COL)
    df = df.dropna(subset=[DATE_COL, TARGET_COL])

    if df.empty:
        return pd.Series(dtype=float, name=TARGET_COL)

    weekly_series = (
        df.set_index(DATE_COL)[TARGET_COL]
        .resample("W-SUN")
        .sum()
    )

    # Remove incomplete final week.
    if (
        len(weekly_series) > 0
        and weekly_series.index[-1] > df[DATE_COL].max()
    ):
        weekly_series = weekly_series.iloc[:-1]

    weekly_series = (
        weekly_series
        .asfreq("W-SUN")
        .fillna(0)
    )

    return weekly_series.astype(float)


def create_monthly_series(df, DATE_COL, TARGET_COL):
    df = clean_date_and_target(df, DATE_COL, TARGET_COL)
    df = df.dropna(subset=[DATE_COL, TARGET_COL])

    if df.empty:
        return pd.Series(dtype=float, name=TARGET_COL)

    monthly_series = (
        df.set_index(DATE_COL)[TARGET_COL]
        .resample("M")
        .sum()
    )

    # Remove incomplete final month.
    if (
        len(monthly_series) > 0
        and monthly_series.index[-1] > df[DATE_COL].max()
    ):
        monthly_series = monthly_series.iloc[:-1]

    monthly_series = (
        monthly_series
        .asfreq("M")
        .fillna(0)
    )

    return monthly_series.astype(float)


def create_ml_features(series, season):
    data = pd.DataFrame({"y": series.copy()})

    lags = [1, 2, 3, 4, 5, 6, 7, 12]

    if season not in lags:
        lags.append(season)

    if season == 52:
        lags.extend([13, 26, 39])

    for lag in sorted(set(lags)):
        data[f"lag_{lag}"] = data["y"].shift(lag)

    data["rolling_mean_4"] = (
        data["y"].shift(1).rolling(4).mean()
    )
    data["rolling_mean_8"] = (
        data["y"].shift(1).rolling(8).mean()
    )
    data["rolling_mean_12"] = (
        data["y"].shift(1).rolling(12).mean()
    )
    data["rolling_std_4"] = (
        data["y"].shift(1).rolling(4).std()
    )
    data["rolling_std_12"] = (
        data["y"].shift(1).rolling(12).std()
    )

    data["trend"] = np.arange(len(data))
    data["month"] = data.index.month
    data["quarter"] = data.index.quarter
    data["year"] = data.index.year

    data["month_sin"] = np.sin(
        2 * np.pi * data.index.month / 12
    )
    data["month_cos"] = np.cos(
        2 * np.pi * data.index.month / 12
    )

    if season == 52:
        week_number = (
            data.index.isocalendar().week.astype(int)
        )

        data["week_sin"] = np.sin(
            2 * np.pi * week_number / 52
        )
        data["week_cos"] = np.cos(
            2 * np.pi * week_number / 52
        )
    else:
        data["week_sin"] = 0.0
        data["week_cos"] = 0.0

    return data.dropna()


def seasonal_naive_forecast(train, horizon, season):
    values = np.asarray(train, dtype=float)

    if len(values) == 0:
        raise ValueError("Training series is empty.")

    if len(values) < season:
        return np.repeat(values[-1], horizon)

    last_season = values[-season:]

    forecast = np.tile(
        last_season,
        int(np.ceil(horizon / season)),
    )

    return forecast[:horizon]


def fit_ets(train, season):
    try:
        model = ExponentialSmoothing(
            train,
            trend="add",
            seasonal="add",
            seasonal_periods=season,
            initialization_method="estimated",
        )

        return model.fit(optimized=True)

    except Exception:
        model = ExponentialSmoothing(
            train,
            trend="add",
            damped_trend=True,
            initialization_method="estimated",
        )

        return model.fit(optimized=True)


def fit_sarima(train, season):
    candidates = [
        ((0, 1, 0), (0, 0, 0, season)),
        ((1, 1, 0), (0, 0, 0, season)),
        ((0, 1, 1), (0, 0, 0, season)),
        ((1, 1, 1), (0, 0, 0, season)),
        ((1, 1, 0), (1, 0, 0, season)),
        ((0, 1, 1), (0, 0, 1, season)),
        ((1, 1, 1), (1, 0, 0, season)),
        ((1, 1, 1), (0, 0, 1, season)),
    ]

    best_fit = None
    best_aic = np.inf
    best_order = None
    best_seasonal = None

    for order, seasonal_order in candidates:
        try:
            model = SARIMAX(
                train,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )

            fitted = model.fit(
                disp=False,
                maxiter=100,
            )

            if (
                np.isfinite(fitted.aic)
                and fitted.aic < best_aic
            ):
                best_fit = fitted
                best_aic = fitted.aic
                best_order = order
                best_seasonal = seasonal_order

        except Exception:
            continue

    if best_fit is None:
        raise RuntimeError("SARIMA fitting failed.")

    return best_fit, best_order, best_seasonal


def fit_random_forest(train, season):
    train_df = create_ml_features(train, season)

    if train_df.empty:
        raise RuntimeError(
            "Not enough observations for Random Forest."
        )

    X_train = train_df.drop(columns=["y"])
    y_train = train_df["y"]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    return model


def _recursive_random_forest_forecast(train, horizon, season):
    model = fit_random_forest(train, season)
    current_series = train.copy()

    for _ in range(horizon):
        if current_series.index.freq is not None:
            next_date = (
                current_series.index[-1]
                + current_series.index.freq
            )
        else:
            if season == 52:
                next_date = (
                    current_series.index[-1]
                    + pd.Timedelta(days=7)
                )
            else:
                next_date = (
                    current_series.index[-1]
                    + pd.offsets.MonthEnd(1)
                )

        temp = pd.concat(
            [
                current_series,
                pd.Series([np.nan], index=[next_date]),
            ]
        )

        features = create_ml_features(temp, season)

        if features.empty:
            raise RuntimeError(
                "Unable to create future features for Random Forest."
            )

        X_next = features.drop(columns=["y"]).iloc[[-1]]
        prediction = float(model.predict(X_next)[0])

        current_series.loc[next_date] = prediction

    return np.asarray(
        current_series.iloc[-horizon:],
        dtype=float,
    )


def forecast_one_model(train, horizon, season, model_name):
    train = pd.Series(train).astype(float)

    if len(train) == 0:
        raise ValueError("Training series is empty.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if model_name == "Seasonal Naive":
            return seasonal_naive_forecast(
                train,
                horizon,
                season,
            )

        if model_name == "ETS":
            fitted = fit_ets(train, season)

            return np.asarray(
                fitted.forecast(horizon),
                dtype=float,
            )

        if model_name == "SARIMA":
            fitted, _, _ = fit_sarima(train, season)

            return np.asarray(
                fitted.forecast(horizon),
                dtype=float,
            )

        if model_name == "Random Forest":
            return _recursive_random_forest_forecast(
                train,
                horizon,
                season,
            )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


def _mae(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return float(
        np.mean(np.abs(actual - predicted))
    )


def _rmse(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return float(
        np.sqrt(
            np.mean(
                (actual - predicted) ** 2
            )
        )
    )


def _mape(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mask = actual != 0

    if not np.any(mask):
        return np.nan

    return float(
        np.mean(
            np.abs(
                (actual[mask] - predicted[mask])
                / actual[mask]
            )
        )
        * 100
    )


def _smape(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    denominator = (
        np.abs(actual) + np.abs(predicted)
    )

    mask = denominator != 0

    if not np.any(mask):
        return np.nan

    return float(
        np.mean(
            2
            * np.abs(actual[mask] - predicted[mask])
            / denominator[mask]
        )
        * 100
    )


def _bias(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return float(np.mean(predicted - actual))


def _r2(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if len(actual) < 2:
        return np.nan

    denominator = np.sum(
        (actual - np.mean(actual)) ** 2
    )

    if denominator == 0:
        return np.nan

    return float(
        1
        - np.sum((actual - predicted) ** 2)
        / denominator
    )


def _directional_accuracy(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if len(actual) < 2:
        return np.nan

    actual_direction = np.sign(
        np.diff(actual)
    )
    predicted_direction = np.sign(
        np.diff(predicted)
    )

    return float(
        np.mean(
            actual_direction == predicted_direction
        )
        * 100
    )


def _safe_ratio(numerator, denominator):
    if (
        denominator is None
        or not np.isfinite(denominator)
        or denominator <= 1e-12
    ):
        return np.nan

    return float(numerator / denominator)


def evaluate_one_window(
    series,
    train_end,
    test_size,
    season,
):
    train = series.iloc[:train_end]
    test = series.iloc[
        train_end:train_end + test_size
    ]

    if test.empty:
        raise RuntimeError(
            "Validation test set is empty."
        )

    rows = []

    for model_name in MODELS:
        try:
            prediction = forecast_one_model(
                train=train,
                horizon=len(test),
                season=season,
                model_name=model_name,
            )

            actual = test.to_numpy(dtype=float)

            rows.append({
                "Model": model_name,
                "MAE": _mae(actual, prediction),
                "RMSE": _rmse(actual, prediction),
                "MAPE": _mape(actual, prediction),
                "sMAPE": _smape(actual, prediction),
                "Bias": _bias(actual, prediction),
                "R2": _r2(actual, prediction),
                "Directional_Accuracy": (
                    _directional_accuracy(
                        actual,
                        prediction,
                    )
                ),
                "Status": "Success",
            })

        except Exception as e:
            rows.append({
                "Model": model_name,
                "MAE": np.inf,
                "RMSE": np.inf,
                "MAPE": np.inf,
                "sMAPE": np.inf,
                "Bias": np.nan,
                "R2": np.nan,
                "Directional_Accuracy": np.nan,
                "Status": f"Failed: {str(e)}",
            })

    return pd.DataFrame(rows)


def rolling_validation(
    series,
    season,
    test_size,
    n_windows=N_ROLLING_WINDOWS,
):
    series = (
        pd.Series(series)
        .dropna()
        .astype(float)
    )

    total = len(series)

    if test_size < 1:
        raise ValueError(
            "test_size must be at least 1."
        )

    first_train_end = (
        total
        - test_size
        - (n_windows - 1) * test_size
    )

    possible_train_ends = []

    for i in range(n_windows):
        train_end = (
            first_train_end
            + i * test_size
        )

        if train_end > max(
            season + 10,
            20,
        ):
            possible_train_ends.append(
                train_end
            )

    if not possible_train_ends:
        raise RuntimeError(
            "Rolling validation failed: "
            "not enough observations."
        )

    all_results = []

    for window_number, train_end in enumerate(
        possible_train_ends,
        start=1,
    ):
        result = evaluate_one_window(
            series=series,
            train_end=train_end,
            test_size=test_size,
            season=season,
        )

        result["Window"] = window_number
        result["Train_End"] = train_end
        result["Test_Size"] = test_size

        all_results.append(result)

    return pd.concat(
        all_results,
        ignore_index=True,
    )


def _fit_training_metrics(
    train,
    season,
    model_name,
):
    """
    Calculate in-sample training error.

    These metrics are NOT used to select the final model.
    They are used only to diagnose overfitting/underfitting.
    """
    actual = np.asarray(train, dtype=float)

    try:
        if model_name == "Seasonal Naive":
            if len(train) < season:
                prediction = np.repeat(
                    actual[-1],
                    len(actual),
                )
            else:
                prediction = np.asarray(
                    train.shift(season),
                    dtype=float,
                )
                mask = np.isfinite(prediction)
                actual_used = actual[mask]
                prediction = prediction[mask]

                return {
                    "Train_MAE": _mae(
                        actual_used,
                        prediction,
                    ),
                    "Train_RMSE": _rmse(
                        actual_used,
                        prediction,
                    ),
                    "Train_MAPE": _mape(
                        actual_used,
                        prediction,
                    ),
                    "Train_sMAPE": _smape(
                        actual_used,
                        prediction,
                    ),
                    "Train_Bias": _bias(
                        actual_used,
                        prediction,
                    ),
                    "Train_R2": _r2(
                        actual_used,
                        prediction,
                    ),
                    "Train_Status": "Success",
                }

        elif model_name == "ETS":
            fitted = fit_ets(train, season)
            prediction = np.asarray(
                fitted.fittedvalues,
                dtype=float,
            )

        elif model_name == "SARIMA":
            fitted, _, _ = fit_sarima(
                train,
                season,
            )
            prediction = np.asarray(
                fitted.fittedvalues,
                dtype=float,
            )

        elif model_name == "Random Forest":
            train_df = create_ml_features(
                train,
                season,
            )

            if train_df.empty:
                raise RuntimeError(
                    "Not enough observations."
                )

            model = fit_random_forest(
                train,
                season,
            )

            prediction = np.asarray(
                model.predict(
                    train_df.drop(columns=["y"])
                ),
                dtype=float,
            )
            actual = train_df["y"].to_numpy(
                dtype=float
            )

        else:
            raise ValueError(
                f"Unknown model: {model_name}"
            )

        if len(prediction) != len(actual):
            n = min(
                len(actual),
                len(prediction),
            )
            actual = actual[-n:]
            prediction = prediction[-n:]

        finite = (
            np.isfinite(actual)
            & np.isfinite(prediction)
        )

        actual = actual[finite]
        prediction = prediction[finite]

        return {
            "Train_MAE": _mae(
                actual,
                prediction,
            ),
            "Train_RMSE": _rmse(
                actual,
                prediction,
            ),
            "Train_MAPE": _mape(
                actual,
                prediction,
            ),
            "Train_sMAPE": _smape(
                actual,
                prediction,
            ),
            "Train_Bias": _bias(
                actual,
                prediction,
            ),
            "Train_R2": _r2(
                actual,
                prediction,
            ),
            "Train_Status": "Success",
        }

    except Exception as e:
        return {
            "Train_MAE": np.inf,
            "Train_RMSE": np.inf,
            "Train_MAPE": np.inf,
            "Train_sMAPE": np.inf,
            "Train_Bias": np.nan,
            "Train_R2": np.nan,
            "Train_Status": (
                f"Failed: {str(e)}"
            ),
        }


def _classify_fit(
    train_rmse,
    validation_rmse,
    baseline_rmse,
):
    """
    Practical diagnostic, not a mathematical proof.

    Overfitting:
      validation error is substantially larger
      than training error.

    Underfitting:
      training and validation errors are both poor
      relative to the Seasonal Naive baseline.

    Good fit:
      validation is competitive/better than baseline
      and the train-validation gap is reasonable.

    Inconclusive:
      insufficient/unstable metrics.
    """
    if not (
        np.isfinite(train_rmse)
        and np.isfinite(validation_rmse)
    ):
        return "Inconclusive"

    gap_ratio = _safe_ratio(
        validation_rmse,
        train_rmse,
    )

    if (
        np.isfinite(gap_ratio)
        and gap_ratio >= 1.75
        and validation_rmse > train_rmse
    ):
        return "Overfitting"

    if np.isfinite(baseline_rmse):
        if (
            train_rmse >= baseline_rmse * 1.10
            and validation_rmse >= baseline_rmse * 1.10
        ):
            return "Underfitting"

        if (
            validation_rmse <= baseline_rmse * 0.95
            and (
                not np.isfinite(gap_ratio)
                or gap_ratio < 1.50
            )
        ):
            return "Good Fit"

    if (
        np.isfinite(gap_ratio)
        and gap_ratio >= 1.50
    ):
        return "Possible Overfitting"

    return "Inconclusive"


def build_model_diagnostics(
    series,
    season,
    test_size,
    n_windows=N_ROLLING_WINDOWS,
):
    """
    Full model health report.

    Includes:
      - train metrics
      - rolling validation metrics
      - accuracy metrics
      - train/validation gap
      - overfitting / underfitting assessment
      - model ranking
      - baseline comparison
      - recommendations
    """
    validation = rolling_validation(
        series=series,
        season=season,
        test_size=test_size,
        n_windows=n_windows,
    )

    summary = (
        validation
        .groupby("Model", as_index=False)
        .agg(
            Validation_MAE=("MAE", "mean"),
            Validation_RMSE=("RMSE", "mean"),
            Validation_MAPE=("MAPE", "mean"),
            Validation_sMAPE=("sMAPE", "mean"),
            Validation_Bias=("Bias", "mean"),
            Validation_R2=("R2", "mean"),
            Directional_Accuracy=(
                "Directional_Accuracy",
                "mean",
            ),
            Successful_Windows=(
                "Status",
                lambda x: int(
                    x.astype(str)
                    .str.startswith("Success")
                    .sum()
                ),
            ),
        )
    )

    baseline_row = summary[
        summary["Model"] == "Seasonal Naive"
    ]

    if baseline_row.empty:
        baseline_rmse = np.nan
    else:
        baseline_rmse = float(
            baseline_row.iloc[0][
                "Validation_RMSE"
            ]
        )

    train_size = (
        len(series)
        - test_size * n_windows
    )

    if train_size < 1:
        train_size = max(
            1,
            len(series) - test_size,
        )

    train = series.iloc[:train_size]

    diagnostic_rows = []

    for model_name in MODELS:
        validation_row = summary[
            summary["Model"] == model_name
        ]

        if validation_row.empty:
            continue

        val = validation_row.iloc[0]

        train_metrics = _fit_training_metrics(
            train=train,
            season=season,
            model_name=model_name,
        )

        train_rmse = train_metrics[
            "Train_RMSE"
        ]
        validation_rmse = float(
            val["Validation_RMSE"]
        )

        gap_ratio = _safe_ratio(
            validation_rmse,
            train_rmse,
        )

        gap_percent = (
            (gap_ratio - 1) * 100
            if np.isfinite(gap_ratio)
            else np.nan
        )

        fit_status = _classify_fit(
            train_rmse=train_rmse,
            validation_rmse=validation_rmse,
            baseline_rmse=baseline_rmse,
        )

        baseline_ratio = _safe_ratio(
            validation_rmse,
            baseline_rmse,
        )

        if fit_status == "Overfitting":
            recommendation = (
                "Model generalizes poorly. "
                "Prefer a simpler model or reduce "
                "model complexity."
            )
        elif fit_status == "Possible Overfitting":
            recommendation = (
                "Check with more rolling windows "
                "and consider a simpler model."
            )
        elif fit_status == "Underfitting":
            recommendation = (
                "Model is not capturing enough "
                "pattern. Try ETS/SARIMA or a "
                "richer feature model."
            )
        elif fit_status == "Good Fit":
            recommendation = (
                "Good validation performance with "
                "a reasonable generalization gap."
            )
        else:
            recommendation = (
                "Fit status is inconclusive. "
                "Use the rolling validation metrics "
                "and inspect the window-level results."
            )

        diagnostic_rows.append({
            "Model": model_name,
            **train_metrics,
            "Validation_MAE": float(
                val["Validation_MAE"]
            ),
            "Validation_RMSE": validation_rmse,
            "Validation_MAPE": float(
                val["Validation_MAPE"]
            ),
            "Validation_sMAPE": float(
                val["Validation_sMAPE"]
            ),
            "Validation_Bias": float(
                val["Validation_Bias"]
            ),
            "Validation_R2": float(
                val["Validation_R2"]
            )
            if np.isfinite(
                val["Validation_R2"]
            )
            else np.nan,
            "Directional_Accuracy": float(
                val["Directional_Accuracy"]
            )
            if np.isfinite(
                val["Directional_Accuracy"]
            )
            else np.nan,
            "Generalization_Gap_Ratio": (
                gap_ratio
            ),
            "Generalization_Gap_Percent": (
                gap_percent
            ),
            "Baseline_RMSE": baseline_rmse,
            "Validation_vs_Baseline_Ratio": (
                baseline_ratio
            ),
            "Fit_Status": fit_status,
            "Recommendation": recommendation,
            "Successful_Windows": int(
                val["Successful_Windows"]
            ),
            "Total_Windows": int(n_windows),
        })

    diagnostics = pd.DataFrame(
        diagnostic_rows
    )

    valid = diagnostics[
        np.isfinite(
            diagnostics["Validation_RMSE"]
        )
    ].copy()

    if valid.empty:
        best_model = None
    else:
        best_model = (
            valid.sort_values(
                by=[
                    "Validation_RMSE",
                    "Validation_MAE",
                ],
                ascending=True,
            )
            .iloc[0]["Model"]
        )

    diagnostics["Rank"] = np.nan

    if not valid.empty:
        ranked = valid.sort_values(
            by=[
                "Validation_RMSE",
                "Validation_MAE",
            ],
            ascending=True,
        )["Model"].tolist()

        rank_map = {
            name: i + 1
            for i, name in enumerate(ranked)
        }

        diagnostics["Rank"] = (
            diagnostics["Model"]
            .map(rank_map)
        )

    diagnostics = diagnostics.sort_values(
        by=["Rank", "Model"],
        na_position="last",
    ).reset_index(drop=True)

    return {
        "diagnostics": diagnostics,
        "rolling_validation": validation,
        "best_model": best_model,
        "baseline_model": "Seasonal Naive",
        "baseline_rmse": baseline_rmse,
        "train_size_for_diagnostics": train_size,
    }


def compare_models(validation_results):
    if (
        validation_results is None
        or validation_results.empty
    ):
        raise ValueError(
            "Validation results are empty."
        )

    summary = (
        validation_results
        .groupby("Model", as_index=False)
        .agg(
            MAE=("MAE", "mean"),
            RMSE=("RMSE", "mean"),
            MAPE=("MAPE", "mean"),
            sMAPE=("sMAPE", "mean"),
            Bias=("Bias", "mean"),
            R2=("R2", "mean"),
            Directional_Accuracy=(
                "Directional_Accuracy",
                "mean",
            ),
            Successful_Windows=(
                "Status",
                lambda x: int(
                    x.astype(str)
                    .str.startswith("Success")
                    .sum()
                ),
            ),
        )
    )

    summary = summary.sort_values(
        by=["RMSE", "MAE"],
        ascending=True,
        na_position="last",
    ).reset_index(drop=True)

    return summary


def get_best_model(validation_results):
    summary = compare_models(
        validation_results
    )

    valid = summary[
        np.isfinite(summary["RMSE"])
    ]

    if valid.empty:
        raise RuntimeError(
            "No forecasting model completed successfully."
        )

    return valid.iloc[0]["Model"]


def forecast_model(series, horizon, season):
    series = (
        pd.Series(series)
        .dropna()
        .astype(float)
    )

    if len(series) < 10:
        raise ValueError(
            "At least 10 historical observations "
            "are required."
        )

    test_size = min(
        horizon,
        max(
            1,
            len(series)
            // (N_ROLLING_WINDOWS + 2),
        ),
    )

    try:
        validation_results = rolling_validation(
            series=series,
            season=season,
            test_size=test_size,
            n_windows=N_ROLLING_WINDOWS,
        )

        best_model = get_best_model(
            validation_results
        )

    except Exception:
        best_model = "Seasonal Naive"

    forecast = forecast_one_model(
        train=series,
        horizon=horizon,
        season=season,
        model_name=best_model,
    )

    return np.asarray(
        forecast,
        dtype=float,
    )




def generate_forecast_data(
    frequency: Literal["weekly", "monthly"],
    count: int,
):
    

    # --------------------------------------------------------
    # CHECK FILTERED DATA
    # --------------------------------------------------------

    if repo.latest_filtered_df is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "No filtered data available. "
                "Please call /tables/{table_name} first."
            ),
        )

    try:
        df = repo.latest_filtered_df.copy()

        # --------------------------------------------------------
        # REQUIRED COLUMNS
        # --------------------------------------------------------

        required_columns = [
            "GUID",
            "Date",
            "Amount",
        ]

        missing_columns = [
            col
            for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": (
                        "Filtered data must contain "
                        "GUID, Date and Amount."
                    ),
                    "missing_columns": missing_columns,
                },
            )

        # --------------------------------------------------------
        # CLEAN DATA
        # --------------------------------------------------------

        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce",
        )

        df["Amount"] = pd.to_numeric(
            df["Amount"],
            errors="coerce",
        )

        df = df.dropna(
            subset=[
                "GUID",
                "Date",
                "Amount",
            ]
        )

        if df.empty:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No valid data available "
                    "for forecasting."
                ),
            )

        # --------------------------------------------------------
        # CREATE TIME SERIES
        # --------------------------------------------------------

        if frequency == "weekly":
            series = create_weekly_series(
                df,
                "Date",
                "Amount",
            )
            season = 52

        else:
            series = create_monthly_series(
                df,
                "Date",
                "Amount",
            )
            season = 12

        # --------------------------------------------------------
        # CHECK DATA LENGTH
        # --------------------------------------------------------

        if len(series) < 10:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": (
                        "Not enough observations "
                        "for forecasting."
                    ),
                    "observations": len(series),
                    "frequency": frequency,
                },
            )

        # --------------------------------------------------------
        # FORECAST
        # --------------------------------------------------------

        forecast_result = forecast_model(
            series=series,
            horizon=count,
            season=season,
        )

        # --------------------------------------------------------
        # CREATE FUTURE DATES
        # --------------------------------------------------------

        if frequency == "weekly":

            future_dates = pd.date_range(
                start=(
                    series.index[-1]
                    + pd.Timedelta(days=7)
                ),
                periods=count,
                freq="W-SUN",
            )

        else:

            future_dates = pd.date_range(
                start=(
                    series.index[-1]
                    + pd.offsets.MonthEnd(1)
                ),
                periods=count,
                freq="M",
            )

        # --------------------------------------------------------
        # CREATE RESPONSE
        # --------------------------------------------------------

        forecast_values = np.asarray(
            forecast_result,
            dtype=float,
        )

        result = pd.DataFrame({
            "Date": future_dates,
            "Forecast": forecast_values,
        })

        result["Date"] = (
            result["Date"]
            .dt.strftime("%Y-%m-%d")
        )

        result["Forecast"] = (
            result["Forecast"]
            .round(2)
        )

        return {
            "frequency": frequency,
            "forecast_count": count,
            "historical_observations": len(series),
            "season": season,
            "data": result.to_dict(
                orient="records"
            ),
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )