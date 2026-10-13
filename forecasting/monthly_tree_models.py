
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from forecasting.monthly_feature import (
    extract_monthly_features,
    make_monthly_future_features,
)


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[valid]
    predicted = predicted[valid]

    if len(actual) == 0:
        return {
            "MAE": None,
            "RMSE": None,
            "WAPE": None,
            "accuracy": None,
        }

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    denominator = np.sum(np.abs(actual))

    if denominator == 0:
        wape = None
        accuracy = None
    else:
        wape = (
            np.sum(np.abs(actual - predicted))
            / denominator
        ) * 100
        accuracy = max(0.0, 100.0 - wape)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "WAPE": None if wape is None else float(wape),
        "accuracy": (
            None if accuracy is None else float(accuracy)
        ),
    }


def create_models():
    return {
        "XGBoost": XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="regression",
            random_state=42,
            verbosity=-1,
            n_jobs=-1,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            loss_function="RMSE",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        ),
    }


def generate_monthly_tree_forecasts(monthly_df, count):
    if count < 1:
        raise ValueError("count must be at least 1.")

    data = monthly_df.copy()

    if not {"Date", "Amount"}.issubset(data.columns):
        raise ValueError(
            "monthly_df must contain Date and Amount columns."
        )

    data["Date"] = pd.to_datetime(
        data["Date"], errors="coerce"
    )
    data["Amount"] = pd.to_numeric(
        data["Amount"], errors="coerce"
    )

    data = data.dropna(subset=["Date", "Amount"])
    data = data.sort_values("Date")

    if data.empty:
        raise ValueError("No valid monthly data available.")

    data["Amount"] = data["Amount"].abs()

    # Aggregate to calendar month-end and retain missing months
    # as zero-sales months.
    # Data is already aggregated monthly by Models.py.
# Do not resample it again.
    data = (
    data.sort_values("Date")
    .drop_duplicates(subset="Date", keep="last")
    .reset_index(drop=True)
)

    if len(data) < 24:
        raise ValueError(
            "At least 24 monthly observations are required. "
            "More history is recommended for reliable evaluation."
        )

    X, y, feature_data = extract_monthly_features(data)

    if len(X) < 10:
        raise ValueError(
            "Not enough feature rows after creating monthly lags."
        )

    # Chronological 80/20 holdout.
    split_index = int(len(X) * 0.8)

    if split_index < 5 or len(X) - split_index < 2:
        raise ValueError(
            "Not enough feature rows for train/test evaluation."
        )

    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    results = []
    forecasts = []

    for model_name in create_models():
        try:
            # Evaluate on the later holdout period.
            evaluation_model = create_models()[model_name]
            evaluation_model.fit(X_train, y_train)

            test_predictions = evaluation_model.predict(X_test)

            metrics = calculate_metrics(
                y_test.to_numpy(),
                test_predictions,
            )

            # Refit on all feature rows before future forecasting.
            model = create_models()[model_name]
            model.fit(X, y)

            history = data["Amount"].astype(float).tolist()
            last_date = pd.Timestamp(data["Date"].iloc[-1])

            future_dates = pd.date_range(
                start=last_date + pd.offsets.MonthEnd(1),
                periods=count,
                freq="ME",
            )

            future_records = []

            for step, future_date in enumerate(
                future_dates, start=1
            ):
                future_X = make_monthly_future_features(
                    history=history,
                    future_date=future_date,
                    time_index=len(data) + step - 1,
                )

                prediction = float(
                    np.asarray(model.predict(future_X)).ravel()[0]
                )

                prediction = max(0.0, prediction)

                future_records.append({
                    "Date": future_date.strftime("%Y-%m-%d"),
                    "Forecast": round(prediction, 2),
                })

                # Append predictions so later months can use
                # earlier predicted amounts as lag features.
                history.append(prediction)

            results.append({
                "model": model_name,
                **metrics,
            })

            forecasts.append({
                "model": model_name,
                "forecast": future_records,
            })

        except Exception as exc:
            results.append({
                "model": model_name,
                "MAE": None,
                "RMSE": None,
                "WAPE": None,
                "accuracy": None,
                "error": str(exc),
            })

    valid_results = [
        item for item in results
        if item.get("WAPE") is not None
        and np.isfinite(item["WAPE"])
    ]

    best_result = (
        min(valid_results, key=lambda item: item["WAPE"])
        if valid_results else None
    )

    return {
        "frequency": "monthly",
        "forecast_count": count,
        "model_comparison": results,
        "best_model": (
            best_result["model"] if best_result else None
        ),
        "best_model_wape": (
            best_result["WAPE"] if best_result else None
        ),
        "best_model_accuracy": (
            best_result["accuracy"] if best_result else None
        ),
        "forecasts": forecasts,
    }
