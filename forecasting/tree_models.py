
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from forecasting.weekly_feature import (
    extract_weekly_features,
    make_future_features,
)


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = float(np.mean(np.abs(actual - predicted)))

    rmse = float(
        np.sqrt(np.mean((actual - predicted) ** 2))
    )

    denominator = float(np.sum(np.abs(actual)))

    wape = (
        float(
            np.sum(np.abs(actual - predicted))
            / denominator
            * 100
        )
        if denominator > 0
        else None
    )

    accuracy = (
        max(0.0, 100.0 - wape)
        if wape is not None
        else None
    )

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "WAPE": round(wape, 2) if wape is not None else None,
        "accuracy": (
            round(accuracy, 2)
            if accuracy is not None
            else None
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


def generate_tree_forecasts(weekly_df, count):
    """
    Train, evaluate and forecast using XGBoost, LightGBM
    and CatBoost.

    Parameters
    ----------
    weekly_df : pandas.DataFrame
        Must contain Date and Amount columns.
        Data should already be aggregated to weekly frequency.
    count : int
        Number of future weeks to forecast.

    Returns
    -------
    dict
        Model metrics, best model and forecasts.
    """

    if count < 1:
        raise ValueError("count must be at least 1.")

    required_columns = {"Date", "Amount"}

    missing_columns = required_columns - set(weekly_df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    weekly_df = weekly_df[["Date", "Amount"]].copy()

    # Clean dates and target values.
    weekly_df["Date"] = pd.to_datetime(
        weekly_df["Date"],
        errors="coerce",
    )

    weekly_df["Amount"] = pd.to_numeric(
        weekly_df["Amount"],
        errors="coerce",
    )

    weekly_df = weekly_df.dropna(
        subset=["Date", "Amount"]
    )

    # Match the existing sales forecasting logic.
    weekly_df["Amount"] = weekly_df["Amount"].abs()

    # Data is expected to be aggregated already.
    # Do not aggregate it a second time.
    weekly_df = (
        weekly_df.sort_values("Date")
        .drop_duplicates(subset="Date", keep="last")
        .reset_index(drop=True)
    )

    print("Weekly observations:", len(weekly_df))

    if len(weekly_df) < 60:
        raise ValueError(
            f"At least 60 weekly observations are required; "
            f"received {len(weekly_df)}."
        )

    # Generate lag, rolling and calendar features.
    X, y, feature_data = extract_weekly_features(weekly_df)

    print("Weekly feature rows:", len(feature_data))

    if len(feature_data) < 10:
        raise ValueError(
            "Not enough feature rows for evaluation."
        )

    # Chronological train/test split.
    split = int(len(feature_data) * 0.80)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    if X_train.empty or X_test.empty:
        raise ValueError(
            "Insufficient training or test data."
        )

    # History is used for recursive forecasting.
    history = weekly_df["Amount"].astype(float).tolist()

    last_date = pd.Timestamp(weekly_df["Date"].max())

    # Future dates are Sunday-ending weeks.
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(weeks=1),
        periods=count,
        freq="W-SUN",
    )

    model_results = []
    forecast_results = []

    # Train each model independently so one failure does not
    # prevent the other models from running.
    for name, evaluation_model in create_models().items():

        try:
            print(f"Training {name}...")

            # 1. Evaluate on historical held-out observations.
            evaluation_model.fit(X_train, y_train)

            test_predictions = evaluation_model.predict(X_test)

            metrics = calculate_metrics(
                y_test.to_numpy(),
                test_predictions,
            )

            # 2. Train a fresh model on all labeled observations.
            final_model = create_models()[name]
            final_model.fit(X, y)

            # 3. Forecast future weeks recursively.
            forecast_history = history.copy()
            forecast_records = []

            for future_date in future_dates:

                future_X = make_future_features(
                    forecast_history,
                    future_date,
                )

                prediction = float(
                    final_model.predict(future_X)[0]
                )

                # Sales forecast should not be negative.
                prediction = max(0.0, prediction)

                forecast_records.append({
                    "Date": future_date.strftime("%Y-%m-%d"),
                    "Forecast": round(prediction, 2),
                })

                # Use each prediction to build the next week's features.
                forecast_history.append(prediction)

            model_results.append({
                "model": name,
                **metrics,
            })

            forecast_results.append({
                "model": name,
                "forecast": forecast_records,
            })

            print(
                f"{name} completed successfully. "
                f"WAPE={metrics['WAPE']}"
            )

        except Exception as exc:
            error_message = repr(exc)

            print(f"{name} failed: {error_message}")

            model_results.append({
                "model": name,
                "MAE": None,
                "RMSE": None,
                "WAPE": None,
                "accuracy": None,
                "error": error_message,
            })

    # Select the model with the lowest valid WAPE.
    valid_models = [
        result
        for result in model_results
        if result.get("WAPE") is not None
    ]

    best_model = (
        min(valid_models, key=lambda item: item["WAPE"])
        if valid_models
        else None
    )

    return {
        "frequency": "weekly",
        "forecast_count": count,
        "model_comparison": model_results,
        "best_model": (
            best_model["model"]
            if best_model
            else None
        ),
        "best_model_wape": (
            best_model["WAPE"]
            if best_model
            else None
        ),
        "best_model_accuracy": (
            best_model["accuracy"]
            if best_model
            else None
        ),
        "forecasts": forecast_results,
    }
