
import os
from typing import Literal, Optional

import numpy as np
import pandas as pd

from fastapi import (

    HTTPException

)
from Models import(
    create_weekly_series,
    create_monthly_series,
    build_model_diagnostics,
    MODELS




)

def build_model_diagnostics_data(
    df,
    frequency: Literal["weekly", "monthly"],
    test_size: Optional[int],
    n_windows: int,
):

    if df is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "No filtered data available. "
                "Please call /tables/{table_name} first."
            ),
        )

    try:
        df = df.copy()

        # ========================================================
        # REQUIRED COLUMNS
        # ========================================================

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

        # ========================================================
        # PREPARE DATA
        # ========================================================

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
                    "No valid data available for diagnostics."
                ),
            )

        # ========================================================
        # CREATE TIME SERIES
        # ========================================================

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

        # ========================================================
        # MINIMUM OBSERVATIONS
        # ========================================================

        if len(series) < 10:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": (
                        "Not enough observations for diagnostics."
                    ),
                    "observations": len(series),
                    "minimum_required": 10,
                },
            )

        # ========================================================
        # AUTO TEST SIZE
        # ========================================================

        if test_size is None:
            test_size = min(
                12,
                max(
                    1,
                    len(series) // (n_windows + 2),
                ),
            )

        # ========================================================
        # VALIDATE ROLLING WINDOW REQUIREMENTS
        # ========================================================

        minimum_needed = (
            test_size * n_windows
            + max(season + 10, 20)
        )

        if len(series) < minimum_needed:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": (
                        "Not enough observations for the requested "
                        "rolling validation."
                    ),
                    "observations": len(series),
                    "required_approximately": minimum_needed,
                    "test_size": test_size,
                    "n_windows": n_windows,
                    "season": season,
                    "suggestion": (
                        "Reduce test_size or n_windows."
                    ),
                },
            )

        # ========================================================
        # BUILD MODEL DIAGNOSTICS
        # ========================================================

        report = build_model_diagnostics(
            series=series,
            season=season,
            test_size=test_size,
            n_windows=n_windows,
        )

        # ========================================================
        # PREPARE DIAGNOSTICS DATA
        # ========================================================

        diagnostics_df = (
            report["diagnostics"]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .where(
                lambda x: pd.notnull(x),
                None,
            )
        )

        # ========================================================
        # PREPARE ROLLING VALIDATION DATA
        # ========================================================

        rolling_df = (
            report["rolling_validation"]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .where(
                lambda x: pd.notnull(x),
                None,
            )
        )

        # ========================================================
        # BASELINE
        # ========================================================

        baseline_rmse = report["baseline_rmse"]

        # ========================================================
        # RESPONSE
        # ========================================================

        return {
            "frequency": frequency,

            "historical_observations": len(series),

            "season": season,

            "test_size": test_size,

            "rolling_windows": n_windows,

            "models_checked": MODELS,

            "best_model": report["best_model"],

            "baseline_model": report["baseline_model"],

            "baseline_rmse": (
                None
                if not np.isfinite(baseline_rmse)
                else round(
                    float(baseline_rmse),
                    6,
                )
            ),

            "train_size_for_diagnostics": (
                report["train_size_for_diagnostics"]
            ),

            "diagnostic_explanation": {
                "MAE": (
                    "Lower is better."
                ),

                "RMSE": (
                    "Lower is better; large errors "
                    "are penalized more."
                ),

                "MAPE": (
                    "Lower is better; zero actuals "
                    "are excluded."
                ),

                "sMAPE": (
                    "Lower is better."
                ),

                "Bias": (
                    "Near zero is better. Positive = "
                    "over-forecast; negative = "
                    "under-forecast."
                ),

                "R2": (
                    "Higher is better, but do not use "
                    "it alone for time-series model selection."
                ),

                "Directional_Accuracy": (
                    "Higher is better; percentage of "
                    "correct up/down movements."
                ),

                "Generalization_Gap": (
                    "Validation RMSE divided by training RMSE. "
                    "A large gap is an overfitting warning."
                ),

                "Fit_Status": (
                    "Heuristic classification using the "
                    "train/validation gap and Seasonal Naive baseline."
                ),
            },

            "models": diagnostics_df.to_dict(
                orient="records"
            ),

            "rolling_validation": rolling_df.to_dict(
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