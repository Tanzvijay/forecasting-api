from typing import Literal, Optional

from fastapi import (
    FastAPI,
    Depends,
    Query,
)

from sqlalchemy.orm import Session

import repo

from auth import require_api_key

from Models import (
    MODELS,
    N_ROLLING_WINDOWS,
    generate_forecast_data,
)

from model_test import (
    build_model_diagnostics_data,
)

from repo import (
    get_db,
    get_tables,
    get_table_columns_data,
    get_unique_values_data,
    get_table_data_data,
    get_latest_filtered_data_data,
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Sales Forecasting API",
    description=(
        "Database table driven forecasting API "
        "using Seasonal Naive, ETS, SARIMA and "
        "Random Forest, with model diagnostics."
    ),
    version="1.0.0",
    dependencies=[Depends(require_api_key)],  # ← all routes protected
)


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# ============================================================
# GET TABLES
# ============================================================

@app.get("/tables")
def db_details(
    db: Session = Depends(get_db),
):
    return get_tables(db)


# ============================================================
# GET TABLE COLUMNS
# ============================================================

@app.get("/tables/{table_name}/columns")
def get_table_columns(
    table_name: str,
    db: Session = Depends(get_db),
):
    return get_table_columns_data(
        table_name,
        db,
    )


# ============================================================
# GET UNIQUE COLUMN VALUES
# ============================================================

@app.get(
    "/tables/{table_name}/columns/{column_name}/values"
)
def get_unique_values(
    table_name: str,
    column_name: str,
    db: Session = Depends(get_db),
):
    return get_unique_values_data(
        table_name,
        column_name,
        db,
    )


# ============================================================
# GET FILTERED TABLE
# ============================================================

@app.get("/tables/{table_name}")
def get_table_data(
    table_name: str,
    filter_column: Optional[str] = None,
    filter_value: Optional[str] = None,
    selected_columns: Optional[str] = None,
    db: Session = Depends(get_db),
):
    return get_table_data_data(
        table_name,
        filter_column,
        filter_value,
        selected_columns,
        db,
    )


# ============================================================
# GET LATEST FILTERED DATA
# ============================================================

@app.get("/latest-filtered-data")
def get_latest_filtered_data():
    return get_latest_filtered_data_data()


# ============================================================
# AVAILABLE FORECASTING MODELS
# ============================================================

@app.get("/models")
def available_models():
    return {
        "models": MODELS,

        "removed_models": [
            "XGBoost",
            "CatBoost",
        ],

        "diagnostics_available": True,
    }


# ============================================================
# MODEL DIAGNOSTICS
# ============================================================

@app.get("/model-diagnostics")
def model_diagnostics(
    frequency: Literal["weekly", "monthly"] = Query(
        ...,
        description=(
            "Diagnostic frequency: weekly or monthly"
        ),
    ),

    test_size: Optional[int] = Query(
        None,
        ge=1,
        description=(
            "Validation periods per rolling window. "
            "If omitted, selected automatically."
        ),
    ),

    n_windows: int = Query(
        N_ROLLING_WINDOWS,
        ge=2,
        le=10,
        description=(
            "Number of time-series rolling validation windows."
        ),
    ),
):
    return build_model_diagnostics_data(
        df=repo.latest_filtered_df,
        frequency=frequency,
        test_size=test_size,
        n_windows=n_windows,
    )


# ============================================================
# FORECAST
# ============================================================

@app.get("/forecast")
def generate_forecast(
    frequency: Literal["weekly", "monthly"] = Query(
        ...,
        description=(
            "Forecast frequency: weekly or monthly"
        ),
    ),

    count: int = Query(
        ...,
        ge=1,
        description=(
            "Number of future periods to forecast"
        ),
    ),
):
    return generate_forecast_data(
        frequency=frequency,
        count=count,
    )