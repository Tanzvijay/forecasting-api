import os
from typing import Literal, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    Query,
)
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker



load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

DATABASE_URL = (
    f"postgresql+psycopg2://"
    f"{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

latest_filtered_df = None
def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()




def get_tables(db: Session):
    try:
        tables = list_tables(db)

        return {
            "tables": tables
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


def list_tables(db: Session):
    query = text(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name;
        """
    )

    result = db.execute(query)

    return [
        row[0]
        for row in result.fetchall()
    ]


def load_table_dataframe(
    table_name: str,
    db: Session,
) -> pd.DataFrame:
    tables = list_tables(db)

    if table_name not in tables:
        raise ValueError(
            f"Table '{table_name}' does not exist."
        )

    return pd.read_sql_table(
        table_name,
        con=db.bind,
    )

def get_table_columns_data(
    table_name: str,
    db: Session,
):
    try:
        df = load_table_dataframe(
            table_name,
            db,
        )

        return {
            "table_name": table_name,
            "columns": df.columns.tolist(),
        }

    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


def get_unique_values_data(
    table_name: str,
    column_name: str,
    db: Session,
):
    try:
        df = load_table_dataframe(
            table_name,
            db,
        )

        if column_name not in df.columns:
            raise HTTPException(
                status_code=404,
                detail=f"Column '{column_name}' not found.",
            )

        values = (
            df[column_name]
            .dropna()
            .unique()
            .tolist()
        )

        return {
            "table_name": table_name,
            "column_name": column_name,
            "values": values,
        }

    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e),
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


def get_table_data_data(
    table_name: str,
    filter_column: Optional[str],
    filter_value: Optional[str],
    selected_columns: Optional[str],
    db: Session,
):
    global latest_filtered_df

    try:

        # ========================================================
        # LOAD TABLE
        # ========================================================

        df = load_table_dataframe(
            table_name,
            db,
        )

        # ========================================================
        # FILTER ROWS
        # ========================================================

        if filter_column is not None:

            if filter_column not in df.columns:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Column '{filter_column}' "
                        f"not found in table "
                        f"'{table_name}'."
                    ),
                )

            if filter_value is not None:

                filter_values = [
                    value.strip().lower()
                    for value in filter_value.split(",")
                    if value.strip()
                ]

                if not filter_values:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "filter_value must contain "
                            "at least one valid value."
                        ),
                    )

                df = df[
                    df[filter_column]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .isin(filter_values)
                ].copy()

        # ========================================================
        # SELECT COLUMNS
        # ========================================================

        if selected_columns is None:
            raise HTTPException(
                status_code=400,
                detail="Please provide selected_columns.",
            )

        cols = [
            col.strip()
            for col in selected_columns.split(",")
            if col.strip()
        ]

        # ========================================================
        # REQUIRE EXACTLY 3 COLUMNS
        # ========================================================

        if len(cols) != 3:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": (
                        "Exactly 3 columns are required."
                    ),
                    "selected_columns": cols,
                },
            )

        # ========================================================
        # CHECK SELECTED COLUMNS
        # ========================================================

        invalid_columns = [
            col
            for col in cols
            if col not in df.columns
        ]

        if invalid_columns:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": (
                        "Selected column(s) not found."
                    ),
                    "invalid_columns": invalid_columns,
                },
            )

        # ========================================================
        # CREATE SELECTED DATAFRAME
        # ========================================================

        selected_df = df[cols].copy()

        # ========================================================
        # REQUIRED COLUMNS
        # ========================================================

        required_columns = [
            "GUID",
            "Date",
            "Amount",
        ]

        missing_required = [
            col
            for col in required_columns
            if col not in selected_df.columns
        ]

        if missing_required:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": (
                        "The selected 3 columns "
                        "must contain GUID, Date "
                        "and Amount."
                    ),
                    "missing_columns": missing_required,
                    "selected_columns": cols,
                },
            )

        # ========================================================
        # PREPARE DATA
        # ========================================================

        selected_df["Date"] = pd.to_datetime(
            selected_df["Date"],
            errors="coerce",
        )

        selected_df["Amount"] = pd.to_numeric(
            selected_df["Amount"],
            errors="coerce",
        )

        selected_df = selected_df.dropna(
            subset=[
                "GUID",
                "Date",
                "Amount",
            ]
        )

        # ========================================================
        # CHECK EMPTY RESULT
        # ========================================================

        if selected_df.empty:
            raise HTTPException(
                status_code=404,
                detail={
                    "message": (
                        "No rows found for the "
                        "specified filter."
                    ),
                    "filter_column": filter_column,
                    "filter_value": filter_value,
                },
            )

        # ========================================================
        # GUID + DATE + POSITIVE AMOUNT SUMMARY
        # ========================================================

        guid_positive_summary = (
            selected_df
            .assign(
                Amount=selected_df["Amount"].abs()
            )
            .groupby(
                ["GUID", "Date"],
                as_index=False,
            )["Amount"]
            .sum()
        )

        # ========================================================
        # SAVE LATEST FILTERED DATA
        # ========================================================

        latest_filtered_df = (
            guid_positive_summary.copy()
        )

        # ========================================================
        # PREVIEW
        # ========================================================

        preview_df = (
            latest_filtered_df
            .head(10)
            .copy()
        )

        preview_df = preview_df.where(
            pd.notnull(preview_df),
            None,
        )

        # ========================================================
        # RESPONSE
        # ========================================================

        return {
            "table_name": table_name,

            "filter_column": filter_column,

            "filter_value": filter_value,

            "filter_values": (
                [
                    value.strip()
                    for value in filter_value.split(",")
                    if value.strip()
                ]
                if filter_value is not None
                else []
            ),

            "selected_columns": cols,

            "rows": len(
                latest_filtered_df
            ),

            "columns": (
                latest_filtered_df
                .columns
                .tolist()
            ),

            "data": preview_df.to_dict(
                orient="records"
            ),
        }

    except ValueError as e:

        raise HTTPException(
            status_code=404,
            detail=str(e),
        )

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )




def get_latest_filtered_data_data():
    global latest_filtered_df

    if latest_filtered_df is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "No filtered data available. "
                "Please call /tables/{table_name} first."
            ),
        )

    df = latest_filtered_df.copy()

    response_df = df.where(
        pd.notnull(df),
        None,
    )

    return {
        "rows": len(df),
        "columns": df.columns.tolist(),
        "data": response_df.to_dict(
            orient="records"
        ),
    }
