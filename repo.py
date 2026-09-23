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
from fastapi import UploadFile, File
from pathlib import Path
import re
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

def upload_file_to_database(
    file: UploadFile,
    table_name: str,
):
    try:
        # -----------------------------------------
        # Validate file extension
        # -----------------------------------------
        file_extension = Path(file.filename).suffix.lower()

        if file_extension not in [".csv", ".xlsx", ".xls"]:
            raise HTTPException(
                status_code=400,
                detail="Only CSV, XLSX and XLS files are supported."
            )

        # -----------------------------------------
        # Validate table name
        # -----------------------------------------
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", table_name):
            raise HTTPException(
                status_code=400,
                detail="Invalid table name."
            )

        # -----------------------------------------
        # Read uploaded file
        # -----------------------------------------
        if file_extension == ".csv":

            df = pd.read_csv(file.file)

        else:

            df = pd.read_excel(file.file)

        # -----------------------------------------
        # Check empty file
        # -----------------------------------------
        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )

        # -----------------------------------------
        # Clean column names
        # -----------------------------------------
        df.columns = [
            str(col).strip()
            .replace(" ", "_")
            .replace("-", "_")
            for col in df.columns
        ]

        # -----------------------------------------
        # Save dataframe to PostgreSQL
        # -----------------------------------------
        df.to_sql(
            table_name,
            con=engine,
            if_exists="replace",
            index=False,
            method="multi"
        )

        return {
            "message": "File uploaded successfully.",
            "file_name": file.filename,
            "table_name": table_name,
            "rows": len(df),
            "columns": df.columns.tolist()
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

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
    use_date_column: bool,
    selected_columns: Optional[str],   # None when use_date_column=True
    date_column: Optional[str],        # None when use_date_column=False
    target_column: Optional[str],      # None when use_date_column=False
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
        # BRANCH — use_date_column=False → HARDCODED tally columns
        #          use_date_column=True  → user input columns
        # ========================================================

        if not use_date_column:

            # ====================================================
            # HARDCODED TALLY COLUMNS
            # ====================================================

            TALLY_ID_COL     = "GUID"
            TALLY_DATE_COL   = "Date"
            TALLY_AMOUNT_COL = "Amount"

            cols = [
                TALLY_ID_COL,
                TALLY_DATE_COL,
                TALLY_AMOUNT_COL,
            ]

            # ====================================================
            # CHECK HARDCODED COLUMNS EXIST IN TABLE
            # ====================================================

            missing_columns = [
                col
                for col in cols
                if col not in df.columns
            ]

            if missing_columns:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": (
                            "Table does not have the "
                            "required tally columns: "
                            "GUID, Date, Amount."
                        ),
                        "missing_columns": missing_columns,
                        "available_columns": (
                            df.columns.tolist()
                        ),
                    },
                )

            # ====================================================
            # CREATE SELECTED DATAFRAME
            # ====================================================

            selected_df = df[cols].copy()

            # ====================================================
            # PREPARE DATA
            # ====================================================

            selected_df[TALLY_DATE_COL] = pd.to_datetime(
                selected_df[TALLY_DATE_COL],
                errors="coerce",
            )

            selected_df[TALLY_AMOUNT_COL] = pd.to_numeric(
                selected_df[TALLY_AMOUNT_COL],
                errors="coerce",
            )

            selected_df = selected_df.dropna(
                subset=cols
            )

            # ====================================================
            # CHECK EMPTY RESULT
            # ====================================================

            if selected_df.empty:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "message": (
                            "No rows found for the "
                            "specified filter."
                        ),
                        "filter_column": filter_column,
                        "filter_value":  filter_value,
                    },
                )

            # ====================================================
            # GUID + Date + Amount SUMMARY
            # ====================================================

            guid_positive_summary = (
                selected_df
                .assign(
                    Amount=selected_df[TALLY_AMOUNT_COL].abs()
                )
                .groupby(
                    [TALLY_ID_COL, TALLY_DATE_COL],
                    as_index=False,
                )[TALLY_AMOUNT_COL]
                .sum()
            )

            latest_filtered_df = guid_positive_summary.copy()

            response_extra = {
                "selected_columns": cols,   # ["GUID", "Date", "Amount"]
            }

        else:

            # ====================================================
            # CHECK date_column + target_column EXIST IN TABLE
            # ====================================================

            if date_column not in df.columns:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": (
                            f"date_column '{date_column}' "
                            f"not found in table '{table_name}'."
                        ),
                        "available_columns": (
                            df.columns.tolist()
                        ),
                    },
                )

            if target_column not in df.columns:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": (
                            f"target_column '{target_column}' "
                            f"not found in table '{table_name}'."
                        ),
                        "available_columns": (
                            df.columns.tolist()
                        ),
                    },
                )

            # ====================================================
            # BUILD DATAFRAME FROM date_column + target_column
            # ====================================================

            selected_df = df[
                [date_column, target_column]
            ].copy()

            selected_df[date_column] = pd.to_datetime(
                selected_df[date_column],
                errors="coerce",
            )

            selected_df[target_column] = pd.to_numeric(
                selected_df[target_column],
                errors="coerce",
            )

            selected_df = selected_df.dropna(
                subset=[date_column, target_column]
            )

            # ====================================================
            # CHECK EMPTY RESULT
            # ====================================================

            if selected_df.empty:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "message": (
                            "No rows found for the "
                            "specified filter."
                        ),
                        "filter_column": filter_column,
                        "filter_value":  filter_value,
                    },
                )

            # ====================================================
            # date_column + target_column SUMMARY
            # ====================================================

            guid_positive_summary = (
                selected_df
                .assign(**{
                    target_column: (
                        selected_df[target_column].abs()
                    )
                })
                .groupby(
                    [date_column],
                    as_index=False,
                )[target_column]
                .sum()
            )

            latest_filtered_df = guid_positive_summary.copy()

            response_extra = {
                "date_column":   date_column,
                "target_column": target_column,
            }

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
            "table_name":      table_name,
            "filter_column":   filter_column,
            "filter_value":    filter_value,
            "filter_values": (
                [
                    value.strip()
                    for value in filter_value.split(",")
                    if value.strip()
                ]
                if filter_value is not None
                else []
            ),
            "use_date_column": use_date_column,
            **response_extra,
            "rows":    len(latest_filtered_df),
            "columns": latest_filtered_df.columns.tolist(),
            "data":    preview_df.to_dict(orient="records"),
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
