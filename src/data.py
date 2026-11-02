"""Load and clean the IBM Telco Customer Churn dataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "Telco-Customer-Churn.csv"

ID_COL = "customerID"
TARGET_COL = "Churn"
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def load_raw(path: Path | str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy.

    TotalCharges is stored as text, and 11 rows hold a blank string. All of them
    have tenure 0 (new customers not yet billed), so the blank means "nothing
    charged yet" and is set to 0 rather than dropped.
    """
    out = df.copy()
    total = out["TotalCharges"].astype(str).str.strip()
    blank = total == ""
    if (out.loc[blank, "tenure"] != 0).any():
        raise ValueError("Blank TotalCharges found for customers with tenure > 0")
    out["TotalCharges"] = pd.to_numeric(total.where(~blank, "0"))
    out["churn_flag"] = (out[TARGET_COL] == "Yes").astype(int)
    return out


def features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split into model inputs and target. The ID and the raw target are never features."""
    X = df.drop(columns=[ID_COL, TARGET_COL, "churn_flag"])
    y = df["churn_flag"]
    return X, y


def categorical_cols(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c not in NUMERIC_COLS]
