"""Descriptive KPIs and segment tables."""
from __future__ import annotations

import pandas as pd

TENURE_BINS = [-1, 6, 12, 24, 48, 72]
TENURE_LABELS = ["0-6", "7-12", "13-24", "25-48", "49-72"]


def kpis(df: pd.DataFrame) -> dict:
    mrr = df["MonthlyCharges"].sum()
    churned = df["churn_flag"] == 1
    return {
        "customers": int(len(df)),
        "churn_rate": round(float(df["churn_flag"].mean()), 4),
        "monthly_revenue": round(float(mrr), 2),
        "avg_monthly_charge": round(float(df["MonthlyCharges"].mean()), 2),
        "monthly_revenue_from_churned": round(float(df.loc[churned, "MonthlyCharges"].sum()), 2),
        "share_of_revenue_from_churned": round(float(df.loc[churned, "MonthlyCharges"].sum() / mrr), 4),
    }


def add_tenure_band(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["tenure_band"] = pd.cut(out["tenure"], TENURE_BINS, labels=TENURE_LABELS)
    return out


def segment_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Churn rate, customer count and monthly revenue per segment."""
    table = (
        df.groupby(col, observed=True)
        .agg(customers=("churn_flag", "size"),
             churn_rate=("churn_flag", "mean"),
             monthly_revenue=("MonthlyCharges", "sum"))
        .reset_index()
    )
    table["churn_rate"] = table["churn_rate"].round(4)
    return table
