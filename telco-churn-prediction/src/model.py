"""Model comparison, threshold selection and evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import NUMERIC_COLS

SEED = 42


def build_pipelines(categorical: list[str]) -> dict[str, Pipeline]:
    """Preprocessing lives inside each pipeline, so CV folds never see each other's data."""
    def prep() -> ColumnTransformer:
        return ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), NUMERIC_COLS),
        ])

    return {
        "Logistic Regression": Pipeline([
            ("prep", prep()),
            ("clf", LogisticRegression(max_iter=2000)),
        ]),
        "Random Forest": Pipeline([
            ("prep", prep()),
            ("clf", RandomForestClassifier(n_estimators=400, min_samples_leaf=10,
                                           random_state=SEED, n_jobs=-1)),
        ]),
    }


def cv_splitter(n_splits: int = 5) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)


def compare_models(pipes: dict[str, Pipeline], X, y, n_splits: int = 5) -> pd.DataFrame:
    rows = []
    for name, pipe in pipes.items():
        scores = cross_validate(pipe, X, y, cv=cv_splitter(n_splits),
                                scoring=["roc_auc", "average_precision"])
        rows.append({
            "model": name,
            "cv_roc_auc_mean": scores["test_roc_auc"].mean(),
            "cv_roc_auc_std": scores["test_roc_auc"].std(),
            "cv_avg_precision_mean": scores["test_average_precision"].mean(),
        })
    return pd.DataFrame(rows).sort_values("cv_roc_auc_mean", ascending=False).reset_index(drop=True)


def select_model(results: pd.DataFrame, simple: str = "Logistic Regression") -> str:
    """Pick the best CV ROC-AUC, but prefer the simpler model when the gap is within one std."""
    top = results.iloc[0]
    simple_row = results.loc[results["model"] == simple].iloc[0]
    if top["cv_roc_auc_mean"] - simple_row["cv_roc_auc_mean"] <= top["cv_roc_auc_std"]:
        return simple
    return str(top["model"])


def choose_threshold(pipe: Pipeline, X, y, n_splits: int = 5) -> float:
    """Threshold maximising F1 on out-of-fold predictions from the training set only."""
    oof = cross_val_predict(pipe, X, y, cv=cv_splitter(n_splits), method="predict_proba")[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, oof)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.argmax(f1))])


def evaluate(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "threshold": round(threshold, 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "average_precision": round(float(average_precision_score(y_true, proba)), 4),
        "precision": round(float(precision_score(y_true, pred)), 4),
        "recall": round(float(recall_score(y_true, pred)), 4),
        "f1": round(float(f1_score(y_true, pred)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def gains_table(y_true, proba, shares=(0.1, 0.2, 0.3, 0.4, 0.5)) -> pd.DataFrame:
    """Share of all churners captured when contacting the top X% highest-risk customers."""
    ranked = pd.DataFrame({"p": np.asarray(proba), "y": np.asarray(y_true)})
    ranked = ranked.sort_values("p", ascending=False).reset_index(drop=True)
    total = ranked["y"].sum()
    rows = []
    for s in shares:
        top = ranked.head(int(round(len(ranked) * s)))
        captured = top["y"].sum() / total
        rows.append({"contacted_share": s,
                     "churners_captured": round(float(captured), 4),
                     "hit_rate": round(float(top["y"].mean()), 4),
                     "lift": round(float(captured / s), 2)})
    return pd.DataFrame(rows)


def logistic_coefficients(pipe: Pipeline) -> pd.Series:
    names = pipe.named_steps["prep"].get_feature_names_out()
    names = [n.split("__", 1)[1] for n in names]
    return pd.Series(pipe.named_steps["clf"].coef_[0], index=names).sort_values()
