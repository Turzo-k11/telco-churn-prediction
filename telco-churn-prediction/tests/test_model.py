import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.data import categorical_cols, clean, features_and_target, load_raw
from src.model import build_pipelines, gains_table, select_model


def test_logistic_pipeline_beats_baseline():
    X, y = features_and_target(clean(load_raw()))
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    pipe = build_pipelines(categorical_cols(X))["Logistic Regression"]
    pipe.fit(X_tr, y_tr)
    assert roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1]) > 0.80


def test_gains_table_perfect_ranking():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    proba = np.linspace(1, 0, 10)
    g = gains_table(y, proba, shares=(0.2, 0.5))
    assert g["churners_captured"].tolist() == [1.0, 1.0]
    assert g["churners_captured"].is_monotonic_increasing


def test_select_model_prefers_simple_when_close():
    res = pd.DataFrame({"model": ["Random Forest", "Logistic Regression"],
                        "cv_roc_auc_mean": [0.846, 0.844], "cv_roc_auc_std": [0.014, 0.015]})
    assert select_model(res) == "Logistic Regression"
    res.loc[0, "cv_roc_auc_mean"] = 0.90
    assert select_model(res) == "Random Forest"
