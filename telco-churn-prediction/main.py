"""Run the full churn analysis: KPIs, segments, model comparison, evaluation and charts.

Usage:
    python main.py
"""
from __future__ import annotations

import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from src import analysis, plots
from src.data import categorical_cols, clean, features_and_target, load_raw
from src.model import (SEED, build_pipelines, choose_threshold, compare_models, evaluate,
                       gains_table, logistic_coefficients, select_model)

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
REPORTS = ROOT / "reports"


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    df = clean(load_raw())

    # 1. Descriptive analysis
    kpis = analysis.kpis(df)
    df_b = analysis.add_tenure_band(df)
    segments = {col: analysis.segment_table(df_b, col)
                for col in ["Contract", "tenure_band", "InternetService", "PaymentMethod", "TechSupport"]}
    for col, table in segments.items():
        table.to_csv(REPORTS / f"segment_{col}.csv", index=False)
    plots.churn_rate_bars(segments["Contract"], "Contract",
                          "Month-to-month customers churn far more often", FIGURES / "churn_by_contract.png")
    plots.churn_rate_bars(segments["tenure_band"], "tenure_band",
                          "Churn is concentrated in the first months (tenure in months)",
                          FIGURES / "churn_by_tenure.png", sort=False)

    # 2. Model: hold out 25% for a single final test
    X, y = features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED)
    pipes = build_pipelines(categorical_cols(X))
    cv_results = compare_models(pipes, X_train, y_train)
    cv_results.round(4).to_csv(REPORTS / "cv_results.csv", index=False)
    chosen = select_model(cv_results)
    best = pipes[chosen]
    threshold = choose_threshold(best, X_train, y_train)

    # 3. Final evaluation on untouched test data
    best.fit(X_train, y_train)
    proba = best.predict_proba(X_test)[:, 1]
    metrics = {"model": chosen, **evaluate(y_test, proba, threshold)}
    gains = gains_table(y_test, proba)
    gains.to_csv(REPORTS / "gains_table.csv", index=False)
    top20 = gains.loc[gains["contacted_share"] == 0.2].iloc[0]
    plots.gains_curve(y_test, proba, 0.2, top20["churners_captured"], FIGURES / "gains_curve.png")
    if chosen == "Logistic Regression":
        coefs = logistic_coefficients(best)
        coefs.round(4).to_csv(REPORTS / "logistic_coefficients.csv", header=["coefficient"])
        plots.driver_bars(coefs, FIGURES / "churn_drivers.png")

    with open(REPORTS / "kpis.json", "w") as f:
        json.dump(kpis, f, indent=2)
    with open(REPORTS / "model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("KPIs:", json.dumps(kpis, indent=2))
    print("\nCross-validation (training set):\n", cv_results.round(4).to_string(index=False))
    print(f"\nChosen model: {chosen}, threshold {threshold:.3f}")
    print("Test metrics:", json.dumps(metrics, indent=2))
    print("\nGains (test set):\n", gains.to_string(index=False))
    for col, table in segments.items():
        print(f"\n{col}:\n", table.to_string(index=False))


if __name__ == "__main__":
    main()
