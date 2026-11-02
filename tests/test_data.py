import pandas as pd

from src.data import ID_COL, clean, features_and_target, load_raw


def test_raw_shape_and_target():
    df = load_raw()
    assert df.shape == (7043, 21)
    assert set(df["Churn"]) == {"Yes", "No"}


def test_blank_total_charges_become_zero_for_new_customers():
    raw = pd.DataFrame({
        "customerID": ["a", "b"], "tenure": [0, 5], "MonthlyCharges": [20.0, 30.0],
        "TotalCharges": [" ", "150.5"], "Churn": ["No", "Yes"],
    })
    out = clean(raw)
    assert out["TotalCharges"].tolist() == [0.0, 150.5]
    assert out["churn_flag"].tolist() == [0, 1]


def test_cleaned_dataset_has_no_missing_values():
    df = clean(load_raw())
    assert df.isna().sum().sum() == 0
    assert df["TotalCharges"].dtype.kind == "f"


def test_id_and_target_are_not_features():
    X, y = features_and_target(clean(load_raw()))
    assert ID_COL not in X.columns
    assert "Churn" not in X.columns and "churn_flag" not in X.columns
    assert len(X) == len(y)
