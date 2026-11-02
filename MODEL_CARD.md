# Model card: Telco customer churn

**Purpose:** rank existing customers by their risk of churning, so a retention team can prioritise outreach. Built as a portfolio project.

**Data:** IBM Telco Customer Churn sample dataset. It has 7,043 customers of a fictional telecom company, one row per customer, and no dates.

**Target:** `Churn` (Yes/No), recoded to 1/0. 26.5% of customers churned.

**Features:** 19 inputs covering demographics (gender, senior citizen, partner, dependents), services (phone, internet, add-ons, streaming), account details (contract, billing, payment method) and charges (tenure, monthly and total charges). `customerID` is excluded.

**Model:** Logistic Regression with one-hot encoding and standard scaling in a scikit-learn `Pipeline`. It was chosen over a Random Forest because their 5-fold CV ROC-AUC was within one standard deviation (0.844 vs 0.846), and the simpler model is easier to explain.

**Evaluation (held-out 25% test set, 1,761 customers):** ROC-AUC 0.847 and average precision 0.638. At a threshold of 0.33 (maximum F1 on out-of-fold training predictions), precision is 54.5% and recall is 72.6%. The top 20% by predicted risk contains 50% of churners.

**Intended use:** prioritising retention outreach, together with human judgement.

**Not intended for:** decisions that deny service, change prices or treat customers differently on the basis of demographics.

**Risks and limitations:**
- No time-based validation, because the data is a snapshot.
- Demographic features (gender, senior citizen) are model inputs, so subgroup performance should be checked before any real use.
- The threshold is not cost-based.
- The coefficients are associations, not causal effects.
