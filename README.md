# Car Insurance Claim Prediction

This project predicts whether a car insurance customer will file a claim,
based on their policy and vehicle details.

## Problem

Insurance companies want to know in advance which customers are likely to
file a claim. Using details like policy tenure, vehicle age, area, and car
specs, I built a model to predict `is_claim` (yes/no). The tricky part was
that the data is very imbalanced — only about 6.4% of policies actually had
a claim.

## Dataset

- 58,592 records, 44 columns, no missing values.
- Mix of numeric, categorical, and Yes/No safety feature columns.
- `max_torque` and `max_power` were combined text values, so I split them
  into separate numbers during feature engineering.

## What I did

1. Explored the data and found the class imbalance.
2. Cleaned and engineered features (split torque/power, encoded Yes/No
   columns, one-hot encoded categories, scaled numeric values).
3. Checked feature importance to confirm the engineered features were
   actually useful.
4. Trained and compared Logistic Regression, Random Forest, and XGBoost,
   using class weighting to handle the imbalance.
5. Evaluated using confusion matrix, ROC curve, and precision-recall curve.
6. Built a Flask app to serve predictions (web form + JSON API) and
   containerized it with Docker.

## Results

| Model               | ROC-AUC | F1 score |
|---------------------|---------|----------|
| Logistic Regression | 0.586   | 0.140    |
| Random Forest       | 0.648   | 0.166    |
| XGBoost              | 0.646   | 0.160    |

Went with Random Forest as the final model.

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, Flask, Docker, GitHub Actions

## Running it

```bash
pip install -r requirements.txt
python -m src.data_ingestion
python app.py
```

App runs at `http://localhost:5000`.

## What I'd add next

- SMOTE/ADASYN resampling
- Hyperparameter tuning with Optuna
- SHAP-based explanations
