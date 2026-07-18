# Car Insurance Claim Prediction

End to end machine learning project that predicts whether a car insurance
policyholder is likely to file a claim, based on policy and vehicle
attributes. Covers the full pipeline: EDA, feature engineering, feature
selection, model training/comparison and a Flask app for serving
predictions, containerized with Docker.

## Problem statement

Given policy details (tenure, area, population density) and vehicle
specifications (make, segment, engine, safety features), predict
`is_claim` - whether the policyholder will file a claim during the
policy period. This is a binary classification problem with a strongly
imbalanced target: only about 6.4% of policies in the dataset resulted
in a claim.

## Dataset

- 58,592 policy records, 44 columns, no missing values.
- Mix of numeric features (policy tenure, vehicle age, dimensions),
  categorical features (area cluster, segment, fuel type, engine type)
  and 17 Yes/No safety feature flags.
- `max_torque` and `max_power` are combined string columns
  (e.g. `113Nm@4400rpm`) that get split into numeric features during
  feature engineering.

## Project structure

```
insurance_claim_prediction/
│
├── notebooks/
│   ├── 01_data_analysis.ipynb        EDA - distributions, correlations, target balance
│   ├── 02_feature_engineering.ipynb  torque/power split, Yes/No encoding, one hot encoding, scaling
│   ├── 03_feature_selection.ipynb    correlation + Random Forest importance checks
│   ├── 04_model_training.ipynb       trains Logistic Regression, Random Forest, XGBoost
│   └── 05_model_evaluation.ipynb     confusion matrix, ROC curve, precision-recall curve
│
├── src/
│   ├── data_ingestion.py       reads raw csv, does the train/test split
│   ├── data_validation.py      schema + null + value range checks
│   ├── data_transformation.py  feature engineering pipeline (fit/transform)
│   ├── model_trainer.py        trains and compares candidate models
│   ├── prediction_pipeline.py  loads saved artifacts, serves predictions
│   └── utils.py                logger, pickle helpers, json helpers
│
├── templates/index.html        prediction form UI
├── static/css/style.css        dark theme styling
├── app.py                      Flask app (web form + JSON API)
├── Dockerfile
├── requirements.txt
├── .github/workflows/ci.yml    installs deps, runs validation, builds Docker image
└── README.md
```

## Approach

1. **EDA** - checked data types, missing values (none found) and the
   target distribution, which revealed the class imbalance.
2. **Feature engineering** - split `max_torque`/`max_power` into
   numeric value + rpm columns, mapped Yes/No columns to 1/0, one hot
   encoded the remaining categorical columns and scaled numeric
   features with `StandardScaler`.
3. **Feature selection** - compared linear correlation against
   `RandomForestClassifier.feature_importances_` to sanity check that
   engineered features (torque, power, tenure, vehicle dimensions)
   actually carry signal. Kept the full feature set since tree based
   models handle sparse one hot columns without issue.
4. **Model training** - trained Logistic Regression (baseline), Random
   Forest and XGBoost with class imbalance handled via
   `class_weight='balanced'` / `scale_pos_weight` instead of
   oversampling, and compared them on ROC-AUC and F1.
5. **Evaluation** - confusion matrix, classification report, ROC and
   precision-recall curves on a held out 20% test split.
6. **Deployment** - a Flask app that loads the saved model, encoder and
   scaler and serves predictions through both a web form and a JSON
   API, containerized with Docker.

## Results

| Model               | ROC-AUC | F1 score |
|----------------------|---------|----------|
| Logistic Regression   | 0.586   | 0.140    |
| Random Forest          | 0.648   | 0.166    |
| XGBoost                 | 0.646   | 0.160    |

Random Forest was selected as the final model. Given the ~6.4% claim
rate, these numbers are in line with what is typical for real world
insurance claim data - most of the signal is weak and spread across
many features rather than concentrated in one or two strong predictors.

## Running locally

```bash
git clone <your-repo-url>
cd insurance_claim_prediction

python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows

pip install -r requirements.txt

# regenerate artifacts (train/test split, encoder, scaler, model)
python -m src.data_ingestion
python -c "
from src.data_transformation import DataTransformation
from src.model_trainer import ModelTrainer
from src.utils import read_csv_safely

train_df = read_csv_safely('artifacts/train.csv')
test_df = read_csv_safely('artifacts/test.csv')

t = DataTransformation()
X_train, y_train = t.fit_transform(train_df)
X_test = t.transform(test_df.drop(columns=['is_claim']))
y_test = test_df['is_claim']

ModelTrainer().train_and_evaluate(X_train, y_train, X_test, y_test)
"

# start the app
python app.py
```

App runs at `http://localhost:5000`.

### Using the notebooks

Notebooks expect to be run from the `notebooks/` folder (they use
relative paths like `../data/train.csv` and `sys.path.insert(0, '..')`
to import from `src/`). Run them in order, 01 through 05.

## Running with Docker

```bash
docker build -t insurance-claim-prediction .
docker run -p 8501:8501 insurance-claim-prediction
```

App runs at `http://localhost:8501`.

## API usage

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
        "policy_id": "TEST1", "policy_tenure": 0.5, "age_of_car": 0.05,
        "age_of_policyholder": 0.4, "area_cluster": "C1",
        "population_density": 15000, "make": 1, "segment": "A",
        "model": "M1", "fuel_type": "Petrol", "max_torque": "113Nm@4400rpm",
        "max_power": "88.50bhp@6000rpm", "engine_type": "1.2 L K12N Dualjet",
        "airbags": 2, "is_esc": "No", "is_adjustable_steering": "No",
        "is_tpms": "No", "is_parking_sensors": "Yes", "is_parking_camera": "No",
        "rear_brakes_type": "Drum", "displacement": 1197, "cylinder": 4,
        "transmission_type": "Manual", "gear_box": 5, "steering_type": "Power",
        "turning_radius": 4.9, "length": 3990, "width": 1680, "height": 1505,
        "gross_weight": 1155, "is_front_fog_lights": "No",
        "is_rear_window_wiper": "No", "is_rear_window_washer": "No",
        "is_rear_window_defogger": "No", "is_brake_assist": "No",
        "is_power_door_locks": "Yes", "is_central_locking": "Yes",
        "is_power_steering": "Yes", "is_driver_seat_height_adjustable": "No",
        "is_day_night_rear_view_mirror": "No", "is_ecw": "No",
        "is_speed_alert": "Yes", "ncap_rating": 0
      }'
```

Response:

```json
{
  "prediction": [0],
  "claim_probability": [0.4465]
}
```

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, Flask, Docker, GitHub
Actions.

## Possible improvements

- Try SMOTE/ADASYN based resampling in addition to class weighting.
- Hyperparameter tuning with Optuna instead of default parameters.
- Add SHAP based explanations to the prediction response.
- Track experiments with MLflow.
