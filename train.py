import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor


# Load Dataset
df = pd.read_csv("dataset/medical_insurance.csv")

# Encode Categorical Columns
le = LabelEncoder()
df["sex"] = le.fit_transform(df["sex"])
df["smoker"] = le.fit_transform(df["smoker"])
df["region"] = le.fit_transform(df["region"])

# Features & Target
X = df.drop("charges", axis=1)
y = df["charges"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models List
models = {
    "LinearRegression": LinearRegression(),
    "RidgeRegression": Ridge(),
    "RandomForest": RandomForestRegressor(),
    "GradientBoosting": GradientBoostingRegressor(),
    "XGBoost": XGBRegressor()
}

best_score = 0
best_model = None
best_name = ""

# MLflow Tracking
mlflow.set_experiment("Medical Insurance Cost Prediction")

for name, model in models.items():
    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)

        mlflow.sklearn.log_model(model, "model")

        print(f"{name} → RMSE:{rmse:.2f}, R2:{r2:.2f}")

        if r2 > best_score:
            best_score = r2
            best_model = model
            best_name = name

# Save Best Model
joblib.dump(best_model, "model/best_model.pkl")

print("\nBest Model:", best_name)
print("Best R2 Score:", best_score)
