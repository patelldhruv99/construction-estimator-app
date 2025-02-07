import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, confusion_matrix

# Load trained models
cost_model = joblib.load("cost_estimator.pkl")
risk_model = joblib.load("risk_predictor.pkl")

# Load the dataset used for training (for evaluation)
cost_data = pd.read_csv("construction_data.csv")
risk_data = pd.read_csv("construction_risk_data.csv")

# Define Features and Target for Cost Model
X_cost = cost_data[["project_size", "floors", "location", "labor_cost", "material_cost"]]
y_cost = cost_data["total_cost"]

# Predict using trained cost model
y_cost_pred = cost_model.predict(X_cost)

# Evaluate Cost Estimation Model
mae = mean_absolute_error(y_cost, y_cost_pred)
mse = mean_squared_error(y_cost, y_cost_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_cost, y_cost_pred)

print("📊 Cost Estimation Model Performance:")
print(f"✅ Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"✅ Mean Squared Error (MSE): ${mse:,.2f}")
print(f"✅ Root Mean Squared Error (RMSE): ${rmse:,.2f}")
print(f"✅ R² Score: {r2:.4f}")

# ---------------------- RISK MODEL EVALUATION ----------------------

# Define Features and Target for Risk Model
X_risk = risk_data[["weather_conditions", "labor_availability", "material_delay", "budget_variation"]]
y_risk = risk_data["risk_level"]

# Predict using trained risk model
y_risk_pred = risk_model.predict(X_risk)

# Evaluate Risk Prediction Model
accuracy = accuracy_score(y_risk, y_risk_pred)
conf_matrix = confusion_matrix(y_risk, y_risk_pred)

print("\n📊 Risk Prediction Model Performance:")
print(f"✅ Accuracy: {accuracy:.4f}")
print("✅ Confusion Matrix:")
print(conf_matrix)

