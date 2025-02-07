import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the dataset
df = pd.read_csv("construction_data.csv")

# Define features and target
features = ["project_size", "floors", "location", "labor_cost", "material_cost"]
target = "total_cost"

X = df[features]
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
cost_model = RandomForestRegressor(n_estimators=100, random_state=42)
cost_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(cost_model, "cost_estimator.pkl")

print("✅ Cost Estimation Model Trained and Saved Successfully!")
