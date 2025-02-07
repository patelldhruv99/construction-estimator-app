import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
df_risk = pd.read_csv("construction_risk_data.csv")

# Define features and target
features = ["weather_conditions", "labor_availability", "material_delay", "budget_variation"]
target = "risk_level"

X = df_risk[features]
y = df_risk[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
risk_model = RandomForestClassifier(n_estimators=100, random_state=42)
risk_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(risk_model, "risk_predictor.pkl")

print("✅ Risk Analysis Model Trained and Saved Successfully!")
python3 train_cost_model.py
