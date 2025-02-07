import joblib
import pandas as pd

# Load the trained models
cost_model = joblib.load("cost_estimator.pkl")
risk_model = joblib.load("risk_predictor.pkl")

# Test Cost Estimation Model
print("🔹 Testing Cost Estimation Model...")
test_project = pd.DataFrame([[2000, 2, 1, 15000, 30000]], 
                            columns=["project_size", "floors", "location", "labor_cost", "material_cost"])
cost_prediction = cost_model.predict(test_project)[0]
print(f"✅ Predicted Construction Cost: ${cost_prediction:,.2f}")

# Test Risk Analysis Model
print("\n🔹 Testing Risk Analysis Model...")
test_risk_factors = pd.DataFrame([[0.6, 1, 0.3, 0.2]], 
                                  columns=["weather_conditions", "labor_availability", "material_delay", "budget_variation"])
risk_prediction = risk_model.predict(test_risk_factors)[0]
risk_levels = ["Low", "Medium", "High"]
print(f"⚠️ Predicted Risk Level: {risk_levels[int(risk_prediction)]}")
