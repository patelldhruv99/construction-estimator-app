import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained models
cost_model = joblib.load("cost_estimator.pkl")
risk_model = joblib.load("risk_predictor.pkl")

# Custom Page Config
st.set_page_config(
    page_title="AI Construction Estimator - Dhruv Patel",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Styling
st.markdown("""
    <style>
        .title {
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            color: #4A90E2;
        }
        .subtitle {
            text-align: center;
            font-size: 20px;
            color: #555;
        }
        .footer {
            text-align: center;
            font-size: 14px;
            margin-top: 50px;
            color: #777;
        }
        .footer a {
            color: #4A90E2;
            text-decoration: none;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Header Section with Banner
st.markdown('<h1 class="title">🏗️ AI Construction Estimator</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="subtitle">Developed by Dhruv Patel | 📧 <a href="mailto:dhruvrajeshbhai.patel@sjsu.edu">dhruvrajeshbhai.patel@sjsu.edu</a></h3>', unsafe_allow_html=True)
st.markdown('<h3 class="subtitle">🔗 <a href="https://www.linkedin.com/in/dhruvpatel20/" target="_blank">Connect with me on LinkedIn</a></h3>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar Inputs
st.sidebar.title("🔹 Input Project Details")
project_size = st.sidebar.number_input("🏠 Project Size (sq. ft.)", min_value=100, value=2000)
floors = st.sidebar.number_input("🏢 Number of Floors", min_value=1, value=2)
location = st.sidebar.selectbox("📍 Location", ["Urban", "Suburban", "Rural"])
labor_cost = st.sidebar.number_input("👷 Labor Cost ($)", value=15000)
material_cost = st.sidebar.number_input("🧱 Material Cost ($)", value=30000)

# Convert location to numeric
location_map = {"Urban": 0, "Suburban": 1, "Rural": 2}
location_numeric = location_map[location]

# Main Content Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Construction Cost Estimation")
    if st.button("💰 Estimate Cost"):
        input_data = pd.DataFrame([[project_size, floors, location_numeric, labor_cost, material_cost]], 
                                  columns=["project_size", "floors", "location", "labor_cost", "material_cost"])
        cost_prediction = cost_model.predict(input_data)[0]
        st.success(f"🏗️ Estimated Construction Cost: **${cost_prediction:,.2f}**")

        # 📊 Pie Chart Visualization
        st.markdown("#### 🔍 Cost Breakdown")
        cost_data = pd.DataFrame({
            "Category": ["Labor Cost", "Material Cost", "Other Costs"],
            "Cost ($)": [labor_cost, material_cost, cost_prediction - (labor_cost + material_cost)]
        })
        fig, ax = plt.subplots()
        ax.pie(cost_data["Cost ($)"], labels=cost_data["Category"], autopct="%1.1f%%", startangle=90, colors=["#ff9999","#66b3ff","#99ff99"])
        ax.axis("equal")
        st.pyplot(fig)

with col2:
    st.subheader("⚠️ Project Risk Analysis")
    if st.button("🔍 Predict Risk Level"):
        input_risk = pd.DataFrame([[0.6, 1, 0.3, 0.2]], 
                                  columns=["weather_conditions", "labor_availability", "material_delay", "budget_variation"])
        risk_prediction = risk_model.predict(input_risk)[0]
        risk_levels = ["🟢 Low Risk", "🟡 Medium Risk", "🔴 High Risk"]
        st.warning(f"⚠️ Predicted Risk Level: **{risk_levels[int(risk_prediction)]}**")

        # 📊 Bar Chart Visualization
        st.markdown("#### 🔍 Risk Factors")
        risk_data = pd.DataFrame({
            "Factor": ["Weather Conditions", "Labor Availability", "Material Delay", "Budget Variation"],
            "Impact": [0.6, 1, 0.3, 0.2]
        })
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.barplot(x="Impact", y="Factor", data=risk_data, palette="coolwarm", ax=ax)
        st.pyplot(fig)

# Footer Section
st.markdown('<p class="footer">Developed by <a href="https://www.linkedin.com/in/dhruvpatel20/" target="_blank">Dhruv Patel</a> | 📧 <a href="mailto:dhruvrajeshbhai.patel@sjsu.edu">Contact Me</a></p>', unsafe_allow_html=True)
