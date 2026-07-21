import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(
    repo_id="vinaykumartv/predictive-maintenance",
    filename="best_tourism_project_model.joblib"
)
model = joblib.load(model_path)

# Page setup
st.set_page_config(page_title="Visit With Us", layout="wide")
st.title("🌿 Visit With Us – Wellness Tourism Predictor")
st.markdown("""
This tool helps staff predict whether a customer is likely to purchase
the newly introduced **Wellness Tourism Package**.
Fill in the customer details below to generate a prediction.
""")

# --- Combined Customer Profile ---
with st.expander("👤 Customer Profile", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        TypeofContact = st.selectbox("📞 Contact Type", ["Company Invited", "Self Inquiry"])
        CityTier = st.selectbox("🏙️ City Tier", ["Tier 1", "Tier 2", "Tier 3"])
        Age = st.number_input("🎂 Age", min_value=18, max_value=100, value=30)
        Gender = st.selectbox("⚧ Gender", ["Male", "Female"])
        MaritalStatus = st.selectbox("💍 Marital Status", ["Single", "Married", "Divorced"])
        Occupation = st.selectbox("💼 Occupation", ["Salaried", "Freelancer"])
        Designation = st.selectbox("🏢 Designation", ["Executive", "Managerial", "Professional", "Other"])

    with col2:
        MonthlyIncome = st.number_input("💰 Monthly Income", min_value=0.0, value=5000.0, step=500.0)
        NumberOfPersonVisiting = st.number_input("👥 People Visiting", min_value=1, value=1)
        NumberOfChildrenVisiting = st.number_input("👶 Children (<5 yrs)", min_value=0, value=0)
        PreferredPropertyStar = st.number_input("⭐ Property Star", min_value=1, max_value=5, value=3)
        NumberOfTrips = st.number_input("🛫 Annual Trips", min_value=1, value=1)
        Passport = st.selectbox("🛂 Passport", ["Yes", "No"])
        OwnCar = st.selectbox("🚗 Own Car", ["Yes", "No"])

# --- Sales Interaction ---
with st.expander("📊 Sales Interaction", expanded=True):
    PitchSatisfactionScore = st.number_input("⭐ Pitch Satisfaction", min_value=1, max_value=5, value=3)
    ProductPitched = st.selectbox("📦 Product Pitched", ["Basic", "Standard", "Premium"])
    NumberOfFollowups = st.number_input("📞 Followups", min_value=0, value=0)
    DurationOfPitch = st.number_input("⏱️ Pitch Duration (min)", min_value=1, value=5)

# Prediction button
classification_threshold = 0.45
if st.button("🔮 Predict Customer Interest"):
    input_data = pd.DataFrame([{
        'Age': Age,
        'TypeofContact': 1 if TypeofContact == "Company Invited" else 0,
        'CityTier': 3 if CityTier == "Tier 1" else 2 if CityTier == "Tier 2" else 1,
        'Occupation': 1 if Occupation == "Salaried" else 0,
        'Gender': Gender,
        'NumberOfPersonVisiting': NumberOfPersonVisiting,
        'PreferredPropertyStar': PreferredPropertyStar,
        'MaritalStatus': MaritalStatus,
        'NumberOfTrips': NumberOfTrips,
        'Passport': 1 if Passport == "Yes" else 0,
        'OwnCar': 1 if OwnCar == "Yes" else 0,
        'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
        'Designation': Designation,
        'MonthlyIncome': MonthlyIncome,
        'PitchSatisfactionScore': PitchSatisfactionScore,
        'ProductPitched': ProductPitched,
        'NumberOfFollowups': NumberOfFollowups,
        'DurationOfPitch': DurationOfPitch
    }])

    try:
        prediction_proba = model.predict_proba(input_data)[0, 1]
        prediction = (prediction_proba >= classification_threshold).astype(int)
        result = "✅ Likely to purchase the package" if prediction == 1 else "❌ Not likely to purchase the package"

        st.subheader("📊 Prediction Result")
        st.metric(label="Purchase Probability", value=f"{prediction_proba:.2f}")
        if prediction == 1:
          st.success(result)
        else:
          st.error(result)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
