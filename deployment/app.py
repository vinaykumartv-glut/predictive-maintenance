import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(
    repo_id="vinaykumartv/predictive-maintenance-model",
    filename="best_predictive_maintenance_model.joblib"
)
model = joblib.load(model_path)

# Page setup
st.set_page_config(page_title="Engine Health Predictor", layout="wide")
st.title("⚙️ Predictive Maintenance: Engine Health Predictor")
st.markdown("""
This tool predicts whether an engine requires maintenance based on real-time sensor readings.
Input the engine's current sensor values to get a prediction on its condition.
""")

# --- Engine Sensor Inputs ---
with st.expander("📊 Engine Sensor Readings", expanded=True):
    col1, col2, col3 = st.columns(3)

    with col1:
        Engine_RPM = st.number_input("Engine RPM (Revolutions per Minute)", min_value=0, value=700, step=10)
        Lub_Oil_Pressure = st.number_input("Lub Oil Pressure (bar/kPa)", min_value=0.0, value=3.0, step=0.1, format="%.2f")

    with col2:
        Fuel_Pressure = st.number_input("Fuel Pressure (bar/kPa)", min_value=0.0, value=10.0, step=0.1, format="%.2f")
        Coolant_Pressure = st.number_input("Coolant Pressure (bar/kPa)", min_value=0.0, value=2.5, step=0.1, format="%.2f")

    with col3:
        Lub_Oil_Temperature = st.number_input("Lub Oil Temperature (°C)", min_value=0.0, value=75.0, step=0.1, format="%.2f")
        Coolant_Temperature = st.number_input("Coolant Temperature (°C)", min_value=0.0, value=80.0, step=0.1, format="%.2f")

# Prediction button
classification_threshold = 0.45
if st.button("🔮 Predict Engine Condition"):
    input_data = pd.DataFrame([{
        'Engine rpm': Engine_RPM,
        'Lub oil pressure': Lub_Oil_Pressure,
        'Fuel pressure': Fuel_Pressure,
        'Coolant pressure': Coolant_Pressure,
        'lub oil temp': Lub_Oil_Temperature,
        'Coolant temp': Coolant_Temperature
    }])

    try:
        prediction_proba = model.predict_proba(input_data)[0, 1]
        prediction = (prediction_proba >= classification_threshold).astype(int)
        result = "🔴 Faulty: Maintenance Required" if prediction == 1 else "🟢 Normal: Operating Normally"

        st.subheader("📊 Prediction Result")
        st.metric(label="Faulty Probability", value=f"{prediction_proba:.2f}")
        if prediction == 1:
          st.error(result)
        else:
          st.success(result)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
