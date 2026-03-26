import os
import pickle
import pandas as pd
import streamlit as st

# -------------------------------
# Path Configuration
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "car_price_prediction_model.pkl")
features_path = os.path.join(BASE_DIR, "model_features.pkl")

# -------------------------------
# Load Model
# -------------------------------
if not os.path.exists(model_path):
    st.error(f"❌ Model file not found at: `{model_path}`\n\nPlease place `car_price_prediction_model.pkl` in: `{BASE_DIR}`")
    st.stop()

if not os.path.exists(features_path):
    st.error(f"❌ Features file not found at: `{features_path}`\n\nPlease place `model_features.pkl` in: `{BASE_DIR}`")
    st.stop()

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(features_path, "rb") as f:
        model_features = pickle.load(f)

except Exception as e:
    st.error(f"❌ Failed to load model files. Error: {e}")
    st.stop()

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗")

st.title("🚗 Used Car Price Prediction App")
st.write("Enter the car details below to estimate its price.")

# -------------------------------
# Inputs
# -------------------------------
year = st.number_input("Year of Manufacture", min_value=1990, max_value=2025, value=2015)
mileage = st.number_input("Mileage (in kilometers)", min_value=0, step=1000)
fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'Electric'])
accident = st.selectbox("Accident History", ['Yes', 'No'])
clean_title = st.selectbox("Clean Title", ['Yes', 'No'])

# -------------------------------
# Prepare Input Data
# -------------------------------
input_data = {
    'year': year,
    'kilometers_driven': mileage,
}
df_input = pd.DataFrame([input_data])

# One-hot encoding manually
categorical_inputs = {
    'fuel_type': fuel_type,
    'accident': accident,
    'clean_title': clean_title
}

for col in model_features:
    if "_" in col:
        feature, value = col.split("_", 1)
        if feature in categorical_inputs and categorical_inputs[feature] == value:
            df_input[col] = 1
        else:
            df_input[col] = 0

# Add any missing columns
for col in model_features:
    if col not in df_input.columns:
        df_input[col] = 0

# Ensure correct column order
df_input = df_input[model_features]

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Price"):
    try:
        prediction = model.predict(df_input)[0]
        st.success(f"💰 Estimated Price: ₹ {int(prediction):,}")
    except Exception as e:
        st.error(f"Prediction error: {e}")