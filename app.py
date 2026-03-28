import os
import pickle
import pandas as pd
import streamlit as st

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗")

# -------------------------------
# Paths
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "car_price_prediction_model.pkl")
features_path = os.path.join(BASE_DIR, "model_features.pkl")

# -------------------------------
# Load Model + Features
# -------------------------------
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(features_path, "rb") as f:
        model_features = pickle.load(f)

except Exception as e:
    st.error(f"❌ Error loading model/files: {e}")
    st.stop()

# -------------------------------
# UI
# -------------------------------
st.title("🚗 Used Car Price Prediction App")
st.write("Fill the details below to estimate the car price 💰")

# -------------------------------
# Inputs
# -------------------------------

year = st.number_input(
    "Year of Manufacture",
    min_value=1990,
    max_value=2025,
    value=2018
)

mileage = st.number_input(
    "Kilometers Driven",
    min_value=0,
    step=1000
)

fuel_type = st.selectbox(
    "Fuel Type",
    ['Petrol', 'Diesel', 'Electric']
)

# Car model input
model_name = st.selectbox(
    "Car Model",
    [
        "Honda City",
        "Hyundai i20",
        "Maruti Swift",
        "Toyota Innova",
        "Hyundai Creta",
        "Tata Nexon"
    ]
)

accident = st.selectbox(
    "Accident History",
    ['None reported', 'At least 1 accident or damage reported']
)

clean_title = st.selectbox(
    "Clean Title",
    ['Yes', 'No']
)

# -------------------------------
# Prepare Input Data
# -------------------------------

# Create empty dataframe with all model features
df_input = pd.DataFrame(columns=model_features)
df_input.loc[0] = 0

# Fill numeric values
df_input['year'] = year
df_input['kilometers_driven'] = mileage

# Encode fuel type
fuel_col = f"fuel_type_{fuel_type}"
if fuel_col in df_input.columns:
    df_input[fuel_col] = 1

# Encode car model
model_col = f"model_{model_name}"
if model_col in df_input.columns:
    df_input[model_col] = 1

# Encode accident history
accident_col = f"accident_{accident}"
if accident_col in df_input.columns:
    df_input[accident_col] = 1

# Encode clean title
clean_col = f"clean_title_{clean_title}"
if clean_col in df_input.columns:
    df_input[clean_col] = 1

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
        st.error(f"❌ Prediction Error: {e}")

# -------------------------------
# Optional Debug
# -------------------------------
# st.write("Model Input:", df_input)
