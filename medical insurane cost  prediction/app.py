import streamlit as st
import numpy as np
import pickle

# ------------------ LOAD MODEL ------------------
model = pickle.load(open("insurance_model.pkl", "rb"))

# ------------------ PAGE SETTINGS ------------------
st.set_page_config(page_title="Insurance Predictor", page_icon="💰")

st.title("💰 Medical Insurance Cost Predictor")
st.write("Enter details below to estimate your insurance cost")

# ------------------ SIDEBAR INFO ------------------
st.sidebar.title("Model Information")
st.sidebar.write("Algorithm: Linear Regression")
st.sidebar.write("Expected Accuracy (R²): ~0.75")
st.sidebar.write("Dataset: Medical Insurance")

# ------------------ USER INPUT ------------------
age = st.slider("Age", 18, 100, 25)
bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
children = st.number_input("Number of Children", 0, 10, 0)

sex = st.selectbox("Sex", ["Male", "Female"])
smoker = st.selectbox("Smoker", ["Yes", "No"])
region = st.selectbox("Region", ["Southeast", "Southwest", "Northeast", "Northwest"])

# ------------------ ENCODING (MATCH TRAINING) ------------------
sex_value = 0 if sex == "Male" else 1
smoker_value = 0 if smoker == "Yes" else 1

region_dict = {
    "Southeast": 0,
    "Southwest": 1,
    "Northeast": 2,
    "Northwest": 3
}

region_value = region_dict[region]

input_data = np.array([[age, sex_value, bmi, children, smoker_value, region_value]])

# ------------------ SESSION STATE ------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ------------------ PREDICTION ------------------
if st.button("Predict Insurance Cost"):

    prediction = model.predict(input_data)[0]

    st.success(f"Estimated Insurance Cost: ₹ {prediction:,.2f}")

    # -------- Risk Level --------
    if prediction < 10000:
        st.info("🟢 Low Insurance Risk")
    elif prediction < 30000:
        st.warning("🟡 Medium Insurance Risk")
    else:
        st.error("🔴 High Insurance Risk")

    # -------- Explanation --------
    st.subheader("Why this cost?")
    if smoker == "Yes":
        st.write("⚠ Smoking significantly increases insurance cost")
    if bmi > 30:
        st.write("⚠ High BMI increases health risk")
    if age > 50:
        st.write("⚠ Higher age increases premium")

    # -------- Comparison Chart --------
    st.subheader("Your Cost vs Average Cost")
    average_cost = 13270
    st.bar_chart([prediction, average_cost])

    # -------- Save History --------
    st.session_state.history.append(round(prediction, 2))

# ------------------ HISTORY ------------------
if st.session_state.history:
    st.subheader("Previous Predictions")
    st.write(st.session_state.history)
