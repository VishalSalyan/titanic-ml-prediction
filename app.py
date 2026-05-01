import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: white;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-size: 18px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)


# -----------------------
# Load Model
# -----------------------
model_package = pickle.load(open("model/final_model.sav", "rb"))
model = model_package["model"]
model_name = model_package["model_name"]

st.set_page_config(page_title="Titanic Predictor", page_icon="🚢")

st.title("🚢 Titanic Survival Prediction")
st.subheader(f"Model: {model_name}")

# -----------------------
# Input Form
# -----------------------
with st.form("form"):

    col1, col2 = st.columns(2)

    with col1:
        Pclass = st.selectbox("Passenger Class", [1, 2, 3])
        Sex = st.selectbox("Sex", ["Male", "Female"])
        Age = st.slider("Age", 0, 80, 25)
        Fare = st.number_input("Fare", 0.0, 500.0, 50.0)

    with col2:
        FamilySize = st.slider("Family Size", 0, 10, 1)
        IsAlone = st.selectbox("Is Alone", [0, 1])
        Title = st.selectbox("Title", [0, 1, 2, 3])  # depends on your encoding
        AgeGroup = st.selectbox("Age Group", [0, 1, 2, 3])
        FareBin = st.selectbox("Fare Bin", [0, 1, 2, 3])

    submit = st.form_submit_button("Predict")

# -----------------------
# Prediction
# -----------------------
if submit:

    # Encode Sex
    Sex = 1 if Sex == "Male" else 0

    # ⚠️ MUST MATCH TRAINING ORDER EXACTLY
    input_data = np.array([[
        Pclass,
        Sex,
        Age,
        Fare,
        FamilySize,
        IsAlone,
        Title,
        AgeGroup,
        FareBin
    ]])

    prediction = model.predict(input_data)[0]

    st.markdown("---")

    if prediction == 1:
        st.success("✅ Survived")
    else:
        st.error("❌ Did Not Survive")

    # Optional probability
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_data)[0][1]
        st.info(f"Survival Probability: {prob:.2f}")