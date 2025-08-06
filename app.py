import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import threading

# Optional lock to prevent concurrent prediction issues
predict_lock = threading.Lock()

# ----------- CACHED LOAD FUNCTIONS -------------------

@st.cache_resource
def load_keras_model():
    return load_model('model.keras')

@st.cache_resource
def load_gender_encoder():
    with open('data_transformers/gender_encoder.pkl', 'rb') as file:
        return pickle.load(file)

@st.cache_resource
def load_geo_encoder():
    with open('data_transformers/geo_encoder.pkl', 'rb') as file:
        return pickle.load(file)

@st.cache_resource
def load_scaler():
    with open('data_transformers/scaler.pkl', 'rb') as file:
        return pickle.load(file)

# ----------- LOAD MODEL AND TRANSFORMERS -------------------

model = load_keras_model()
gender_encoder = load_gender_encoder()
geo_encoder = load_geo_encoder()
scaler = load_scaler()

# ----------- STREAMLIT PAGE CONFIG -------------------

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📈", layout="centered")

# ----------- CUSTOM CSS STYLING -------------------

st.markdown("""
    <style>
    .main {
        background-color: #f9faff;
        color: #333333;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .title {
        color: #2B6CB0;
        font-weight: 700;
        font-size: 3rem;
        margin-bottom: 10px;
    }
    .subtitle {
        color: #3182CE;
        font-size: 1.2rem;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    label {
        font-weight: 600;
        color: #2C5282;
    }
    div.stButton > button:first-child {
        background-color: #3182CE;
        color: white;
        font-weight: 600;
        padding: 0.6em 1.4em;
        border-radius: 10px;
    }
    div.stButton > button:hover {
        background-color: #2B6CB0;
        color: #e2e8f0;
    }
    .prediction-box {
        background-color: #bee3f8;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        color: #2a4365;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

# ----------- PAGE TITLE -------------------

st.markdown('<h1 class="title">📈 Customer Churn Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Fill in the customer data below to get the churn prediction probability.</p>', unsafe_allow_html=True)

# ----------- INPUT FIELDS -------------------

col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
    gender = st.selectbox("Gender", gender_encoder.classes_)
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    tenure = st.slider("Tenure (Years with Bank)", 0, 10, 5)

with col2:
    balance = st.number_input("Account Balance", min_value=0.0, step=100.0, value=50000.0)
    num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
    has_cr_card = 1 if st.selectbox("Has Credit Card?", ['No', 'Yes']) == 'Yes' else 0
    is_active_member = 1 if st.selectbox("Is Active Member?", ['No', 'Yes']) == 'Yes' else 0

estimated_salary = st.number_input("Estimated Salary", min_value=0.0, step=100.0, value=50000.0)
geography_options = geo_encoder.categories_[0]
geography = st.selectbox("Geography", geography_options)

# ----------- PREDICTION BUTTON -------------------

if st.button("Predict"):
    try:
        with predict_lock:
            # Encode categorical features
            gender_encoded = gender_encoder.transform([gender])[0]
            geo_encoded = geo_encoder.transform([[geography]]).toarray()[0]

            # Assemble input features
            input_data = np.array([[credit_score, gender_encoded, age, tenure, balance,
                                    num_of_products, has_cr_card, is_active_member, estimated_salary]])
            input_data = np.concatenate((input_data, geo_encoded.reshape(1, -1)), axis=1)

            # Scale input
            input_scaled = scaler.transform(input_data)

            # Predict churn probability
            prediction = model.predict(input_scaled)[0][0]
            predicted_class = "Churn" if prediction >= 0.5 else "Stay"

            # Display results
            st.markdown(f"""
                <div class="prediction-box">
                <h3>📊 Prediction Result</h3>
                <p><strong>Probability of Churn:</strong> {prediction:.2f}</p>
                <p><strong>Predicted Class:</strong> {predicted_class}</p>
                </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ An error occurred during prediction: {str(e)}")
