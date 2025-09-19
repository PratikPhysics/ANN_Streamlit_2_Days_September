import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Silence TensorFlow logs (optional)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Silence Pandas FutureWarning
pd.set_option('future.no_silent_downcasting', True)

# Load the model and preprocessor
try:
    model = tf.keras.models.load_model('best_ann_model.h5')
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model or preprocessor: {e}")
    st.stop()

# Set up the Streamlit app
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("Customer Churn Prediction")
st.markdown("Enter customer details to predict if they will churn.")

# Create input widgets for user data
with st.container():
    st.header("Customer Information")
    age = st.slider("Age", 18, 90, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
    has_phone_service = st.selectbox("Has Phone Service?", ["Yes", "No"])
    multiple_lines = st.selectbox("Has Multiple Lines?", ["Yes", "No"])
    has_internet_service = st.selectbox("Has Internet Service?", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=float(monthly_charges * tenure))

    # Auto-calculate minimum expected total charges
    expected_min_total = monthly_charges * tenure
    if total_charges < expected_min_total:
        st.warning(f"💡 Total Charges seems low. Expected at least ${expected_min_total:.2f} for {tenure} months.")

    # Dynamic Senior Citizen based on age (65+)
    senior_citizen = 1 if age >= 65 else 0
    partner = st.selectbox("Has Partner?", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents?", ["No", "Yes"])

    # Create a dictionary from the user inputs — ALL VALUES AS STRINGS OR ORIGINAL TYPE
    input_data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,  # This is numerical — will be scaled
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,  # Numerical
        'PhoneService': has_phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': has_internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,  # Numerical
        'TotalCharges': total_charges,  # Numerical
    }

# Convert input data to a DataFrame
input_df = pd.DataFrame([input_data])

# Button to make a prediction
if st.button("Predict Churn"):
    try:
        # ✅ CRITICAL: DO NOT MANUALLY ENCODE — Let preprocessor handle it
        # The preprocessor expects raw categories (strings) for categorical columns
        # and raw numbers for numerical columns — just like during training.

        # Ensure numerical columns are float (optional safety)
        numerical_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
        for col in numerical_cols:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)

        # ✅ Apply the preprocessor — it will OneHotEncode categorical and Scale numerical
        input_processed = preprocessor.transform(input_df)

        # Make prediction
        prediction = model.predict(input_processed)
        churn_probability = prediction[0][0]

        # Display result
        st.subheader("Prediction Result")
        if churn_probability > 0.5:
            st.error(f"🚨 Prediction: This customer is likely to churn. (Probability: {churn_probability:.2%})")
        else:
            st.success(f"✅ Prediction: This customer is not likely to churn. (Probability: {churn_probability:.2%})")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.exception(e)  # For debugging — remove in production

st.markdown("---")
st.caption("💡 Tip: Adjust inputs to see how churn probability changes.")
st.info("This app is a demonstration of a machine learning model deployed with Streamlit.")
