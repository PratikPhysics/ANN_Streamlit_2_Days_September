import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

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
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=50.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, value=600.0)
    
    # Create a dictionary from the user inputs
    input_data = {
        'gender': gender,
        'SeniorCitizen': 0, # Assuming no senior citizen field for simplicity, set to 0
        'Partner': 'No',    # Assuming no partner field for simplicity, set to 'No'
        'Dependents': 'No', # Assuming no dependents field for simplicity, set to 'No'
        'tenure': tenure,
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
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
    }

# Convert input data to a DataFrame
input_df = pd.DataFrame([input_data])

# Button to make a prediction
if st.button("Predict Churn"):
    try:
        # Preprocess the input data
        # Handle 'No' and 'Yes' values to avoid the ValueError
        for col in ['PhoneService', 'MultipleLines', 'PaperlessBilling', 'Partner', 'Dependents']:
            if col in input_df.columns:
                input_df[col] = input_df[col].replace({'Yes': 1, 'No': 0})
        
        # Handle 'No' and 'Yes' for internet service columns
        internet_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        for col in internet_cols:
            if col in input_df.columns:
                input_df[col] = input_df[col].replace({'Yes': 1, 'No': 0, 'No internet service': 0})

        # Apply the preprocessor
        input_processed = preprocessor.transform(input_df)

        # Make a prediction
        prediction = model.predict(input_processed)
        churn_probability = prediction[0][0]

        # Display the result
        st.subheader("Prediction Result")
        if churn_probability > 0.5:
            st.error(f"Prediction: This customer is likely to churn. (Probability: {churn_probability:.2f})")
        else:
            st.success(f"Prediction: This customer is not likely to churn. (Probability: {churn_probability:.2f})")
    
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.markdown("---")
st.info("This app is a demonstration of a machine learning model deployed with Streamlit.")
