import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

# Define a function to load the model and preprocessor
# Using st.cache_resource to cache the model for performance
@st.cache_resource
def load_assets():
    """
    Loads the preprocessor and the deep learning model.
    """
    try:
        # Load the preprocessor
        with open('preprocessor.pkl', 'rb') as file:
            preprocessor = pickle.load(file)
        
        # Load the trained Keras model
        model = tf.keras.models.load_model('best_ann_model.h5')
        return preprocessor, model
    except FileNotFoundError:
        st.error("Error: The 'preprocessor.pkl' or 'best_ann_model.h5' file was not found.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the assets: {e}")
        st.stop()

# Load the assets
preprocessor, model = load_assets()

# --- Streamlit UI ---
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("Customer Churn Prediction App")
st.markdown("""
This application predicts whether a customer will churn based on their demographic and service data.
Fill in the details below and click 'Predict' to see the result.
""")

# Create a sidebar for user input
st.sidebar.header("Customer Information")

# Function to get user input from the sidebar
def user_input_features():
    # User input for each feature
    gender = st.sidebar.selectbox("Gender", ('Male', 'Female'))
    SeniorCitizen = st.sidebar.selectbox("Senior Citizen", ('No', 'Yes'))
    Partner = st.sidebar.selectbox("Partner", ('No', 'Yes'))
    Dependents = st.sidebar.selectbox("Dependents", ('No', 'Yes'))
    tenure = st.sidebar.slider("Tenure (in months)", 0, 72, 30)
    PhoneService = st.sidebar.selectbox("Phone Service", ('No', 'Yes'))
    MultipleLines = st.sidebar.selectbox("Multiple Lines", ('No phone service', 'No', 'Yes'))
    InternetService = st.sidebar.selectbox("Internet Service", ('DSL', 'Fiber optic', 'No'))
    OnlineSecurity = st.sidebar.selectbox("Online Security", ('No internet service', 'No', 'Yes'))
    OnlineBackup = st.sidebar.selectbox("Online Backup", ('No internet service', 'No', 'Yes'))
    DeviceProtection = st.sidebar.selectbox("Device Protection", ('No internet service', 'No', 'Yes'))
    TechSupport = st.sidebar.selectbox("Tech Support", ('No internet service', 'No', 'Yes'))
    StreamingTV = st.sidebar.selectbox("Streaming TV", ('No internet service', 'No', 'Yes'))
    StreamingMovies = st.sidebar.selectbox("Streaming Movies", ('No internet service', 'No', 'Yes'))
    Contract = st.sidebar.selectbox("Contract", ('Month-to-month', 'One year', 'Two year'))
    PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ('No', 'Yes'))
    PaymentMethod = st.sidebar.selectbox("Payment Method", ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
    MonthlyCharges = st.sidebar.slider("Monthly Charges", 18.0, 120.0, 60.0)
    TotalCharges = st.sidebar.slider("Total Charges", 0.0, 8700.0, 1000.0)

    # Store data in a dictionary
    data = {'gender': gender,
            'SeniorCitizen': SeniorCitizen,
            'Partner': Partner,
            'Dependents': Dependents,
            'tenure': tenure,
            'PhoneService': PhoneService,
            'MultipleLines': MultipleLines,
            'InternetService': InternetService,
            'OnlineSecurity': OnlineSecurity,
            'OnlineBackup': OnlineBackup,
            'DeviceProtection': DeviceProtection,
            'TechSupport': TechSupport,
            'StreamingTV': StreamingTV,
            'StreamingMovies': StreamingMovies,
            'Contract': Contract,
            'PaperlessBilling': PaperlessBilling,
            'PaymentMethod': PaymentMethod,
            'MonthlyCharges': MonthlyCharges,
            'TotalCharges': TotalCharges}
            
    # Convert dictionary to DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = user_input_features()

# Display user input
st.subheader("User Input Parameters")
st.write(input_df)

# Prediction button
if st.button("Predict"):
    # Preprocess the input data
    processed_data = preprocessor.transform(input_df)
    
    # Make prediction
    prediction = model.predict(processed_data)
    
    # Convert the prediction probability to a single value
    churn_probability = prediction[0][0]
    
    # Display the result
    st.subheader("Prediction Result")
    
    if churn_probability > 0.5:
        st.metric(label="Churn Prediction", value="High Risk (Likely to Churn)", delta=f"{churn_probability:.2f}", delta_color="inverse")
        st.error(f"The model predicts a high probability of churn: {churn_probability:.2f}")
    else:
        st.metric(label="Churn Prediction", value="Low Risk (Not Likely to Churn)", delta=f"{churn_probability:.2f}", delta_color="normal")
        st.success(f"The model predicts a low probability of churn: {churn_probability:.2f}")
    
    st.markdown("---")
    st.info("The `delta` value in the metric widget above shows the churn probability. A value greater than 0.5 indicates a higher risk of churn.")
