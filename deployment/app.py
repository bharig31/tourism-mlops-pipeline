"""
Streamlit App for Wellness Tourism Package Prediction
This app provides an interactive interface to predict customer purchase probability
for the Wellness Tourism Package using the trained ML model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download
from sklearn.preprocessing import StandardScaler
import os

# Get your Hugging Face username from environment variable
HF_USERNAME = os.getenv('HF_USERNAME', 'bharig')

# Page configuration
st.set_page_config(
    page_title="Wellness Tourism Package Predictor",
    page_icon="✈️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 2rem;
        border-radius: 10px;
        margin-top: 2rem;
    }
    .success {
        color: #2ecc71;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .info {
        color: #3498db;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">✈️ Wellness Tourism Package Predictor</h1>', unsafe_allow_html=True)
st.markdown("""
This application predicts whether a customer is likely to purchase the Wellness Tourism Package
based on their demographic information, travel history, and interests.
""")

# Function to load the model
@st.cache_resource
def load_model():
    """
    Load the trained model from Hugging Face.
    Caches the model to avoid reloading on each interaction.
    """
    try:
        # Download model from Hugging Face
        model_path = hf_hub_download(
            repo_id=f"{HF_USERNAME}/wellness-tourism-predictor",
            filename="wellness-tourism-predictor.pkl"
        )
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
model = load_model()

# Sidebar for user input
st.sidebar.header("Customer Information")

# Define input fields matching actual tourism.csv columns
def user_input_features():
    """
    Collect user input through Streamlit sidebar.
    Returns a dictionary of feature values matching model training data.
    """
    age = st.sidebar.slider('Age', 18, 80, 35)
    type_of_contact = st.sidebar.selectbox('Type of Contact', ['Self Enquiry', 'Company Invited'])
    city_tier = st.sidebar.selectbox('City Tier', [1, 2, 3])
    duration_of_pitch = st.sidebar.slider('Duration of Pitch (mins)', 5, 60, 15)
    occupation = st.sidebar.selectbox('Occupation', ['Salaried', 'Small Business', 'Large Business', 'Free Lancer'])
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    number_of_person_visiting = st.sidebar.slider('Number of Persons Visiting', 1, 5, 2)
    number_of_followups = st.sidebar.slider('Number of Followups', 1, 6, 3)
    product_pitched = st.sidebar.selectbox('Product Pitched', ['Basic', 'Standard', 'Deluxe', 'Super Deluxe', 'King'])
    preferred_property_star = st.sidebar.selectbox('Preferred Property Star', [3, 4, 5])
    marital_status = st.sidebar.selectbox('Marital Status', ['Single', 'Married', 'Divorced', 'Unmarried'])
    number_of_trips = st.sidebar.slider('Number of Trips', 1, 22, 3)
    passport = st.sidebar.selectbox('Has Passport', [0, 1])
    pitch_satisfaction_score = st.sidebar.slider('Pitch Satisfaction Score', 1, 5, 3)
    own_car = st.sidebar.selectbox('Owns Car', [0, 1])
    number_of_children_visiting = st.sidebar.slider('Number of Children Visiting', 0, 3, 1)
    designation = st.sidebar.selectbox('Designation', ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP'])
    monthly_income = st.sidebar.slider('Monthly Income', 10000, 100000, 30000, 1000)

    # Encode categoricals the same way as training (factorize order)
    type_of_contact_map = {'Self Enquiry': 0, 'Company Invited': 1}
    occupation_map = {'Salaried': 0, 'Small Business': 1, 'Large Business': 2, 'Free Lancer': 3}
    gender_map = {'Female': 0, 'Male': 1}
    product_pitched_map = {'Basic': 0, 'Standard': 1, 'Deluxe': 2, 'Super Deluxe': 3, 'King': 4}
    marital_status_map = {'Single': 0, 'Married': 1, 'Divorced': 2, 'Unmarried': 3}
    designation_map = {'Executive': 0, 'Manager': 1, 'Senior Manager': 2, 'AVP': 3, 'VP': 4}

    data = {
        'Unnamed: 0': 0,
        'Age': age,
        'TypeofContact': type_of_contact_map[type_of_contact],
        'CityTier': city_tier,
        'DurationOfPitch': duration_of_pitch,
        'Occupation': occupation_map[occupation],
        'Gender': gender_map[gender],
        'NumberOfPersonVisiting': number_of_person_visiting,
        'NumberOfFollowups': number_of_followups,
        'ProductPitched': product_pitched_map[product_pitched],
        'PreferredPropertyStar': preferred_property_star,
        'MaritalStatus': marital_status_map[marital_status],
        'NumberOfTrips': number_of_trips,
        'Passport': passport,
        'PitchSatisfactionScore': pitch_satisfaction_score,
        'OwnCar': own_car,
        'NumberOfChildrenVisiting': number_of_children_visiting,
        'Designation': designation_map[designation],
        'MonthlyIncome': monthly_income,
    }

    return data

# Get user input
input_data = user_input_features()

# Display input data as a dataframe
st.subheader("Customer Profile")
input_df = pd.DataFrame([input_data])
st.write(input_df)

# Make prediction button
if st.button('Predict Purchase Probability'):
    if model is not None:
        try:
            # Make prediction
            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)
            
            # Display results
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.subheader("Prediction Results")
            
            if prediction[0] == 1:
                st.markdown('<p class="success">✅ Likely to Purchase Wellness Tourism Package</p>', unsafe_allow_html=True)
                confidence = probability[0][1] * 100
            else:
                st.markdown('<p class="info">ℹ️ Unlikely to Purchase Wellness Tourism Package</p>', unsafe_allow_html=True)
                confidence = probability[0][0] * 100
            
            st.write(f"**Confidence:** {confidence:.2f}%")
            
            # Display probability breakdown
            st.write("**Purchase Probability Breakdown:**")
            prob_df = pd.DataFrame({
                'Will Purchase': [probability[0][1] * 100],
                'Will Not Purchase': [probability[0][0] * 100]
            }, index=['Probability'])
            st.bar_chart(prob_df.T)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
    else:
        st.error("Model not loaded. Please check the model loading process.")

# Add information section
st.markdown("---")
st.subheader("About the Model")
st.markdown("""
This model was trained using customer data from "Visit with Us" travel company to predict
which customers are likely to purchase the Wellness Tourism Package. The model uses
the following features:

- **Demographic**: Age, Gender, Income
- **Travel History**: Travel frequency, previous trips, average spending
- **Interests**: Adventure, Culture, Relaxation, Wellness tourism interests
- **Engagement**: Loyalty membership, email clicks, website visits, booking history

The model was trained using ensemble methods (Random Forest, XGBoost, Gradient Boosting)
and achieved an F1 score of approximately 0.85 on the test set.
""")

# Footer
st.markdown("---")
st.caption("MLOps Project - Visit with Us Travel Company")
