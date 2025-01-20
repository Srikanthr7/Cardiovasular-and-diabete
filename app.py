import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load the pre-trained models
rf_cardio = RandomForestClassifier(random_state=42)
rf_cardio.load("rf_cardio_model.pkl")

rf_diabetes = RandomForestClassifier(random_state=42)
rf_diabetes.load("rf_diabetes_model.pkl")

# Function to make prediction for cardiovascular disease
def predict_cardio(features):
    return rf_cardio.predict([features])

# Function to make prediction for diabetes
def predict_diabetes(features):
    return rf_diabetes.predict([features])

# Streamlit UI
st.title("Cardiovascular Disease and Diabetes Prediction")

# Create user input fields for the cardiovascular dataset
st.header("Cardiovascular Disease Prediction")
age = st.number_input("Age (years)", min_value=18, max_value=100)
gender = st.radio("Gender", ("Male", "Female"))
height = st.number_input("Height (cm)", min_value=100, max_value=250)
weight = st.number_input("Weight (kg)", min_value=30, max_value=200)
blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=80, max_value=200)
cholesterol = st.number_input("Cholesterol level (mg/dL)", min_value=100, max_value=500)
glucose = st.number_input("Glucose level (mg/dL)", min_value=70, max_value=300)

# Create user input fields for the diabetes dataset
st.header("Diabetes Prediction")
pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20)
insulin = st.number_input("Insulin level", min_value=0, max_value=1000)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5)

# Prediction button
if st.button("Make Predictions"):
    # Prepare the input features for prediction
    cardio_features = [age, gender, height, weight, blood_pressure, cholesterol, glucose]
    diabetes_features = [pregnancies, insulin, bmi, diabetes_pedigree]
    
    # Make predictions
    cardio_prediction = predict_cardio(cardio_features)
    diabetes_prediction = predict_diabetes(diabetes_features)
    
    # Display the results
    st.subheader("Cardiovascular Disease Prediction Result")
    st.write("Risk of Cardiovascular Disease: " + ("Yes" if cardio_prediction[0] == 1 else "No"))
    
    st.subheader("Diabetes Prediction Result")
    st.write("Risk of Diabetes: " + ("Yes" if diabetes_prediction[0] == 1 else "No"))
