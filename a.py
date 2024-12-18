import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the trained model, encoder, and scaler
with open('comparison_df_lr.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('encoder.pkl', 'rb') as encoder_file:
    one_hot_encoder = pickle.load(encoder_file)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Streamlit app starts here
st.title("Car Price Prediction Application")

# Load dataset for dropdown options
details = pd.read_csv('C:/Users/HP/Desktop/car_dekho_files/new files/overall_details.csv')

# Separate columns for categorical and numerical features
categorical_cols = ['ft', 'bt', 'transmission', 'ownerNo', 'oem', 'model', 'modelYear', 'centralVariantId', 'variantName']
numerical_cols = ['km']  # Assuming 'km' is the numerical feature for mileage

# Define user input options with dynamic filtering
ft = st.selectbox('Fuel Type', details['ft'].unique())
filtered_details = details[details['ft'] == ft]

bt = st.selectbox('Body Type', filtered_details['bt'].unique())
filtered_details = filtered_details[filtered_details['bt'] == bt]

transmission = st.selectbox('Transmission', filtered_details['transmission'].unique())
filtered_details = filtered_details[filtered_details['transmission'] == transmission]

ownerNo = st.selectbox('Number of Previous Owners', filtered_details['ownerNo'].unique())
filtered_details = filtered_details[filtered_details['ownerNo'] == ownerNo]

oem = st.selectbox('Car Brand (OEM)', filtered_details['oem'].unique())
filtered_details = filtered_details[filtered_details['oem'] == oem]

model_name = st.selectbox('Model Name', filtered_details['model'].unique())
filtered_details = filtered_details[filtered_details['model'] == model_name]

modelYear = st.selectbox('Model Year', filtered_details['modelYear'].unique())
filtered_details = filtered_details[filtered_details['modelYear'] == modelYear]

centralVariantId = st.selectbox('Variant ID', filtered_details['centralVariantId'].unique())
filtered_details = filtered_details[filtered_details['centralVariantId'] == centralVariantId]

variantName = st.selectbox('Variant Name', filtered_details['variantName'].unique())

# Collect user input for the numerical feature
km = st.number_input('Kilometers Driven', min_value=0)

# Organize user input into a DataFrame
user_input = pd.DataFrame({
    'ft': [ft],
    'bt': [bt],
    'transmission': [transmission],
    'ownerNo': [ownerNo],
    'oem': [oem],
    'model': [model_name],
    'modelYear': [modelYear],
    'centralVariantId': [centralVariantId],
    'variantName': [variantName],
    'km': [km]
})

# Step 1: Apply One-Hot Encoding to the categorical data
encoded_user_input = one_hot_encoder.transform(user_input[categorical_cols])
encoded_user_input_df = pd.DataFrame(encoded_user_input, columns=one_hot_encoder.get_feature_names_out(categorical_cols))

# Step 2: Scale the numerical feature (km)
scaled_num = scaler.transform(user_input[numerical_cols])
scaled_num_df = pd.DataFrame(scaled_num, columns=numerical_cols)

# Step 3: Combine encoded categorical and scaled numerical features
processed_input = pd.concat([encoded_user_input_df, scaled_num_df], axis=1)

# Ensure all features match the training data
missing_cols = set(model.feature_names_in_) - set(processed_input.columns)
for col in missing_cols:
    processed_input[col] = 0

# Reorder columns to match model training order
processed_input = processed_input[model.feature_names_in_]

# Make predictions with the model
if st.button('Predict Price'):
    prediction = model.predict(processed_input)
    st.success(f"Predicted Price: ₹{round(prediction[0], 2)} lakhs")

# Display user input for debugging purposes
st.write("User Input Data:", user_input)
