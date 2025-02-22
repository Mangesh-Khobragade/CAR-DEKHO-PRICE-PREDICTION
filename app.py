import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler

with open('comparison_df_rf.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('encoder.pkl', 'rb') as encoder_file:
    one_hot_encoder = pickle.load(encoder_file)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file) 
# Streamlit app starts here
st.title("Car Price Prediction Application")

# Load dataset for dropdown options
details = pd.read_csv('C:/Users/mange/Desktop/MY PROJECTS/CAR DEKHO/car_dekho_files/new files/overalldetails.csv')

# Define relevant categorical and numerical columns
categorical_cols = ['ft', 'bt', 'transmission', 'ownerNo', 'oem', 'model', 'modelYear', 'variantName']
numerical_cols = ['km']

# Collect user input dynamically
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

variantName = st.selectbox('Variant Name', filtered_details['variantName'].unique())

# Collect user input for numerical feature
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
    'variantName': [variantName],
    'km': [km]
})

# Apply One-Hot Encoding and Scale numerical feature
encoded_user_input = one_hot_encoder.transform(user_input[categorical_cols])
encoded_user_input_df = pd.DataFrame(encoded_user_input, columns=one_hot_encoder.get_feature_names_out(categorical_cols))
scaled_num = scaler.transform(user_input[numerical_cols])
scaled_num_df = pd.DataFrame(scaled_num, columns=numerical_cols)

# Combine processed features
processed_input = pd.concat([encoded_user_input_df, scaled_num_df], axis=1)

# Ensure all features match the training data
missing_cols = set(model.feature_names_in_) - set(processed_input.columns)
for col in missing_cols:
    processed_input[col] = 0
processed_input = processed_input[model.feature_names_in_]

# Make predictions
if st.button('Predict Price'):
    prediction = model.predict(processed_input)
    st.success(f"Predicted Price: ₹{round(prediction[0], 2)} lakhs")