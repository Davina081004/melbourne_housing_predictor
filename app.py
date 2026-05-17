import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load model and scaler
model = joblib.load('housing_model.pkl')
scaler = joblib.load('scaler.pkl')

# App title
st.title('🏠 Melbourne Housing Price Predictor')
st.write('Enter the property details below to get a predicted price')

# Input fields
suburb = st.selectbox('Suburb', ['Hawthorn', 'Ringwood', 'Burwood'])
property_type = st.selectbox('Property Type', ['House', 'Unit'])
bedrooms = st.slider('Bedrooms', 1, 6, 3)
bathrooms = st.slider('Bathrooms', 1, 4, 2)
car_spaces = st.slider('Car Spaces', 0, 5, 1)
land_size = st.number_input('Land Size (m2)', min_value=0, max_value=2000, value=500)
travel_time = st.slider('Travel Time to CBD (min)', 15, 50, 30)
has_garden = st.selectbox('Has Garden?', ['Yes', 'No'])
sale_year = st.selectbox('Sale Year', [2024, 2025, 2026])
sale_month = st.slider('Sale Month', 1, 12, 6)

# Predict button
if st.button('Predict Price'):

    # Encode inputs
    suburb_burwood = 1 if suburb == 'Burwood' else 0
    suburb_hawthorn = 1 if suburb == 'Hawthorn' else 0
    suburb_ringwood = 1 if suburb == 'Ringwood' else 0
    type_house = 1 if property_type == 'House' else 0
    type_unit = 1 if property_type == 'Unit' else 0
    garden = 1 if has_garden == 'Yes' else 0

    # Scale numerical features
    numerical = scaler.transform([[bedrooms, bathrooms, car_spaces,
                                   land_size, travel_time,
                                   sale_year, sale_month]])

    # Build input array
    features = np.array([[
        numerical[0][0], numerical[0][1], numerical[0][2],
        numerical[0][3], numerical[0][4], garden,
        numerical[0][5], numerical[0][6],
        suburb_burwood, suburb_hawthorn, suburb_ringwood,
        type_house, type_unit
    ]])

    # Make prediction
    prediction = model.predict(features)[0]

    # Show result
    st.success(f'💰 Predicted Price: ${prediction:,.0f}')