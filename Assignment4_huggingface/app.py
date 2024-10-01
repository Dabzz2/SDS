import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler, OneHotEncoder
import shap
from streamlit_shap import st_shap

# Page configuration
st.set_page_config(
    page_title="Kiva loan amount predictor",
    page_icon="💰")

st.title('Predict Kiva loan amounts')


# Load model and preprocessing objects
@st.cache_resource
def load_model_objects():
    model_rf = joblib.load('best_reg.joblib')
    scaler = joblib.load('scaler.joblib')
    ohe = joblib.load('ohe.joblib')
    return model_rf, scaler, ohe

model_rf, scaler, ohe = load_model_objects()

# Create SHAP explainer
explainer = shap.TreeExplainer(model_rf)

# App description
with st.expander("What's this app?"):
    st.markdown("""
    This app helps you determine how much you will be succesfully funded with on Kiva
    """)

st.subheader('Describe what you want to loan to')

# User inputs
col1, col2 = st.columns(2)

with col1:
    Sector = st.selectbox('sector', options=ohe.categories_[0])
    Country = st.selectbox('country', options=ohe.categories_[1])
    Gender = st.selectbox('borrower_genders', options=ohe.categories_[2])

with col2:
    term_in_months = st.number_input('Lenght of loan in months', min_value=0, value=1)
    lender_count = st.number_input('Number of Lenders', min_value=1,value=1)

# Prediction button
if st.button('Predict loan amount 🚀'):
    # Prepare categorical features
    cat_features = pd.DataFrame({'sector': [Sector], 'country': [Country],'borrower_genders': [Gender]})
    cat_encoded = pd.DataFrame(ohe.transform(cat_features).todense(), 
                               columns=ohe.get_feature_names_out(['sector', 'country', 'borrower_genders']))
    
    # Prepare numerical features
    num_features = pd.DataFrame({
        'term_in_months': [term_in_months],
        'lender_count': [lender_count],
       })
    num_scaled = pd.DataFrame(scaler.transform(num_features), columns=num_features.columns)
    
    # Combine features
    features = pd.concat([num_scaled, cat_encoded], axis=1)
    
    # Make prediction
    predicted_price = model_rf.predict(features)[0]
    
    # Display prediction
    st.metric(label="Predicted loan amount", value=f'{round(predicted_price)} USD')
    
    
    # SHAP explanation
    st.subheader('Price Factors Explained 🤖')
    shap_values = explainer.shap_values(features)
    st_shap(shap.force_plot(explainer.expected_value, shap_values, features), height=400, width=600)
    
    st.markdown("""
    This plot shows how each feature contributes to the predicted price:
    - Blue bars push the price lower
    - Red bars push the price higher
    - The length of each bar indicates the strength of the feature's impact
    """)

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit")