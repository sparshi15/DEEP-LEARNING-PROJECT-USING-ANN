import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load model and preprocessing tools
model = load_model('model.h5')
scaler = joblib.load('scaler.pkl')
label_encoder_gender = joblib.load('label_encoder_gender.pkl')
onehot_encoder_geo = joblib.load('onehot_encoder_geo.pkl')

st.title("Customer Churn Prediction App (CSV Upload)")
st.markdown("""
**📌 Note:**  
Please upload a CSV file with the following columns only:  
`CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`  

if possible do not include columns like `CustomerId`, `RowNumber`, `Surname`, or `Exited`, as they were not used during model training.
""")

uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)# Load the uploaded CSV

    # Drop columns not used during training
    data = data.drop(['CustomerId', 'RowNumber', 'Surname', 'Exited'], axis=1)
    # Encode 'Gender'
    data['Gender'] = label_encoder_gender.transform(data['Gender'])

    # One-hot encode 'Geography'
    geo_encoded = onehot_encoder_geo.transform(data[['Geography']]).toarray()
    geo_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
    data = pd.concat([data.drop(['Geography'], axis=1), geo_df], axis=1)

    # Scale features
    input_scaled = scaler.transform(data)

    # Predict churn probabilities
    predictions = model.predict(input_scaled).flatten()

    # Display results
    result_df = data.copy()
    result_df['Churn Probability'] = predictions
    result_df['Churn Prediction'] = np.where(predictions > 0.5, 'Likely to Churn', 'Not Likely to Churn')

    st.write(result_df)
else:
    st.info("Please upload a CSV file with the required columns.")
