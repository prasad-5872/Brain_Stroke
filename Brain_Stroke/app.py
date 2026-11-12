import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Load trained model
model = joblib.load('brain_stroke_model.pkl')

st.title('🧠 Brain Stroke Prediction App')
st.write(
    'Enter patient details below to predict the likelihood of a brain stroke.'
)

# Input form
with st.form('prediction_form'):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox('Gender', ['Male', 'Female'])
        age = st.number_input('Age', min_value=1.0, max_value=100.0, value=50.0)
        hypertension = st.selectbox(
            'Hypertension', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No'
        )
        heart_disease = st.selectbox(
            'Heart Disease', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No'
        )
        ever_married = st.selectbox('Ever Married', ['No', 'Yes'])

    with col2:
        work_type = st.selectbox(
            'Work Type',
            [
                'Private',
                'Self-employed',
                'Govt_job',
                'children',
                'Never_worked',
            ],
        )
        Residence_type = st.selectbox('Residence Type', ['Urban', 'Rural'])
        avg_glucose_level = st.number_input(
            'Average Glucose Level', min_value=50.0, max_value=300.0, value=100.0
        )
        bmi = st.number_input('BMI', min_value=10.0, max_value=60.0, value=25.0)
        smoking_status = st.selectbox(
            'Smoking Status',
            ['never smoked', 'formerly smoked', 'smokes', 'Unknown'],
        )

    submit_button = st.form_submit_button(label='Predict Stroke Risk')

if submit_button:
    input_data = pd.DataFrame(
        {
            'gender': [gender],
            'age': [age],
            'hypertension': [hypertension],
            'heart_disease': [heart_disease],
            'ever_married': [ever_married],
            'work_type': [work_type],
            'Residence_type': [Residence_type],
            'avg_glucose_level': [avg_glucose_level],
            'bmi': [bmi],
            'smoking_status': [smoking_status],
        }
    )

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader('Prediction Result:')
    if prediction == 1:
        st.error(
            f'⚠️ High Risk: The model predicts a higher risk of stroke.'
            f' (Probability: {probability:.2%})'
        )
    else:
        st.success(
            f'✅ Low Risk: The model predicts a lower risk of stroke.'
            f' (Probability: {probability:.2%})'
        )