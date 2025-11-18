import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Custom CSS using harmonious reds, oranges, and warm earth tones (no blue or its shades)
custom_css = """
<style>
body {
    background: linear-gradient(135deg, #322a27 0%, #322a27 100%) !important;
}
.stApp, .main, .block-container {
    background: linear-gradient(120deg,#5d4e49 0%,#5d4e49 100%) !important;
}
.report-container {
    background: linear-gradient(135deg, #322a27 0%, #322a27 100%);
    border-radius: 18px;
    box-shadow: 0 2px 18px 0 rgba(203, 94, 94, 0.11);
    padding: 2.5rem 2.5rem 1.5rem 2.5rem;
    margin: 1.5rem 0 1.5rem 0;
}
.report-title {
    font-size: 2rem;
    color: #c0392b;
    font-weight: bold;
    text-align: center;
    letter-spacing: 1px;
    margin-bottom: 2rem;
}
.result-high {
    color: #c0392b;     /* dark red */
    font-size: 1.3rem;
    font-weight: 700;
}
.result-low {
    color: #d35400;     /* pumpkin/orange */
    font-size: 1.3rem;
    font-weight: 700;
}
.proba-bar {
    height: 2.2rem;
    border-radius: 1.2rem;
    margin: 14px 0;
    background: #ffe2db;  /* light red background */
    box-shadow: 0 1px 6px 0 rgba(192, 57, 43, 0.13);
    overflow: hidden;
    position: relative;
}
.proba-high {
    background: linear-gradient(90deg,#e74c3c 0%,#f37c2a 100%);
    height: 100%;
    border-radius: 1.2rem 0 0 1.2rem;
}
.proba-low {
    background: linear-gradient(90deg,#ffc371 0%,#e96443 100%);
    height: 100%;
    border-radius: 1.2rem 0 0 1.2rem;
}
.report-section-label {
    color: #a63d2d;
    font-weight: 600;
    letter-spacing: 0.02em;
    margin-bottom: 7px;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

st.set_page_config(layout="wide")

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# Define the preprocessing steps (LabelEncoders) as used in the notebook
def get_label_encoders(df):
    encoders = {}
    categorical_cols = [
        'sex','chest_pain_type','fasting_blood_sugar',
        'resting_electrocardiogram','exercise_induced_angina',
        'st_slope','thalassemia'
    ]
    dummy_data = {
        'sex': ['female', 'male'],
        'chest_pain_type': ['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'],
        'fasting_blood_sugar': ['lower than 120mg/ml', 'greater than 120mg/ml'],
        'resting_electrocardiogram': ['normal', 'ST-T wave abnormality', 'left ventricular hypertrophy'],
        'exercise_induced_angina': ['no', 'yes'],
        'st_slope': ['upsloping', 'flat', 'downsloping'],
        'thalassemia': ['fixed defect', 'normal', 'reversable defect']
    }
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(dummy_data[col])
        encoders[col] = le
    return encoders

# Load the original data to fit the encoders properly
original_df = pd.read_csv('heart.csv')
original_df = original_df.rename(
    columns = {'cp':'chest_pain_type',
               'trestbps':'resting_blood_pressure',
               'chol': 'cholesterol',
               'fbs': 'fasting_blood_sugar',
               'restecg' : 'resting_electrocardiogram',
               'thalach':'max_heart_rate_achieved',
               'exang': 'exercise_induced_angina',
               'oldpeak': 'st_depression',
               'slope': 'st_slope',
               'ca':'num_major_vessels',
               'thal': 'thalassemia'},
    errors="raise")

original_df['sex'][original_df['sex'] == 0] = 'female'
original_df['sex'][original_df['sex'] == 1] = 'male'
original_df['chest_pain_type'][original_df['chest_pain_type'] == 0] = 'typical angina'
original_df['chest_pain_type'][original_df['chest_pain_type'] == 1] = 'atypical angina'
original_df['chest_pain_type'][original_df['chest_pain_type'] == 2] = 'non-anginal pain'
original_df['chest_pain_type'][original_df['chest_pain_type'] == 3] = 'asymptomatic'
original_df['fasting_blood_sugar'][original_df['fasting_blood_sugar'] == 0] = 'lower than 120mg/ml'
original_df['fasting_blood_sugar'][original_df['fasting_blood_sugar'] == 1] = 'greater than 120mg/ml'
original_df['resting_electrocardiogram'][original_df['resting_electrocardiogram'] == 0] = 'normal'
original_df['resting_electrocardiogram'][original_df['resting_electrocardiogram'] == 1] = 'ST-T wave abnormality'
original_df['resting_electrocardiogram'][original_df['resting_electrocardiogram'] == 2] = 'left ventricular hypertrophy'
original_df['exercise_induced_angina'][original_df['exercise_induced_angina'] == 0] = 'no'
original_df['exercise_induced_angina'][original_df['exercise_induced_angina'] == 1] = 'yes'
original_df['st_slope'][original_df['st_slope'] == 0] = 'upsloping'
original_df['st_slope'][original_df['st_slope'] == 1] = 'flat'
original_df['st_slope'][original_df['st_slope'] == 2] = 'downsloping'
original_df['thalassemia'][original_df['thalassemia'] == 1] = 'fixed defect'
original_df['thalassemia'][original_df['thalassemia'] == 2] = 'normal'
original_df['thalassemia'][original_df['thalassemia'] == 3] = 'reversable defect'

encoders = get_label_encoders(original_df)

st.title("Heart Disease Prediction")
st.write("Enter the patient's information to predict the likelihood of heart disease.")

# Input widgets
with st.form("prediction_form"):
    st.header("Patient Information")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 18, 100, 50)
        sex = st.selectbox("Sex", ['male', 'female'])
        chest_pain_type = st.selectbox("Chest Pain Type", ['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'])
        resting_blood_pressure = st.slider("Resting Blood Pressure (mm/Hg)", 90, 200, 120)

    with col2:
        cholesterol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
        fasting_blood_sugar = st.selectbox("Fasting Blood Sugar", ['lower than 120mg/ml', 'greater than 120mg/ml'])
        resting_electrocardiogram = st.selectbox("Resting Electrocardiogram", ['normal', 'ST-T wave abnormality', 'left ventricular hypertrophy'])
        max_heart_rate_achieved = st.slider("Max Heart Rate Achieved", 70, 220, 150)

    with col3:
        exercise_induced_angina = st.selectbox("Exercise Induced Angina", ['no', 'yes'])
        st_depression = st.slider("ST Depression induced by exercise relative to rest", 0.0, 6.2, 1.0, 0.1)
        st_slope = st.selectbox("Slope of the peak exercise ST segment", ['upsloping', 'flat', 'downsloping'])
        num_major_vessels = st.slider("Number of major vessels (0-3) colored by fluoroscopy", 0, 3, 0)
        thalassemia = st.selectbox("Thalassemia", ['fixed defect', 'normal', 'reversable defect'])

    submitted = st.form_submit_button("Predict")

    if submitted:
        # Create a DataFrame from inputs
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'chest_pain_type': [chest_pain_type],
            'resting_blood_pressure': [resting_blood_pressure],
            'cholesterol': [cholesterol],
            'fasting_blood_sugar': [fasting_blood_sugar],
            'resting_electrocardiogram': [resting_electrocardiogram],
            'max_heart_rate_achieved': [max_heart_rate_achieved],
            'exercise_induced_angina': [exercise_induced_angina],
            'st_depression': [st_depression],
            'st_slope': [st_slope],
            'num_major_vessels': [num_major_vessels],
            'thalassemia': [thalassemia]
        })

        # Preprocess categorical features using the fitted encoders
        categorical_cols = [
            'sex','chest_pain_type','fasting_blood_sugar',
            'resting_electrocardiogram','exercise_induced_angina',
            'st_slope','thalassemia'
        ]
        for col in categorical_cols:
            input_data[col] = encoders[col].transform(input_data[col])

        # Make prediction
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)[:, 1]
        proba_percent = prediction_proba[0]*100

        # Advanced colored report display using custom HTML and CSS
        result_html = '<div class="report-container">'
        result_html += '<div class="report-title">Heart Disease Prediction Report</div>'

        # Section: Patient Info
        result_html += '<div class="report-section-label">Patient Summary</div>'
        result_html += f'''
        <ul style="list-style-type:none;padding-left:0;margin-bottom:16px;">
          <li><b>Age:</b> {age} &nbsp; <b>Sex:</b> {sex}</li>
          <li><b>Chest Pain Type:</b> {chest_pain_type}</li>
          <li><b>Resting BP:</b> {resting_blood_pressure} mm/Hg</li>
          <li><b>Cholesterol:</b> {cholesterol} mg/dl</li>
          <li><b>Fasting Blood Sugar:</b> {fasting_blood_sugar}</li>
          <li><b>Resting ECG:</b> {resting_electrocardiogram}</li>
          <li><b>Max Heart Rate:</b> {max_heart_rate_achieved}</li>
          <li><b>Exercise Induced Angina:</b> {exercise_induced_angina}</li>
          <li><b>ST Depression:</b> {st_depression}</li>
          <li><b>Slope of ST Segment:</b> {st_slope}</li>
          <li><b>Major Vessels:</b> {num_major_vessels}</li>
          <li><b>Thalassemia:</b> {thalassemia}</li>
        </ul>
        '''
        # Section: Prediction
        result_html += '<div class="report-section-label">Prediction</div>'
        if prediction[0] == 1:
            result_html += f'<div class="result-high">HIGH likelihood of heart disease.</div>'
        else:
            result_html += f'<div class="result-low">LOW likelihood of heart disease.</div>'

        # Section: Probability meter
        result_html += '<div class="report-section-label" style="margin-top:18px;">Probability:</div>'
        if prediction[0] == 1:
            bar_color = "proba-high"
        else:
            bar_color = "proba-low"
        result_html += f'''
            <div class="proba-bar">
                <div class="{bar_color}" style="width: {proba_percent:.1f}%"></div>
            </div>
            <div style="margin-top:0.25em;font-size:1.08rem;">
                Probability of heart disease: <b>{proba_percent:.2f}%</b>
            </div>
        '''
        result_html += '</div>'

        st.markdown(result_html, unsafe_allow_html=True)

st.caption("This application is for educational purposes only and should not be used for medical diagnosis.")
