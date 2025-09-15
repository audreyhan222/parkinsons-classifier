import streamlit as st
import joblib
import json
import pandas as pd

@st.cache_resource
def load_artifacts():
    model = joblib.load('Logistic Regression.joblib')

model = load_artifacts()


def preprocess_data(data, feature_list):
    input_data = {
        'mean_hold': data['mean_hold'],
        'std_hold':data['std_hold'], 
        'mean_latency': data['mean_latency'], 
        'std_latency': data['std_latency'], 
        'Age': data['Age'],
        'mean_flight': data['mean_flight'], 
        'std_flight': data['std_flight'], 
        'median_hold': data['median_hold'],
        'Gender_Male': data['Gender_Male'], 
        'Gender_Female': data['Gender_Female'], 
        'Sided_Left': data['Sided_Left'], 
        'Sided_Right': data['Sided_Right'], 
        'Impact': data['Impact'], 
        'keystroke_count': data['keystroke_count']
    }

    df = pd.Dataframe(input_data)
    df = df[feature_list]
    
    return df


def predict(df, model):
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0]
    return prediction, probability


st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

with st.form(key="parkinsons_form"):
    gender = st.radio(
        "Gender", ('Male', 'Female'),
        horizontal=True,
        help="Pick a gender"
    ) 
    age = st.number_input('Age', 0, 100)
    mean_hold = st.number_input('Mean Hold Time', 0.0, help='Average amount of time each key is held down.')
    std_hold = st.number_input('Standard Deviation of Hold Time', 0.0)
    mean_latency = st.number_input('Mean Latency', 0.0)
    std_latency = st.number_input('Standard Deviation of Latency', 0.0)
    mean_flight = st.number_input('Mean Flight Time', 0.0, help='Average time between key presses.')
    std_flight = st.number_input('Standard Deviation of Flight Time', 0.0)
    median_hold = st.number_input('Median Hold Time', 0.0)


    sided = st.radio(
        "Sided", ("Left", "Right"),
        horizontal=True
    )

    impact = st.radio(
        "Impact", ("Mild", "Medium", "Severe"),
        horizontal=True
    )

    keystroke_count = st.number_input('Keystroke Count', 0)

    is_pressed = st.form_submit_button(
        label="Predict if I have Parkinson's"
    )

if is_pressed:
    impact_map = {'Mild': 1.0, 'Medium': 2.0, 'Severe': 3.0}
    input_data = {
        'mean_hold': mean_hold,
        'std_hold': std_hold, 
        'mean_latency': mean_latency, 
        'std_latency': std_latency, 
        'Age': age,
        'mean_flight': mean_flight, 
        'std_flight': std_flight, 
        'median_hold': median_hold,
        'Gender_Male': 1 if gender == 'Male' else 0, 
        'Gender_Female': 1 if gender == 'Female' else 0, 
        'Sided_Left': 1 if sided == 'Left' else 0, 
        'Sided_Right': 1 if sided == 'Right' else 0, 
        'Impact': impact_map[impact], 
        'keystroke_count': keystroke_count
    }

    df = pd.DataFrame(input_data)

    prediction, probability = predict(df, model)

    prob_parkinsons = probability[1]

    if prediction == 1:
        st.error('Prediction: You might have Parkinson\'s Disease.' , icon="😢")
    else:
        st.success('Prediction: You do not have Parkinson\'s Disease.', icon='😃')

    st.progress(prob_parkinsons)
    st.info("Disclaimer: This is a machine learning prediction and not a substitute for professional medical advice.")
