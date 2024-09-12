import streamlit as st
import pandas as pd
import joblib  # Use this if the model was saved with joblib
from tensorflow.keras.models import load_model  # Use this if the model is a Keras model

# Load the pre-trained model
try:
    model = load_model('heart_prediction.keras')  # Use this if the model is a Keras model
    # model = joblib.load('heart_prediction.pkl')  # Use this if the model is a joblib file
except Exception as e:
    st.error(f"Error loading model: {e}")

# Load and display accuracy
try:
    with open('accuracy.txt', 'r') as file:
        accuracy = file.read().strip()
        # Assuming the accuracy file contains a line like "Accuracy: 0.85"
        accuracy_value = float(accuracy.split(': ')[1])
except Exception as e:
    st.error(f"Error reading accuracy file: {e}")
    accuracy_value = None

st.title("Model Accuracy and Real-Time Prediction")

if accuracy_value is not None:
    st.write(f"Model Accuracy: {accuracy_value:.2f}")

# User input for real-time prediction
st.header("Real-Time Prediction")

# Load the test data
test_data = pd.read_csv('heart-disease.csv')

# Assuming the last column is the target
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

# Assume the model expects the same input features as X_test
input_data = []
for col in X_test.columns:
    input_value = st.number_input(f"Input for {col}", value=0.0)
    input_data.append(input_value)

# Convert to DataFrame for prediction
input_df = pd.DataFrame([input_data], columns=X_test.columns)

# Make prediction
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)
        st.write(f"Prediction: {prediction[0][0]:.2f}")  # Adjust based on model output
    except Exception as e:
        st.error(f"Error making prediction: {e}")

# Plot accuracy
st.header("Accuracy Plot")
if accuracy_value is not None:
    st.bar_chart([accuracy_value])
