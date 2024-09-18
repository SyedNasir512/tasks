import streamlit as st
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras.models import load_model  # Import for loading Keras models
import numpy as np

# Correct model loading for Keras models
model = load_model('heart_prediction.keras')  # Use Keras load_model instead of joblib

# Assuming you have a precomputed accuracy value (for demonstration, let's set it manually)
accuracy = 0.85  # Replace with actual accuracy if available

st.title("Model Accuracy and Real-Time Prediction")
st.write(f"Model Accuracy: {accuracy}")

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
    input_value = st.number_input(f"Input for {col}", value=0.0)  # Adjust default value
    input_data.append(input_value)

# Convert to DataFrame for prediction
input_df = pd.DataFrame([input_data], columns=X_test.columns)

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_df)
    predicted_class = np.argmax(prediction, axis=1)  # Use if the model has softmax output
    st.write(f"Prediction: {predicted_class[0]}")  # Display predicted class

# Plot accuracy as a bar chart
st.header("Accuracy Plot")
st.bar_chart([accuracy])  # Plot accuracy


