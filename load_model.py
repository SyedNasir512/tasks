import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

model = joblib.load('heart_prediction.keras') 

# Load the test data
test_data = pd.read_csv('heart-disease.csv')

# Assuming the last column is the target
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

# Generate test predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Save accuracy to a file
with open('accuracy.txt', 'w') as file:
    file.write(f'Accuracy: {accuracy}')

print(f'Accuracy: {accuracy}')