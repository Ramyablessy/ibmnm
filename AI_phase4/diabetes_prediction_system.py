# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the diabetes dataset
diabetes_data = pd.read_csv('diabetes.csv')  # Replace 'diabetes.csv' with the actual file path

# Display the first few rows of the dataset to inspect the data
print(diabetes_data.head())

# Data Preprocessing

# Split the data into features (X) and the target (y)
X = diabetes_data.drop('Outcome', axis=1)
y = diabetes_data['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (mean=0, variance=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Now, you have X_train, y_train, X_test, and y_test ready for training and testing your diabetes prediction model.

# Example usage (a simple classifier using Logistic Regression):
from sklearn.linear_model import LogisticRegression

# Initialize the model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of the model: {accuracy}')
