#logitic regression iris
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the breast cancer dataset
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the logistic regression model with increased max_iter
model = LogisticRegression(max_iter=1000)  # Increase max_iter
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy on test set: {accuracy * 100:.2f}%')

# Example prediction with new data
new_sample = np.array([[1.5, 2.0, 3.5, 1.0, 1.2, 3.2, 2.5, 4.1, 1.8, 1.3, 
                        2.4, 3.1, 2.8, 1.9, 2.7, 3.9, 4.2, 1.7, 2.9, 1.6, 
                        3.6, 2.3, 4.5, 1.4, 2.2, 3.8, 4.0, 1.1, 2.6, 3.7]])
new_sample_scaled = scaler.transform(new_sample)
predicted = model.predict(new_sample_scaled)
predicted_proba = model.predict_proba(new_sample_scaled)
print(f'Predicted class: {predicted[0]}, Probability: {predicted_proba[0]}')
