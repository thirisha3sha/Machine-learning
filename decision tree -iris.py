#decision tree -iris
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the breast cancer dataset
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Decision Tree classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy on test set: {accuracy * 100:.2f}%')

# Example prediction with new data
new_sample = np.array([[1.5, 2.0, 3.5, 1.0, 1.2, 3.2, 2.5, 4.1, 1.8, 1.3, 
                        2.4, 3.1, 2.8, 1.9, 2.7, 3.9, 4.2, 1.7, 2.9, 1.6, 
                        3.6, 2.3, 4.5, 1.4, 2.2, 3.8, 4.0, 1.1, 2.6, 3.7]])
predicted = model.predict(new_sample)
predicted_proba = model.predict_proba(new_sample)  # Decision trees do not have predict_proba, but we can still call it
print(f'Predicted class: {predicted[0]}, Probability: {predicted_proba[0]}')
