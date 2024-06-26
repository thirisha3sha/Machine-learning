
"""3.	Develop a Python code for implementing Logistic regression and show its performance"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Create a synthetic dataset
data = {
    'TransactionAmount': np.random.normal(100, 50, 1000).tolist() + np.random.normal(1000, 200, 50).tolist(),  # Normal transactions + Fraudulent transactions
    'CustomerAge': np.random.randint(18, 70, 1050).tolist(),
    'CustomerRegion': np.random.choice(['North', 'South', 'East', 'West'], 1050).tolist(),
    'Fraud': [0] * 1000 + [1] * 50  # 1000 normal transactions, 50 fraudulent transactions
}

# Create a DataFrame
df = pd.DataFrame(data)

# One-hot encode the 'CustomerRegion' column
df = pd.get_dummies(df, columns=['CustomerRegion'], drop_first=True)

# Define features and target variable
X = df.drop('Fraud', axis=1)
y = df['Fraud']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Handle class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Train a logistic regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_res, y_train_res)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Print classification report and confusion matrix
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

