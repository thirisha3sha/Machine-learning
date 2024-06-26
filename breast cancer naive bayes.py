"""1.	Can the breast cancer classification problem be solved using Naive Bayes classification
a)	print the 1st five rows. (b) Basic statistical computations on the data set or distribution of data (c) The columns and their data types
b)	Detects null values in the dataset. If there is any null values replaced it with mode value (e) Split the data in to test and train 
c)	evaluate the performance of  the model  by  evaluation metrics such as confusion matrix.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Load breast cancer dataset
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

# Convert to DataFrame for exploration (optional)
df = pd.DataFrame(data=X, columns=breast_cancer.feature_names)
df['target'] = y

# (a) Print the first five rows
print("First five rows of the dataset:")
print(df.head())

# (b) Basic statistical computations
print("\nBasic statistical computations:")
print(df.describe())

# (c) Columns and their data types
print("\nColumns and their data types:")
print(df.dtypes)

# (d) Check for null values and replace with mode
null_counts = df.isnull().sum()
if null_counts.sum() > 0:
    mode_values = df.mode().iloc[0]  # Calculate mode for each column
    df.fillna(mode_values, inplace=True)
    print("\nNull values replaced with mode values:\n", df.isnull().sum())
else:
    print("\nNo null values found in the dataset.")

# (e) Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Train a Naive Bayes classifier
model = GaussianNB()
model.fit(X_train, y_train)

# Predictions on test set
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy on test set: {accuracy:.2f}")

# Confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_mat)
