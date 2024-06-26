import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# Sample dataset (replace with your actual dataset)
data = {
    'Occupation': ['Engineer', 'Doctor', 'Teacher', 'Doctor', 'Engineer', 'Teacher'],
    'Income': [60000, 80000, np.nan, 90000, 70000, 50000],
    'CreditScore': [680, 720, 640, 750, 690, 660]
}

# Create DataFrame
df = pd.DataFrame(data)

# a) Print the first five rows
print("First five rows of the dataset:")
print(df.head())

# b) Basic statistical computations
print("\nBasic statistical computations:")
print(df.describe())

# c) Columns and their data types
print("\nColumns and their data types:")
print(df.dtypes)

# d) Detect null values and replace with mode value
print("\nNull values in the dataset:")
print(df.isnull().sum())

# Replace null values in 'Income' column with mode
imputer = SimpleImputer(strategy='most_frequent')
df['Income'] = imputer.fit_transform(df[['Income']])

# Verify nulls after imputation
print("\nNull values after imputation:")
print(df.isnull().sum())

# e) Explore the dataset using box plot (Credit Scores Based on Occupation)
plt.figure(figsize=(8, 6))
sns.boxplot(x='Occupation', y='CreditScore', data=df)
plt.title('Credit Scores Based on Occupation')
plt.show()

# f) Split the data into features (X) and target (y)
X = df[['Income']]
y = df['CreditScore']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# g) Initialize Naive Bayes Classifier (Gaussian Naive Bayes)
nb_classifier = GaussianNB()

# Fit the model on the training data
nb_classifier.fit(X_train, y_train)

# i) Predictions on the test data
y_pred = nb_classifier.predict(X_test)

# Print predictions
print("\nPredictions on test data:")
print(y_pred)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")
