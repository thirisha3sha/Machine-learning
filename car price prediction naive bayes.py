"""1.	Jack is a car enthusiast and wants to buy a new car. He wants to find the best deal and decides to use machine learning to predict the prices of different car models.Jack collects data on various features such as the make, model, year, engine size, and number of doors, as well as the sale price of each car. He splits the data into a training set and a test set and trains a linear regression model on the training data.Car Price Prediction with Machine Learning
a)	Read the dataframe using the Pandas module 
b)	print the 1st five rows. 
c)	Basic statistical computations on the data set or distribution of data
d)	the columns and their data types
e)	Detects null values in the dataset. If there is any null values replaced it with mode value
f)	Explore the data set using   heatmap
g)	Split the data in to test and train 
h)	Fit in to the model Naive Bayes Classifier
i)	Predict the model
j)	Find the accuracy of the model
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

# Sample dataset (replace with your actual dataset)
data = {
    'Make': ['Toyota', 'Honda', 'Ford', 'Toyota', 'Honda'],
    'Model': ['Camry', 'Civic', 'F-150', 'Corolla', 'Accord'],
    'Year': [2018, 2019, 2017, 2018, 2019],
    'EngineSize': [2.5, 1.8, 3.5, 1.8, 2.0],
    'NumDoors': [4, 4, 2, 4, 4],
    'Price': [28000, 22000, 35000, 25000, 27000]
}

# Create DataFrame
df = pd.DataFrame(data)

# a) Read the dataframe using the Pandas module
# b) Print the first five rows
print("First five rows of the dataset:")
print(df.head())

# c) Basic statistical computations
print("\nBasic statistical computations:")
print(df.describe())

# d) Columns and their data types
print("\nColumns and their data types:")
print(df.dtypes)

# e) Detect null values and replace with mode value (assuming no nulls in this example)
print("\nNull values in the dataset:")
print(df.isnull().sum())

# f) Explore the dataset using heatmap (excluding non-numeric columns)
plt.figure(figsize=(10, 6))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# g) Split the data into features (X) and target (y)
X = df[['Year', 'EngineSize', 'NumDoors']]
y = df['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# h) Initialize Naive Bayes Classifier (Gaussian Naive Bayes)
nb_classifier = GaussianNB()

# Fit the model on the training data
nb_classifier.fit(X_train, y_train)

# i) Predictions on the test data
y_pred = nb_classifier.predict(X_test)

# j) Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Performance:")
print(f"Accuracy: {accuracy:.2f}")

# Calculate precision and recall (suppressing warnings for now)
with np.errstate(divide='ignore', invalid='ignore'):
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# Additional output for clarity
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
