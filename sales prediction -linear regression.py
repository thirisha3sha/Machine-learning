"""1.	You are a data scientist at a retail company and your manager has asked you to create a model to predict future sales. The company has been collecting data on sales, and advertising expenditures, for the past 5 years. Your manager wants to use this information to forecast sales for the next quarter and make informed decisions about advertising and inventory.
       Your task is to build a predictive model that takes into account past sales data, and  advertising expenditures, to forecast sales for the next quarter. You decide to use linear regression to build your model because it is a simple and interpretable method for predicting a continuous outcome.
a)	print the 1st five rows. 
b)	Basic statistical computations on the data set or distribution of data
c)	the columns and their data types
d)	Explore the data using scatterplot
e)	Detects null values in the dataset. If there is any null values replaced it with mode value
f)	Split the data in to test and train 
g)	Predict the model
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Creating a DataFrame with simulated data
np.random.seed(42)

# Generate synthetic data for sales and advertising expenditures
years = np.arange(1, 21)  # Assuming 20 quarters of data
sales = np.random.normal(loc=1000, scale=200, size=20)  # Sales data
advertising = np.random.uniform(low=50, high=200, size=20)  # Advertising expenditures

# Creating the DataFrame
df = pd.DataFrame({
    'Quarter': years,
    'Sales': sales,
    'Advertising': advertising
})

# (a) Print the first five rows
print("First five rows of the dataframe:")
print(df.head())

# (b) Basic statistical computations
print("\nBasic statistical computations:")
print(df.describe())

# (c) Columns and their data types
print("\nColumns and their data types:")
print(df.dtypes)

# (d) Explore the data using scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Advertising', y='Sales', data=df)
plt.title('Scatter Plot of Advertising Expenditures vs. Sales')
plt.xlabel('Advertising Expenditures')
plt.ylabel('Sales')
plt.grid(True)
plt.show()

# (e) Detect and replace null values with mode (no nulls in synthetic data)
print("\nNo null values found in synthetic dataset.")

# (f) Split the data into training and test sets
X = df[['Advertising']]  # Features as DataFrame
y = df['Sales']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# (g) Predict the model using Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nModel Performance:")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Example prediction for future quarter (hypothetical)
future_advertising = np.array([[180]])  # Example: predicting for an advertising expenditure of 180
predicted_sales = model.predict(future_advertising)
print(f"\nPredicted sales for an advertising expenditure of ${future_advertising[0][0]}: ${predicted_sales[0]:,.2f}")

# Bivariate analysis: Scatter plot with regression line
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Advertising', y='Sales', data=df)
plt.title('Scatter Plot of Advertising Expenditures vs. Sales with Regression Line')
plt.xlabel('Advertising Expenditures')
plt.ylabel('Sales')
plt.grid(True)

# Plotting the regression line
plt.plot(X_test['Advertising'], y_pred, color='red', linewidth=2)
plt.show()
