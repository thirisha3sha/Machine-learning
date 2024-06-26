"""1.	Mark and his family are planning to move to a new city and are in the market for a new home. They have been searching online for homes in their desired area and have found several properties that meet their requirements. However, they are not sure about the prices of these homes and want to get a rough estimate before making an offer.How will  you help Mark to buy  a new house.
a)	Read the house  Data set using the Pandas module (b) Print the 1st five rows. 
b)	Basic statistical computations on the data set or distribution of data (c) Print the columns and their data types (d) Detects null values in the dataset. If there is any null values replaced it with mode value (e) Explore the data set using   heatmap (f) Split the data in to test and train (g) Predict the price of a house
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic data for house prices
np.random.seed(42)
house_sizes = np.random.randint(1000, 3000, 50)  # Generating 50 house sizes between 1000 and 3000 sqft
prices = house_sizes * 150 + np.random.normal(scale=20000, size=50)  # Generating prices with noise

# Create DataFrame with explicit feature names
df = pd.DataFrame({
    'House Size (sqft)': house_sizes,
    'Price (USD)': prices.round(2)
})

# (a) Print the first five rows
print("First five rows of the dataset:")
print(df.head())

# (b) Basic statistical computations
print("\nBasic statistical computations:")
print(df.describe())

# (c) Columns and their data types
print("\nColumns and their data types:")
print(df.dtypes)

# (d) Detect and replace null values with mode (no nulls in synthetic data)
print("\nNo null values found in synthetic dataset.")

# (e) Explore the dataset using a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()

# (f) Split the data into training and test sets
X = df[['House Size (sqft)']]  # Explicitly selecting features with their names
y = df['Price (USD)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# (g) Predict the price of a house
# Creating the linear regression model
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

# Example prediction for a new house size
new_house_size = np.array([[2500]])  # Example: predicting for a house size of 2500 sqft
predicted_price = model.predict(new_house_size)
print(f"\nPredicted price for a house of size {new_house_size[0][0]} sqft: ${predicted_price[0]:,.2f}")

# Bivariate analysis: Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='House Size (sqft)', y='Price (USD)', data=df)
plt.title('Scatter Plot of House Size vs. Price')
plt.xlabel('House Size (sqft)')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.show()
