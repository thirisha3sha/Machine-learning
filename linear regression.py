"""3.	Develop a Python code for implementing Linear regression and show its performance
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Expanded dataset
data = {
    'House Size (sqft)': [1000, 1500, 1200, 1800, 1350, 2000, 2200, 1600, 1750, 1450, 1900, 2100, 1700, 1300, 1250, 1400, 1550, 1850, 1950, 2050],
    'Price (USD)': [200000, 300000, 240000, 350000, 280000, 400000, 420000, 320000, 360000, 290000, 370000, 410000, 340000, 260000, 250000, 275000, 310000, 355000, 385000, 395000]
}
df = pd.DataFrame(data)
print("Dataset Preview:")
print(df)
# Bivariate analysis: Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='House Size (sqft)', y='Price (USD)', data=df)
plt.title('Scatter Plot of House Size vs. Price')
plt.xlabel('House Size (sqft)')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.show()
# Preparing the data for the linear regression model
X = df[['House Size (sqft)']]
y = df['Price (USD)']
# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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
# Visualizing the regression line
plt.figure(figsize=(10, 6))
sns.scatterplot(x='House Size (sqft)', y='Price (USD)', data=df, label='Actual Data')
plt.plot(X_test, y_pred, color='red', label='Regression Line')
plt.title('House Size vs. Price with Regression Line')
plt.xlabel('House Size (sqft)')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

    
