"""4.	Develop a Python code for implementing the KNN algorithm with an example.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Define the dataset
data = {
    'House Size (sqft)': [1000, 1500, 1200, 1800, 1350],
    'Price (USD)': [200000, 300000, 240000, 350000, 280000]
}
df = pd.DataFrame(data)

# Print dataset preview
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

# Prepare data for KNN
X = np.array(df[['House Size (sqft)']])
y = np.array(df['Price (USD)'])

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create KNN model
k = 3  # Number of neighbors to consider
model = KNeighborsRegressor(n_neighbors=k)

# Fit the model
model.fit(X_train, y_train)

# Predict house prices for test data
y_pred = model.predict(X_test)

# Evaluate KNN model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nModel Performance:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")

# Visualize results
plt.figure(figsize=(10, 6))
sns.scatterplot(x='House Size (sqft)', y='Price (USD)', data=df, label='Actual Data')
plt.scatter(X_test, y_pred, color='red', label='Predicted Data')
plt.title('House Size vs. Price with KNN Predictions')
plt.xlabel('House Size (sqft)')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()
