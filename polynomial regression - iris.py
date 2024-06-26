#polynomial regression -iris
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load the Iris dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['species'] = iris.target

# Display the first few rows of the dataset
print(data.head())

# 2. Plot the data using a scatter plot "sepal_width" versus "sepal_length" and color species
species = iris.target_names
plt.figure(figsize=(10, 6))
for i, species_name in enumerate(species):
    species_data = data[data['species'] == i]
    plt.scatter(species_data['sepal length (cm)'], species_data['sepal width (cm)'], label=species_name)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend()
plt.title('Sepal Width vs Sepal Length')
plt.show()

# 3. Split the data
X = data[['sepal length (cm)']]
y = data['sepal width (cm)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Transform features for Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# 5. Train the Polynomial Regression model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# 6. Make predictions on the test set
y_pred = model.predict(X_test_poly)

# 7. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Model performance on test set:')
print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')

# Example prediction with new test data [5, 3, 1, 0.3]
new_sample = pd.DataFrame([[5]], columns=['sepal length (cm)'])  # Use sepal length for prediction
new_sample_poly = poly.transform(new_sample)
predicted_sepal_width = model.predict(new_sample_poly)
print(f'Predicted sepal width for the sample {new_sample.values.tolist()}: {predicted_sepal_width[0]:.2f}')
