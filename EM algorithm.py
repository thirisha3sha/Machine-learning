import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# a) Load the Iris dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['species'] = iris.target

# Display the first few rows of the dataset
print(data.head())

# b) Plot the data using a scatter plot "sepal_width" versus "sepal_length" and color species
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

# c) Split the data
X = data.drop('species', axis=1)
y = data['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# d) Fit the data to the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(f'Model accuracy on test set: {score * 100:.2f}%')

# e) Predict the model with new test data [5, 3, 1, 0.3]
new_sample = pd.DataFrame([[5, 3, 1, 0.3]], columns=X.columns)
predicted_species = model.predict(new_sample)
predicted_species_name = iris.target_names[predicted_species[0]]
print(f'The predicted species for the sample {new_sample.values.tolist()} is {predicted_species_name}')
