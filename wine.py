import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

# Specify the CSV file path
csv_file = 'wine_combined_cleaned.csv'

# Read the CSV file into a Pandas DataFrame
data = pd.read_csv(csv_file)

# The first 12 columns are the features (X) and the last column is the target (y)
X = data.iloc[:, :12].values
y = data.iloc[:, -1].values

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Creating and training the decision tree classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Making predictions on the test set
predictions = model.predict(X_test)

# Evaluating the model
accuracy = np.mean(predictions == y_test)
print(f"Decision tree Accuracy (Stratified): {accuracy}")

# Creating and training the random forest classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Making predictions on the test set
predictions = model.predict(X_test)

# Evaluating the model
accuracy = np.mean(predictions == y_test)
print(f"Random Forest Accuracy (Stratified): {accuracy}")

''' Ensemble Learning '''

# Splitting the data into training, validation, and testing sets with stratified sampling
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Creating and training the Decision Tree classifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)

# Creating and training the Random Forest classifier
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)

# One-hot encoding the target variable
encoder = OneHotEncoder(sparse=False)
y_val_encoded = encoder.fit_transform(y_val.reshape(-1, 1))

# Making predictions on the validation set for each model
decision_tree_predictions = decision_tree.predict(X_val)
random_forest_predictions = random_forest.predict(X_val)

# One-hot encoding the predictions
decision_tree_predictions_encoded = encoder.transform(decision_tree_predictions.reshape(-1, 1))
random_forest_predictions_encoded = encoder.transform(random_forest_predictions.reshape(-1, 1))

# Evaluating the accuracies of the individual models on the validation set
decision_tree_accuracy = np.mean(decision_tree_predictions == y_val)
random_forest_accuracy = np.mean(random_forest_predictions == y_val)

# Calculating the weights based on the accuracies
weights = [decision_tree_accuracy, random_forest_accuracy]
weights = [weight / sum(weights) for weight in weights]

# Making predictions on the test set for each model
decision_tree_test_predictions = decision_tree.predict(X_test)
random_forest_test_predictions = random_forest.predict(X_test)

# One-hot encoding the test set predictions
decision_tree_test_predictions_encoded = encoder.transform(decision_tree_test_predictions.reshape(-1, 1))
random_forest_test_predictions_encoded = encoder.transform(random_forest_test_predictions.reshape(-1, 1))

# Combining predictions using weighted voting
ensemble_predictions_encoded = (decision_tree_test_predictions_encoded * weights[0]) + (random_forest_test_predictions_encoded * weights[1])

# Converting the ensemble predictions back to the original classes
ensemble_predictions = encoder.inverse_transform(ensemble_predictions_encoded)

# Reshaping the predictions to match the original shape
ensemble_predictions = ensemble_predictions.reshape(-1)

# Evaluating the ensemble model
accuracy = np.mean(ensemble_predictions == y_test)
print(f"Weighted Ensemble Accuracy: {accuracy}")