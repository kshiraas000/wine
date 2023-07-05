import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import tensorflow as tf
import matplotlib.pyplot as plt

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

# Create individual classifiers
decision_tree = DecisionTreeClassifier()
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

# Create the VotingClassifier with majority voting
voting_classifier = VotingClassifier(
    estimators=[('dt', decision_tree), ('rf', random_forest)],
    voting='hard'
)

# Fit the VotingClassifier on the training data
voting_classifier.fit(X_train, y_train)

# Make predictions on the test set using the VotingClassifier
ensemble_predictions = voting_classifier.predict(X_test)

# Evaluate the ensemble model
accuracy = accuracy_score(y_test, ensemble_predictions)
print(f"Voting Ensemble Accuracy: {accuracy}")

''' Deep Learning '''

# # Encode the target variable
# encoder = LabelEncoder()
# y = encoder.fit_transform(y)

# # Splitting the data into training and testing sets with stratified sampling
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# # Define the model architecture
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(256, activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(256, activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Train the model
# model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)

# # Evaluate the model
# _, accuracy = model.evaluate(X_test, y_test, verbose=0)
# print(f"Neural Net Accuracy: {accuracy}")

# Create the VotingClassifier with majority voting
voting_classifier = VotingClassifier(
    estimators=[('dt', decision_tree), ('rf', random_forest)],
    voting='hard'
)

# Define the parameter grid for grid search
param_grid = {
    'dt__max_depth': [2, 4, 6, 8, 10],
    'dt__min_samples_leaf': [1, 2, 3, 4, 5]
}

# Perform grid search
grid_search = GridSearchCV(voting_classifier, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best estimator from the grid search
best_estimator = grid_search.best_estimator_

# Get the grid search results
results = grid_search.cv_results_
depths = param_grid['dt__max_depth']
leaf_sizes = param_grid['dt__min_samples_leaf']
mean_scores = results['mean_test_score']

# Plot the accuracy based on different combinations of depth and leaf size
fig, ax = plt.subplots()
scores = np.array(mean_scores).reshape(len(depths), len(leaf_sizes))

for i, depth in enumerate(depths):
    ax.plot(leaf_sizes, scores[i], marker='o', label=f"Depth {depth}")

ax.set_xlabel('Min Samples Leaf')
ax.set_ylabel('Accuracy')
ax.legend(title='Max Depth')
ax.set_title('Accuracy vs. Depth and Leaf Size')
plt.grid(True)
plt.show()

# Make predictions on the test set using the best estimator
ensemble_predictions = best_estimator.predict(X_test)

# Evaluate the ensemble model
accuracy = accuracy_score(y_test, ensemble_predictions)
print(f"Accuracy: {accuracy}")



#test commit