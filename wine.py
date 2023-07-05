import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Specify the CSV file path
csv_file = 'wine_combined_cleaned.csv'

# Read the CSV file into a Pandas DataFrame
data = pd.read_csv(csv_file)

# The first 12 columns are the features (X) and the last column is the target (y)
X = data.iloc[:, :12].values
y = data.iloc[:, -1].values

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the decision tree classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Making predictions on the test set
predictions = model.predict(X_test)

# Evaluating the model
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy}")
