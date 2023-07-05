import pandas as pd

# Specify the CSV file path
csv_file = 'wine_combined_cleaned.csv'

# Read the CSV file into a Pandas DataFrame
data = pd.read_csv(csv_file)

# Print the DataFrame
# print(data)

print(data.head())
