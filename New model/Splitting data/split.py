import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Define the path to the data folder
# Replace with the absolute path or ensure the data is in the current working directory
data_folder = os.path.abspath(os.path.join("..", "data"))
data_file = os.path.join(data_folder, "dat07_u.csv")

# Check if the file exists
if not os.path.exists(data_file):
    raise FileNotFoundError(f"The file '{data_file}' was not found.")

# Read the specific CSV file
data = pd.read_csv(data_file, low_memory=False)

# Check for missing data and remove rows with NaN values
initial_rows = len(data)
data = data.dropna()  # Removes rows with any missing values
removed_rows = initial_rows - len(data)

# Split the data into training (80%) and temporary (20%) sets
train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)

# Further split the temporary set into validation (10%) and test (10%) sets
validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Save the train, validation, and test sets to new CSV files
train_data.to_csv(os.path.join(data_folder, 'train_data.csv'), index=False)
validation_data.to_csv(os.path.join(data_folder, 'validation_data.csv'), index=False)
test_data.to_csv(os.path.join(data_folder, 'test_data.csv'), index=False)

print("Data has been split into training (80%), validation (10%), and test (10%) sets.")
print("Files saved as 'train_data.csv', 'validation_data.csv', and 'test_data.csv'.")