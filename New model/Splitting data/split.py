import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Define the path to the data folder
data_folder = os.path.join(os.path.dirname(__file__), "..", 'data')
print(data_folder)
# Read all CSV files in the data folder
data_files = ["dat07_u.csv"]
data_frames = [pd.read_csv(os.path.join(data_folder, file)) for file in data_files]

# Concatenate all data frames into one
data = pd.concat(data_frames, ignore_index=True)

# Split the data into train and test sets (80/20 split)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Save the train and test sets to new CSV files
train_data.to_csv(os.path.join(data_folder, 'train_data.csv'), index=False)
test_data.to_csv(os.path.join(data_folder, 'test_data.csv'), index=False)

print("Data has been split and saved to 'train_data.csv' and 'test_data.csv'")