import os
from matplotlib import pyplot as plt
import pandas as pd

# File used to fix the rainfall data!

def read_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # Skip the first 6 rows
        lines = lines[6:]
    
    results = []
    num_rows = len(lines)
    num_cols = len(lines[0].split())
    
    for row_index, line in enumerate(lines):
        values = line.split()
        for col_index, value in enumerate(values):
            if float(value) != 99999.90:
                lon = 112.0 + col_index * 0.05
                lat = -44.5 + (num_rows - row_index - 1) * 0.05
                results.append((lat, lon, float(value)))
    
    results = pd.DataFrame(results, columns=['lat', 'lon', 'Rain mm/y'])
    
    return results

file_path = os.path.join(os.path.dirname(__file__), 'data', 'rainan.txt')
data = read_file(file_path)

file_path = os.path.join(os.path.dirname(__file__), 'data', 'Australia_grid_0p05_data.csv')
old_data = pd.read_csv(file_path)

# Merge data based on lat and lon
merged_data = pd.merge(old_data, data, on=['lat', 'lon'], how='left', suffixes=('', '_new'))
old_data['Rain mm/y'] = merged_data['Rain mm/y_new']
old_data.interpolate(inplace=True)

old_data.to_csv(file_path, index=False)
print("Data updated successfully")




