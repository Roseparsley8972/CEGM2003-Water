import os
from matplotlib import pyplot as plt
import pandas as pd

def read_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # Skip the first 6 rows
        lines = lines[6:]
    
    results = []
    middle_row = len(lines) // 2
    middle_col = len(lines[0].split()) // 2
    
    for row_index, line in enumerate(lines):
        values = line.split()
        for col_index, value in enumerate(values):
            if float(value) != 99999.90:
                lon = 112.0 - (middle_col - col_index) * 0.05
                lat = -44.5 + (middle_row - row_index) * 0.05
                results.append((lat, lon, float(value)))
    
    results = pd.DataFrame(results, columns=['lat', 'lon', 'Rain mm/y'])
    
    return results

file_path = os.path.join(os.path.dirname(__file__), 'rainan.txt')
file_path = os.path.join(os.path.dirname(__file__), 'r2011-2020.txt')
data = read_file(file_path)

fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(data['lon'], data['lat'], s=0.1, c=data['Rain mm/y'], cmap='Blues', vmax=1000)
fig.colorbar(sc, ax=ax, label='Precipition (mm/yr)')
ax.set(xlabel=r'Longitude ($\degree$E)', ylabel='Latitude ($\degree$N)', aspect='equal')
plt.show()
