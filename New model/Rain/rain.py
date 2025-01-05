import pandas as pd
import matplotlib.pyplot as plt
import os
from Rainloader import read_file
from scipy.spatial import cKDTree


file_path = os.path.join(os.path.dirname(__file__), "..", "Data", 'rain.txt')
rain = read_file(file_path)

file_path = os.path.join(os.path.dirname(__file__), "..", "Data", 'Australia_grid_0p05_data.csv')
data = pd.read_csv(file_path)

# Create KDTree for fast nearest-neighbor lookup
rain_tree = cKDTree(rain[['lon', 'lat']].values)

# Find the closest rain data point for each data point
distances, indices = rain_tree.query(data[['lon', 'lat']].values)

# Add the closest rain data to the data DataFrame
data['Rain mm/y'] = rain.iloc[indices]['Rain mm/y'].values


fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(data['lon'], data['lat'], s=0.1, c=data['Rain mm/y'], cmap='Blues', vmax=1000)
fig.colorbar(sc, ax=ax, label='Precipition (mm/yr)')
ax.set(xlabel=r'Longitude ($\degree$E)', ylabel='Latitude ($\degree$N)', aspect='equal')
plt.show()

# Save the data with the rain data added
file_path = os.path.join(os.path.dirname(__file__), "..", "Data", 'Australia_grid_0p05_data_with_rain.csv')
data.to_csv(file_path, index=False)
