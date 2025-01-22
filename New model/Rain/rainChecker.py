import pandas as pd
import matplotlib.pyplot as plt
import os
from Rainloader import read_file
import numpy as np
from scipy.spatial import cKDTree

# file_path = os.path.join(os.path.dirname(__file__), "..", "Data", 'dat07_u.csv')
# data = pd.read_csv(file_path)


file_path = os.path.join(os.path.dirname(__file__), "..", "Data", 'rain.txt')
rain = read_file(file_path)

file_path = os.path.join(os.path.dirname(__file__), "..", 'Data', 'dat07_u.csv')
df = pd.read_csv(file_path)
# Create KDTree for fast spatial search
rain_tree = cKDTree(rain[['lon', 'lat']])

# Query the closest point in rain for each point in df
distances, indices = rain_tree.query(df[['lon', 'lat']])
print(np.max(distances))

# Add the "Rain mm/y" value from the closest point in rain to df
df['test_rain'] = rain.iloc[indices]['Rain mm/y'].values

# print(data, df)

df['rain_difference'] = df['test_rain'] - df['Rain mm/y']
    # print(df[['lat', 'lon', 'average_rain', 'Rain mm/y', 'rain_difference']].head())
df["rain_difference"] = np.abs(df["rain_difference"])
print(np.sum(abs((df["rain_difference"])))/len(df["rain_difference"]))
print(np.max(df["rain_difference"]))
low_95_percentile = np.percentile(df['rain_difference'], 5)
average_low_95 = df[df['rain_difference'] <= low_95_percentile]['rain_difference'].mean()
print("Average of the 95% low of the rain_difference:", average_low_95)

fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(df['lon'], df['lat'], s=0.1, c=df['rain_difference'], cmap='viridis')#, vmin=-100, vmax=100)
fig.colorbar(sc, ax=ax, label='Precipition (mm/yr)')
ax.set(xlabel=r'Longitude ($\degree$E)', ylabel='Latitude ($\degree$N)', aspect='equal')
plt.show()
