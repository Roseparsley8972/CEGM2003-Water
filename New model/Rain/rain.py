import pandas as pd
import matplotlib.pyplot as plt
import os

file_path = os.path.join(os.path.dirname(__file__), 'Australia_grid_0p05_data.csv')
data = pd.read_csv(file_path)

fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(data['lon'], data['lat'], s=0.1, c=data['Rain mm/y'], cmap='Blues', vmax=1000)
fig.colorbar(sc, ax=ax, label='Precipition (mm/yr)')
ax.set(xlabel=r'Longitude ($\degree$E)', ylabel='Latitude ($\degree$N)', aspect='equal')
plt.show()
