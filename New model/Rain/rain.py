import pandas as pd
import matplotlib.pyplot as plt
import os
from Rainloader import read_file

# file_path = os.path.join(os.path.dirname(__file__), "..", "Data", 'dat07_u.csv')
# data = pd.read_csv(file_path)


file_path = os.path.join(os.path.dirname(__file__), "..", "Data", 'rain.txt')
# file_path = os.path.join(os.path.dirname(__file__), "..", "Data", 'r2011-2020.txt')
data = read_file(file_path)

fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(data['lon'], data['lat'], s=0.1, c=data['Rain mm/y'], cmap='Blues', vmax=1000)
fig.colorbar(sc, ax=ax, label='Precipition (mm/yr)')
ax.set(xlabel=r'Longitude ($\degree$E)', ylabel='Latitude ($\degree$N)', aspect='equal')
plt.show()
