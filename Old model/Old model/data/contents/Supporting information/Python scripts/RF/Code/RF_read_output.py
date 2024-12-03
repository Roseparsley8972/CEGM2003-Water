import pandas as pd
import matplotlib.pyplot as plt
import os

InputData = os.path.join(os.path.dirname(__file__), '..', 'InputData')
OutputFiles = os.path.join(os.path.dirname(__file__), '..', 'OutputFiles')

data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'OutputFiles', 'model_predictions_aus_8par_grp6_250trees_mf0.33_10fold_out.csv'))
data =pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'InputData', 'Australia_grid_0p05_data.csv'))


fig, ax = plt.subplots(figsize=(8, 5))

sc = ax.scatter(data['lon'], data['lat'], s=0.1, c=data["Rain mm/y"], cmap='Blues', vmax=1000)

fig.colorbar(sc, ax=ax, label='Precipition (mm/yr)')

ax.set(xlabel=r'Longitude ($\degree$E)', ylabel='Latitude ($\degree$N)', aspect='equal')

plt.show()