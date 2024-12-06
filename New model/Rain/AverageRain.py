import os
import pandas as pd
from scipy.spatial import cKDTree
import numpy as np
from Rainloader import read_file
import matplotlib.pyplot as plt

def add_average_rain(df):
    rain_data_path = os.path.join(os.path.dirname(__file__), 'rain_data')
    rain_files = [file for file in os.listdir(rain_data_path) if file != "rainan.txt"]
    # rain_files = [file for file in rain_files if int(file[1:5]) % 10 == 1]
    # print(rain_files)
    
    all_rain_data = []

    for file in rain_files:
        file_path = os.path.join(rain_data_path, file)
        data = read_file(file_path)
        all_rain_data.append(data)

    # Concatenate all rain data
    # print(all_rain_data)

    # Merge all dataframes on 'lat' and 'lon' and take the average of 'Rain mm/y'
    merged_rain_data = all_rain_data[0]
    for data in all_rain_data[1:]:
        merged_rain_data = pd.merge(merged_rain_data, data, on=['lat', 'lon'], suffixes=('', '_drop'))
        merged_rain_data.drop([col for col in merged_rain_data.columns if 'drop' in col], axis=1, inplace=True)
    
    merged_rain_data['Rain mm/y'] = merged_rain_data.filter(like='Rain mm/y').mean(axis=1)
    all_rain_data = merged_rain_data[['lat', 'lon', 'Rain mm/y']]
    
    # Create KDTree for fast nearest-neighbor lookup
    tree = cKDTree(all_rain_data[['lat', 'lon']].values)
    
    # Find the closest location
    distances, indices = tree.query(df[['lat', 'lon']].values, k=1)
    
    # Assign the closest rain data to the dataframe
    df['average_rain'] = all_rain_data.iloc[indices]['Rain mm/y'].values
    
    return df

if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(__file__), "..", 'Data', 'dat07_u.csv')
    df = pd.read_csv(file_path)
    df = add_average_rain(df)
    # print(df.head())
    df['rain_difference'] = df['average_rain'] - df['Rain mm/y']
    # print(df[['lat', 'lon', 'average_rain', 'Rain mm/y', 'rain_difference']].head())
    print(np.sum(abs((df["rain_difference"])))/len(df["rain_difference"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(df['lon'], df['lat'], s=0.1, c=df['rain_difference'], cmap='viridis', vmin=-100, vmax=100)
    fig.colorbar(sc, ax=ax, label='Precipition (mm/yr)')
    ax.set(xlabel=r'Longitude ($\degree$E)', ylabel='Latitude ($\degree$N)', aspect='equal')
    plt.show()

