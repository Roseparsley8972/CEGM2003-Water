# Function to add all rain to the dataset

from Rainloader import read_file
import os
import pandas as pd
from scipy.spatial import cKDTree
import numpy as np


def add_rain(df):
    for file in os.listdir(os.path.join(os.path.dirname(__file__), 'rain_data')):
        file_path = os.path.join(os.path.dirname(__file__), 'rain_data', file)
        data = read_file(file_path)
        
        if file == "rainan.txt":
            period = "mean"

        else:
            period = file.split(".")[0][1:]

        # Create KDTree for fast nearest-neighbor lookup
        tree = cKDTree(data[['lat', 'lon']].values)
        
        # Find the closest location
        distances, indices = tree.query(df[['lat', 'lon']].values, k=1)
        
        # Assign the closest rain data to the dataframe
        df[period] = data.iloc[indices]['Rain mm/y'].values
        
        # df.interpolate(inplace=True)
    return df
    

if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(__file__), "..", 'Data', 'dat07_u.csv')
    df = pd.read_csv(file_path)
    df = add_rain(df)

    # print(df.columns)
    # print(df["Rain mm/y"], df["mean"])
    # print(df['Rain mm/y'] - df['mean'])
    print(np.max(df['Rain mm/y'] - df['mean']))

    diff = []

    for i, col in enumerate(['1916-1925', '1921-1930', '1926-1935',
       '1931-1940', '1936-1945', '1941-1950', '1946-1955', '1951-1960',
       '1956-1965', '1961-1970', '1966-1975', '1971-1980', '1976-1985',
       '1981-1990', '1986-1995', '1991-2000', '1996-2005', '2001-2010',
       '2006-2015', '2011-2020', 'mean']):
        
        diff.append(np.sum(np.abs(df['Rain mm/y'] - df[col]))/len(df[col]))

    print(['1916-1925', '1921-1930', '1926-1935', '1931-1940', '1936-1945', '1941-1950', '1946-1955', '1951-1960', '1956-1965', '1961-1970', '1966-1975', '1971-1980', '1976-1985', '1981-1990', '1986-1995', '1991-2000', '1996-2005', '2001-2010', '2006-2015', '2011-2020', 'mean'][np.argmin(diff)])
    print(diff[np.argmin(diff)])
    print(diff)
    





