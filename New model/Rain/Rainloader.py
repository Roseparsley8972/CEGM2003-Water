import os
from matplotlib import pyplot as plt
import pandas as pd

# File used to fix the rainfall data!

def read_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # Read header information
        ncols = int(lines[0].split()[1])
        nrows = int(lines[1].split()[1])
        xllcenter = float(lines[2].split()[1])
        yllcenter = float(lines[3].split()[1])
        cellsize = float(lines[4].split()[1])
        nodata_value = float(lines[5].split()[1])
        
        results = []
        num_rows = len(lines) - 6
        
        for row_index, line in enumerate(lines[6:]):
            values = line.split()
            for col_index, value in enumerate(values):
                value = float(value)
                if value != nodata_value:
                    lon = xllcenter + col_index * cellsize
                    lat = yllcenter + (num_rows - row_index - 1) * cellsize
                    results.append((lat, lon, value))
        
        results = pd.DataFrame(results, columns=['lat', 'lon', 'Rain mm/y'])
        
        return results
