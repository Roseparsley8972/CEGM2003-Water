import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def plot_map_from_csv(data_file='Australia_grid_0p05_data.csv'):
    # Set the data location
    DataLocation = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.chdir(DataLocation)

    # Load data from CSV
    data = pd.read_csv(data_file)

    # Extract latitude and longitude
    latitudes = data['lat']
    longitudes = data['lon']

    # Create a new figure with a larger size
    plt.figure(figsize=(14, 10))

    # Set up the Basemap with wider boundaries to cover all of Australia
    m = Basemap(projection='merc', 
                llcrnrlat=-43.0, urcrnrlat=-10.0, 
                llcrnrlon=113.0, urcrnrlon=153.0, 
                lat_ts=-25, resolution='i')

    # Draw country borders and other map details
    m.drawcoastlines()
    m.drawcountries()
    m.drawmapboundary(fill_color='lightblue')
    m.fillcontinents(color='lightgreen', lake_color='lightblue')
    m.drawmapscale(145, -30, 145, -30, 1000, barstyle='fancy')

    # Convert latitude and longitude to map projection coordinates
    x, y = m(longitudes.values, latitudes.values)

    # Plot the points
    m.scatter(x, y, marker='o', color='red', zorder=5)

    # Title and show the plot
    plt.title('Locations Based on Latitude and Longitude')
    plt.show()

# Example usage
if __name__ == "__main__":
    plot_map_from_csv()