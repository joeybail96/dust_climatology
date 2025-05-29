import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np




def plot_trajectories_for_each_time(ds):
    """
    Plots all particle trajectories for each timestamp on the same map.

    Parameters:
        ds (xarray.Dataset): The dataset containing longitude, latitude, and time_mst dimensions.

    Returns:
        None (Displays a plot for each timestamp)
    """
    times = ds.time_mst.values  # Get all timestamps

    for time in times:
        fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})

        # Add map features
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS)
        ax.add_feature(cfeature.STATES)
        ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='#D2B48C')
        ax.add_feature(cfeature.LAKES, edgecolor='black')
        
        
        # Set the zoom limits to focus on Northern Utah
        ax.set_extent([-118, -108.5, 38, 43], crs=ccrs.PlateCarree())

        # Extract longitude and latitude for the given time
        lon = ds.longitude.sel(time_mst=time).values  # Shape: (particle, location)
        lat = ds.latitude.sel(time_mst=time).values   # Shape: (particle, location)

        # Assign colors to each particle
        particles = ds.particle.values
        colors = plt.cm.viridis(np.linspace(0, 1, len(particles)))

        # Loop through particles and plot locations
        for i, particle in enumerate(particles):
            ax.plot(lon[i, :], lat[i, :], marker="o", markersize=2, linestyle="-", color=colors[i], label=f'P{particle}')
        
        # Customize plot
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        
        plt.show()

