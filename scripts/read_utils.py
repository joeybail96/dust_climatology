# import necessary libraries needed by included functions
import os
import re
import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import xarray as xr
from datetime import datetime
import re

import cartopy.crs as ccrs
import cartopy.feature as cfeature



# %% read and save raw grimm data to hourly netcdf file

# check to see if GRIMM was reset during any of the collection periods
def _fix_x00_issue(file_path, save_dir, title):
    
    # open the csv file as a text file
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # remove all the null \x00 values for all lines
    cleaned_lines = [line.replace('\x00', '') for line in lines]

    # save the cleaned content to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, mode='w', newline='') as tmpfile:
        tmpfile.writelines(cleaned_lines)
        cleaned_file_path = tmpfile.name
    
    # read the cleaned file with pandas
    df = pd.read_csv(cleaned_file_path, header=None)
    
    # remove the temporary file
    os.remove(cleaned_file_path)
    
    
    raw_time = df.iloc[:,0]
    timestamps = pd.Series(raw_time)
    timestamps = pd.to_datetime(timestamps)
    dates = timestamps.dt.strftime('%Y-%m-%d')   
    date_str = dates.iloc[1]

    
    # Create the error message text file
    error_message = (
        "Error occurred in this csv file. One of the UTC time signatures contained "
        "extraneous null values of x00\\x00..."
    )
    error_filename = f"{date_str}_{title}_error.txt"
    error_file_path = os.path.join(save_dir, error_filename)
    
    with open(error_file_path, 'w') as error_file:
        error_file.write(error_message)
    
    # return the dataframe with no extraneous null \x00 values
    return df


# preprocess & format the raw grimm data into dataframe
def _format_grimm(df, eff):
    
    logging.info('Formatting data')
    
    # truncate time to HH:MM:SS
    df[0] = df[0].str.split('.').str[0]
    
    logging.info('Truncated time to hh:mm:ss')

    # grab time (UTC)
    utc = pd.to_datetime(df.iloc[:,0],format='mixed')     
    
    # translate into local time (MST)
    #mst = utc.dt.tz_localize('UTC').dt.tz_convert('MST')

    # drop time columns from df
    df = df.iloc[:, 1:]
    
    # trim away smallest and largest bin (inaccurate measurements)
    df = df.iloc[:, 1:-1]
    
    # convert units to n/cm3
    # # originally n/100ml
    # # converting to n/ml (n/cm3)
    df = df.replace('c0', 0)
    df = df.replace('c', 0)
    df = df.astype(float) / 100
    
    logging.info('Converted to #/cm3 for all data')
    
    # 
    if eff != []:
        # adjust dust concentrations based on inlet efficiencies
        num_columns = df.shape[1]
        df = df / eff[0:num_columns]
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        logging.info('Adjusted data using efficiencies')
    
    # combine the time and data dataframes
    result_df = pd.concat([utc, df], axis=1)
    result_df.columns = ['Time_UTC'] + list(result_df.columns[1:])
    
    result_df = result_df.sort_values(by='Time_UTC').reset_index(drop=True)
    
    # return formatted grimm data
    return result_df


# read the raw csv files and format data into a dataframe
def read_grimm(data_dir, file, eff, years, months, save_dir):
        
  
    # list all subdirectories in the data_dir
    subdirectories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    # initialize an empty list to store dataframes that will contain raw grimmm data for each day
    dfs = []
    
    # initialize iterator and folder counts that will be used to keep track of loading progress
    i = 0
    total_subdir = len(subdirectories)
    
    # iterate through each folder (e.g., GRIMM_data_2020, etc)
    for subdirectory in subdirectories:
        
        # index iterator
        i = i+1
        
        # check if the subdirectory label contains a year from the input years
        if any(str(year) in subdirectory for year in years):
             
            # get a list of csv files in the year's subdirectory
            csv_files = [f for f in os.listdir(os.path.join(data_dir, subdirectory)) if f.endswith('.csv')]
            
            # grab months that user specified
            if months:
                
                # grab all csv files for specified months only
                csv_month_files = [] 
                for month in months:
                    
                    # define search criteria 
                    regex_pattern = r'^\d{4}-' + month + '-\d{2}\.csv$'
            
                    # filter out dates to only include specified months
                    csv_month_files = csv_month_files + [file for file in csv_files if re.match(regex_pattern, file)]
                
                # transfer the cumulative month files to csv_files
                csv_files = csv_month_files
                                                   
            # iterator that will track progress as csv files are opened in current subdirectory
            ii = 0
            total_csv_files = len(csv_files)
            
            # iterate through each csvfile (e.g., 2020-01-04.csv in subdirectory
            for csv_file in csv_files:
                
                # index iterator
                ii = ii+1
                
                # identify date
                file_path = os.path.join(data_dir, subdirectory, csv_file)
                
                # attempt to read csv
                try:
                    df = pd.read_csv(file_path, header=None)
                    
                    has_nan = df.iloc[:, 0].isna().any()
                    if has_nan   :    
                        df = _fix_x00_issue(file_path, save_dir, csv_file)
                    
                    dfs.append(df)
                    logging.info(f'File {ii} of {total_csv_files} & subdirectory {i} of {total_subdir}...')
                 
                # log when a file is not able to be opened
                except:
                    logging.error(f"Error reading {file_path}")
        

    # combine data from all csv files into one dataframe
    combined_df = pd.concat(dfs, axis=0, ignore_index=True)
        
    # return all formatted and combined data
    return _format_grimm(combined_df, eff)
  
    
# add column titles to the raw aerosol data stored in GRIMM dataframe
def bin(data, bins):
    
    # grab indicies of neighboring elements
    x1_indices = np.arange(0, len(bins) - 1, dtype=int)
    x2_indices = x1_indices + 1
    
    # calculate mean of neighboring bins
    mean_dp = (bins[x2_indices] + bins[x1_indices]) / 2
    
    # convert data array into dataframe
    df = pd.DataFrame(data)

    #
    column_names = df.columns.tolist()
    column_names[1:] = mean_dp
    
    #
    df.columns = column_names
    
    logging.info('Calculated and labeled mean bin diameters')

    return df


# assemble time and aerosol data into single dataframe
def combine(date, data):
    
    logging.info('Combining date and data dataframes')
    
    # combine date with data    
    result_df = pd.concat([date, data], axis=1)
        
    # add a column title over the dates
    result_df.columns = ['Time_UTC'] + list(result_df.columns[1:])
    
    logging.info('Time and data combined into single dataframe')
    
    # returned combined dataframe
    return result_df  
    
    







































#

   

# Function to read and parse the tdump file




def add_hysplit(grimm_ds, base_dir, hysplit_file, num_particles=10):
    """
    Processes HYSPLIT trajectory data from tdump files and saves all trajectories in a single NetCDF file.

    Parameters:
    grimm_ds (xarray.Dataset): Dataset containing GRIMM data.
    base_dir (str): Directory containing tdump files.
    num_particles (int): Number of particles per trajectory date (default: 10).
    hysplit_file (str): Path to the output NetCDF file.

    Returns:
    hysplit_ds (xarray.Dataset): The HYSPLIT trajectory data saved in the NetCDF file.
    """
    tdump_files = _get_tdump_files(base_dir)  # Assume this function gets the list of tdump files

    time_mst_vals = []
    latitudes = []
    longitudes = []
    altitudes = []
    particles = []

    # Process each tdump file
    for tdump_file in tdump_files:
        print(f"Processing tdump file: {tdump_file}")
        for particle_id in range(num_particles):
            date_str, trajectory_data = _read_tdump(tdump_file, particle_id)  # Assume this reads data correctly
            if trajectory_data is not None:
                trajectory_date = pd.to_datetime(date_str).tz_localize('MST')  # Correctly localize the time

                time_mst_vals.append(trajectory_date)

                latitudes.extend(trajectory_data[:, 1])
                longitudes.extend(trajectory_data[:, 0])
                altitudes.extend(trajectory_data[:, 2])
                particles.extend([particle_id] * len(trajectory_data))

    # Convert time_mst_vals to pandas datetime
    time_mst_vals = pd.to_datetime(time_mst_vals)

    # Convert time to nanoseconds since the Unix epoch
    time_mst_vals_ns = time_mst_vals.values.astype('int64')

    # Create xarray Dataset with correct time coordinate
    hysplit_ds = xr.Dataset(
        {
            "latitude": (("time", "particle"), np.zeros((len(time_mst_vals), num_particles))),
            "longitude": (("time", "particle"), np.zeros((len(time_mst_vals), num_particles))),
            "altitude": (("time", "particle"), np.zeros((len(time_mst_vals), num_particles))),
        },
        coords={
            "time": ("Time_MST", time_mst_vals_ns),  # Create 'Time_MST' as the dimension of the time coordinate
        },
        attrs={
            "description": "HYSPLIT trajectory data with particles over time",
            "source": base_dir
        }
    )

    # Set the attribute for the time coordinate to match grimm_ds
    hysplit_ds["time"].attrs["long_name"] = "Time in MST"
    
    # Fill the dataset with trajectory data directly
    for i, time in enumerate(time_mst_vals_ns):
        particle_id = particles[i]
    
        # Directly assign latitude, longitude, and altitude
        hysplit_ds["latitude"].loc[{"time": i, "particle": particle_id}] = latitudes[i]
        hysplit_ds["longitude"].loc[{"time": i, "particle": particle_id}] = longitudes[i]
        hysplit_ds["altitude"].loc[{"time": i, "particle": particle_id}] = altitudes[i]

    

    # Resample GRIMM dataset to hourly averages
    grimm_ds = grimm_ds.set_index("time")  # Set index to Time_MST for resampling
    hourly_grimm = grimm_ds.resample('H').mean()  # Resample to hourly averages

    # Ensure the index is datetime if not already
    time_index = pd.to_datetime(hourly_grimm.index)
    hourly_grimm.index = time_index

    # Align GRIMM data with HYSPLIT time coordinates
    grimm_time_aligned = hourly_grimm.sel(Time_MST=pd.to_datetime(hysplit_ds["time"].values))

    # Add the particle_count data to the HYSPLIT dataset
    hysplit_ds["particle_count"] = (("time", "size_bins"), np.zeros((len(hysplit_ds.time), len(grimm_ds.size_bins))))

    # Assuming that grimm_ds.size_bins is aligned, map the resampled particle count to hysplit_ds
    hysplit_ds["particle_count"].loc[:, :] = grimm_time_aligned["particle_count"].values

    # Save to NetCDF
    hysplit_ds.to_netcdf(hysplit_file)
    print(f"Saved NetCDF file: {hysplit_file}")
    
    return hysplit_ds
    
    
    

    
    
    
    
    
    
    
# %% plot hysplit and distribution results

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
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')
        ax.add_feature(cfeature.LAKES, edgecolor='black')

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
        ax.set_title(f"Particle Trajectories at {np.datetime_as_string(time, unit='h')}")
        ax.legend(loc="upper right", fontsize="small", markerscale=2, ncol=3)
        
        plt.show()