import numpy as np
import os
import xarray as xr

import read_utils
#import plot_utils

import pandas as pd


from process_utils import grimm_processor
from plot_utils import grimm_plotter


grimm_processor = grimm_processor()
grimm_plotter = grimm_plotter()



# %% user defined inputs



# define the path to where grimm data is stored
grimm_dir = '/uufs/chpc.utah.edu/common/home/hallar-group2/data/site/alta'

# define the path to where the grimm.pickle is stored
grimm_file = "/uufs/chpc.utah.edu/common/home/hallar-group2/climatology/grimm/processing/grimm.nc"
#hysplit_file = "/uufs/chpc.utah.edu/common/home/hallar-group2/climatology/hysplit.nc"
#merged_file = "/uufs/chpc.utah.edu/common/home/hallar-group2/climatology/merged.nc"
grimm_5min = "/uufs/chpc.utah.edu/common/home/hallar-group2/climatology/grimm/processing/grimm_5min.nc"
grimm_10min = "/uufs/chpc.utah.edu/common/home/hallar-group2/climatology/grimm/processing/grimm_10min.nc"


# define years & months to analyze
years_to_analyze = [2018, 2019, 2020, 2021, 2022, 2023]
months_to_analyze = None   # (e.g., months_to_analyze = ['01', '02', '03'] or None)

# define inlet efficiencies for each grimm size channel
inlet_efficiencies = [0.9984480257, 0.9981661124, 0.9979658865, 0.9974219954, 
                      0.9968158814, 0.9961473774, 0.9954165081, 0.9941177784, 
                      0.9928514440, 0.9918730870, 0.9897330241, 0.9847273933, 
                      0.9754386725, 0.9640684329, 0.9457855443, 0.9181626064, 
                      0.8856016482, 0.8485313359, 0.8074187886, 0.7150943161,
                      0.5595642092, 0.4512056860, 0.3445254765, 0.1974961192, 
                      0.0234574173, 0.0000000000, 0.0000000000, 0.0000000000, 
                      0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000]

# define the different size channels of the GRIMM
grimm_bins = np.array([0.25, 0.28, 0.30, 0.35, 0.40, 0.45, 0.50, 0.58, 0.65,
              0.70, 0.80, 1.00, 1.30, 1.60, 2.00, 2.50, 3.00, 3.50, 4.00, 5.00,
              6.50, 7.50, 8.50, 10.0, 12.5, 15.0, 17.5, 20.0, 25.0, 30.0, 32.0])



# %% read grimm and hysplit data in netcdf format

# check if an existing .pickle file is already saved for grimm
if os.path.isfile(grimm_file):
    grimm_ds = xr.open_dataset(grimm_file)
    
# if file does not already exist, process and save a new netcdf file for grimm data
else:
    # load all raw grimm data for specified months
    save_dir = "/uufs/chpc.utah.edu/common/home/hallar-group2/climatology/error_collection"
    grimm = read_utils.read_grimm(grimm_dir, "", inlet_efficiencies, years_to_analyze, months_to_analyze, save_dir)
    
    # edit the grimm dataframe to have column titles that indicate particle diameters
    grimm = read_utils.bin(grimm, grimm_bins)
    
    # sort grimm data by time
    grimm.sort_values(by='Time_UTC')
    
    # convert grimm data to netcdf and save
    grimm_ds = grimm_processor.convert_grimm_to_nc(grimm, grimm_file=None)
    
    # add dust variables (raw and avg aerosol counts > 0.8um diameter)
    grimm_ds = grimm_processor.add_dust(grimm_ds, grimm_file, d_dust=0.8)
    

# read or generate average grimm files
if os.path.isfile(grimm_5min):
    grimm_5min = xr.open_dataset(grimm_5min)
else:  
    grimm_5min = grimm_processor.save_averages(grimm_ds, grimm_5min, avg_time='5T')
    
# read or generate average grimm files
if os.path.isfile(grimm_10min):
    grimm_10min = xr.open_dataset(grimm_10min)
else:  
    grimm_10min = grimm_processor.save_averages(grimm_ds, grimm_10min, avg_time='10T')
    
    

#
kslc_file = '../../synoptics/kslc/KSLC.2023-06-01.csv'
asp_file  = '../../synoptics/alta/ATH20.2023-06-01.csv'


user_input = int(input("Enter interval (5 or 10): "))

if user_input == 5:
    output_5min_nc = '../processing/grimm_clean_5min.nc'
    if os.path.isfile(output_5min_nc):
        grimm_5min_clean = xr.open_dataset(output_5min_nc)
    else:
        grimm_5min_clean = grimm_processor.find_shared_threshold_times(kslc_file, asp_file, grimm_5min, output_5min_nc)
    grimm_clean = grimm_5min_clean
    grimm_avg = grimm_5min
    
elif user_input == 10:
    output_10min_nc = '../processing/grimm_clean_10min.nc'   
    if os.path.isfile(output_10min_nc):
        grimm_10min_clean = xr.open_dataset(output_10min_nc)
    else:
        grimm_10min_clean = grimm_processor.find_shared_threshold_times(kslc_file, asp_file, grimm_10min, output_10min_nc)
    grimm_clean = grimm_10min_clean
    grimm_avg = grimm_10min
    
else:
    raise ValueError("Invalid input. Please enter 5 or 10.")


# calculate stats on the following:
    # kslc windspeeds & visibility
    # asp windspeeds
    # asp dust number concentrations
grimm_clean_stats = grimm_processor.clean_stats(grimm_clean)

clean_mean = grimm_clean_stats['dust_mean'].item()
clean_std = grimm_clean_stats['dust_std'].item()
clean_thresh = clean_mean + 1*clean_std

event_thresh = clean_mean * 100

# retrieve list of times when grimm exceeded dust event threshold
    # defining threshold by 2 orders magnitude greater than average clean dust # conc
    # https://acp.copernicus.org/articles/22/9161/2022/
grimm_events = grimm_processor.identify_events(grimm_10min, threshold=event_thresh, clean_thresh=clean_thresh)
grimm_events = grimm_processor.merge_adjacent_events(grimm_events)

# plot dust events
#grimm_plotter.plot_dust_thresholds(grimm_ds, 30)

# plot dust event days
#grimm_plotter.plot_dust_days_from_events(grimm_10min, grimm_events, user_input, threshold = event_thresh, clean_avg=clean_thresh, wind_csv_path=kslc_file, fig_dir="../figures")
#grimm_plotter.plot_dust_days_from_events(grimm_10min, grimm_events, user_input, threshold = event_thresh, clean_avg=clean_thresh, wind_csv_path=kslc_file, fig_dir=None)





similarity_df = grimm_processor.compare_event_wind_similarity(grimm_events, kslc_file)









meso_asp_file = "../../synoptics/alta/ATH20.2023-06-01.csv"
#grimm_plotter.plot_wind_alta_timeseries(meso_asp_file)










