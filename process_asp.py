import numpy as np
import os
import xarray as xr

import read_utils
import plot_utils


# %% user defined inputs



# define the path to where grimm data is stored
grimm_dir = '/uufs/chpc.utah.edu/common/home/hallar-group2/data/site/alta'

# define the path to where the grimm.pickle is stored
grimm_file = "/uufs/chpc.utah.edu/common/home/hallar-group2/climatology/grimm.nc"
hysplit_file = "/uufs/chpc.utah.edu/common/home/hallar-group2/climatology/hysplit.nc"
merged_file = "/uufs/chpc.utah.edu/common/home/hallar-group2/climatology/merged.nc"


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
    grimm.sort_values(by='Time_MST')
    
    # convert grimm data to netcdf and save
    grimm_ds = read_utils.convert_grimm_to_nc(grimm, grimm_file)
    











