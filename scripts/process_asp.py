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
    grimm_ds = grimm_processor.convert_grimm_to_nc(grimm, grimm_file=None)
    
    # add dust variables (raw and avg aerosol counts > 0.8um diameter)
    grimm_ds = grimm_processor.add_dust_and_averages(grimm_ds, grimm_file, d_dust=0.8)



# load observations
metar_excel_path = "../../observations/asos_2018_2023.xlsx"
metar_nc_path = "../../observations/asos_2018_2023.nc"

# convert observations to nc file for easier processing
if os.path.isfile(metar_nc_path):
    metar_nc = xr.open_dataset(metar_nc_path)
else:    
    metar_nc = grimm_processor.convert_metar_to_nc(metar_excel_path, metar_nc_path)






    
#grimm_plotter.plot_spring_dust_by_year(grimm_ds, metar_excel_path=obs_file)



#stats = grimm_processor.get_spring_dust_stats_by_metar(grimm_ds, obs_file)
#print(stats)



# take rolling average of the counts above 2.5 um
# numeric_columns = grimm.columns[grimm.columns.to_series().apply(lambda x: isinstance(x, (int, float)))]
# columns_above_2_5_and_below_10 = numeric_columns[(numeric_columns > 2.5)]
# grimm['Count_>2.5'] = grimm[columns_above_2_5_and_below_10].sum(axis=1)
# grimm['Time_MST'] = pd.to_datetime(grimm['Time_MST'])
# grimm = grimm.set_index('Time_MST')
# rolling_avg = '0.25H'
# grimm['Count_>2.5'] = grimm['Count_>2.5'].rolling(rolling_avg).mean()
# grimm = grimm.reset_index()







