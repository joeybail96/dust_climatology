import pandas as pd
import xarray as xr
import numpy as np
import pytz


class grimm_processor:
    def __init__(self):
        pass   
    
  
    def convert_grimm_to_nc(self, grimm, grimm_file):
        # Ensure 'Time_MST' (rename to 'Time_UTC' for clarity) is in datetime format with UTC timezone
        grimm['Time_UTC'] = pd.to_datetime(grimm['Time_UTC'], errors='coerce', utc=True)
    
        # Round to nearest second (still timezone-aware UTC)
        grimm['Time_UTC'] = grimm['Time_UTC'].dt.round('S')

        # Explicitly localize to UTC if not already tz-aware (just for safety)
        # Usually pd.to_datetime(..., utc=True) makes it tz-aware already
        if grimm['Time_UTC'].dt.tz is None:
            grimm['Time_UTC'] = grimm['Time_UTC'].dt.tz_localize('UTC')    

        # Drop any non-particle columns except the new 'Time_UTC'
        particle_columns = grimm.columns.difference(['Time_UTC'])
    
        grimm_ds = xr.Dataset(
            data_vars=dict(
                size_dist=(["time_utc", "size"], grimm[particle_columns].values),
            ),
            coords=dict(
                time_utc=("time_utc", grimm['Time_UTC']),
                size=("size", particle_columns.astype(float)),
            ),
            attrs=dict(description="Particle size distribution data from GRIMM.")
        )
    
        if grimm_file:
            grimm_ds.to_netcdf(grimm_file, mode="w")
            print(f"NetCDF file '{grimm_file}' has been created successfully.")
    
        return grimm_ds 
    
    
    def save_averages(self, grimm_ds, grimm_file, avg_time):     
        grimm_avg = grimm_ds.resample(time_utc=avg_time).mean()
        grimm_avg.to_netcdf(grimm_file, mode='w')
        print(f"{avg_time} averaged NetCDF file '{grimm_file}' has been created successfully.")
        
        return grimm_avg
        
      
    def convert_metar_to_nc(self, metar_excel_path, matar_nc_path):
        
        metar_du_times = []
    
        sheets = pd.read_excel(metar_excel_path, sheet_name=None)
        for sheet_name, sheet_df in sheets.items():
            if 'valid' in sheet_df.columns and 'metar' in sheet_df.columns:
                sheet_df['valid'] = pd.to_datetime(sheet_df['valid'], errors='coerce')
                du_df = sheet_df[
                    sheet_df['metar'].str.contains("DU", na=False) &
                    sheet_df['valid'].dt.month.isin([3, 4, 5])
                ]
                metar_du_times.extend(du_df['valid'].dropna().tolist())
    
        # Convert to xarray Dataset
        metar_du_times = pd.to_datetime(metar_du_times)
        ds = xr.Dataset({
            'du_event_time': ('time', metar_du_times)
        })
    
        # Save to NetCDF
        ds.to_netcdf(matar_nc_path)
        
        return ds
    
    
    def add_dust(self, grimm_ds, grimm_file, d_dust=0.8):
    
        # Step 1: Create dust variable for particles > specified min dust diameter
        mask = grimm_ds['size'] > d_dust
        grimm_ds['dust'] = grimm_ds['size_dist'].sel(size=mask).sum(dim='size')
    
        # Step 2: Ensure time coordinate is datetime and sorted
        grimm_ds = grimm_ds.sortby('time_utc')
    
        # Save to a NetCDF file
        if grimm_file:
            grimm_ds.to_netcdf(grimm_file, mode="w")
            print(f"NetCDF file '{grimm_file}' has been created successfully.")
        
        return grimm_ds
    
   
    def find_shared_threshold_times(self, kslc, asp, grimm_5min, output_nc):
        
        # --- Load and clean KSLC data ---
        kslc_df = pd.read_csv(kslc, skiprows=10)
        kslc_df['Date_Time'] = pd.to_datetime(kslc_df['Date_Time'], errors='coerce', utc=True)  # UTC aware
        kslc_df = kslc_df.dropna(subset=['Date_Time'])
        kslc_df['wind_speed_set_1'] = pd.to_numeric(kslc_df['wind_speed_set_1'], errors='coerce')
        kslc_df['visibility_set_1'] = pd.to_numeric(kslc_df['visibility_set_1'], errors='coerce')
    
        # --- Load and clean ASP data ---
        asp_df = pd.read_csv(asp, skiprows=12, names=[
            'Station_ID', 'Date_Time', 'wind_speed_set_1', 'wind_direction_set_1'
        ])
        asp_df['Date_Time'] = pd.to_datetime(asp_df['Date_Time'], errors='coerce', utc=True)  # UTC aware
        asp_df = asp_df.dropna(subset=['Date_Time'])
        asp_df['wind_speed_set_1'] = pd.to_numeric(asp_df['wind_speed_set_1'], errors='coerce')
    
        # --- Thresholds ---
        wind_thresh_kslc = kslc_df['wind_speed_set_1'].quantile(0.25)
        vis_thresh_kslc = kslc_df['visibility_set_1'].quantile(0.75)
        wind_thresh_asp = asp_df['wind_speed_set_1'].quantile(0.25)
    
        # --- Filter to within thresholds ---
        kslc_within = kslc_df[
            (kslc_df['wind_speed_set_1'] <= wind_thresh_kslc) &
            (kslc_df['visibility_set_1'] >= vis_thresh_kslc)
        ].copy()
        asp_within = asp_df[
            asp_df['wind_speed_set_1'] <= wind_thresh_asp
        ].copy()
    

        # --- Find shared UTC times ---
        shared_df = pd.merge(
            kslc_within[['Date_Time', 'wind_speed_set_1', 'visibility_set_1']],
            asp_within[['Date_Time', 'wind_speed_set_1']],
            on='Date_Time',
            how='inner',
            suffixes=('_kslc', '_asp')  # rename overlapping columns to avoid collision
        )

        shared_times = pd.DatetimeIndex(shared_df['Date_Time'])

        grimm_times = pd.DatetimeIndex(grimm_5min['time_utc'].values)
        
        # grimm_times is naive, localize to UTC
        if grimm_times.tz is None:
            grimm_times = grimm_times.tz_localize('UTC')
        
        # Both are now tz-aware (UTC)
        common_times = shared_times.intersection(grimm_times)
        
        # 1. Create DataFrame indexed by common_times
        common_times_naive = common_times.tz_convert(None)
        df_common = pd.DataFrame(index=common_times_naive)
        
        # 2. Add dust values from grimm_5min
        
        dust_vals = grimm_5min['dust'].sel(time_utc=common_times_naive).values
        df_common['dust'] = dust_vals
        
        # 3. Add KSLC and ASP data by aligning on Date_Time (which is same as common_times)
        # Make sure shared_df is indexed by Date_Time for easy joining
        shared_df['Date_Time'] = shared_df['Date_Time'].dt.tz_convert(None)
        shared_df_indexed = shared_df.set_index('Date_Time')


        df_common = df_common.join(shared_df_indexed, how='left')
        
        # 4. Rename columns for clarity if needed, e.g.:
        df_common.rename(columns={
            'wind_speed_set_1_kslc': 'kslc_wind_speed',
            'visibility_set_1_kslc': 'kslc_visibility',
            'wind_speed_set_1_asp': 'asp_wind_speed'
        }, inplace=True)
        
        df_common.rename(columns={'wind_speed_set_1_kslc': 'kslc_wind_speed'}, inplace=True)
        df_common.rename(columns={'visibility_set_1': 'kslc_visibility'}, inplace=True)
        df_common.rename(columns={'wind_speed_set_1_asp': 'asp_wind_seed'}, inplace=True)
    

        
        # 5. Convert to xarray Dataset
        ds = df_common.to_xarray()
        
        # 6. Rename coordinate from 'index' to 'time_utc'
        ds = ds.rename({'index': 'time_utc'})
        
        # 7. Save to NetCDF
        ds.to_netcdf(output_nc)
        
        print(f"Saved combined data to {output_nc}")
        
        
    
        return ds
 

    def clean_stats(self, ds):
        
        stats_funcs = {
            'mean': lambda x: x.mean(dim='time_utc', skipna=True),
            'median': lambda x: x.median(dim='time_utc', skipna=True),
            'std': lambda x: x.std(dim='time_utc', skipna=True),
            'min': lambda x: x.min(dim='time_utc', skipna=True),
            'max': lambda x: x.max(dim='time_utc', skipna=True),
            'count': lambda x: x.count(dim='time_utc')
        }
        
        data_vars = {}
        for var in ds.data_vars:
            for stat_name, func in stats_funcs.items():
                data_vars[f"{var}_{stat_name}"] = func(ds[var])
        
        stats_ds = xr.Dataset(data_vars)
        return stats_ds
    
   
    def identify_events(self, grimm_ds, threshold):
        
        print('placeholder')





    
    def get_spring_dust_stats_by_metar(self, grimm_ds, matar_nc):

        # Extract DU event times from NetCDF Dataset
        metar_du_times = pd.to_datetime(matar_nc['du_event_time'].values)
        metar_du_times = metar_du_times[metar_du_times.month.isin([3, 4, 5])]  # Spring months
    
        # Convert grimm_ds['dust'] to pandas Series with time index
        dust_series = grimm_ds['dust'].to_series()
    
        # Only spring months in the dust data
        dust_series = dust_series[dust_series.index.month.isin([3, 4, 5])]
    
        # Calculate resampled averages on the fly (without storing in grimm_ds)
        dust_resampled = {
            'raw': dust_series,
            '1min': dust_series.resample('1T').mean(),
            '5min': dust_series.resample('5T').mean(),
            '10min': dust_series.resample('10T').mean(),
            '1hr': dust_series.resample('1H').mean()
        }
    
        stats_by_resolution = {}
    
        for label, series in dust_resampled.items():
            matched_values = []
    
            for du_time in metar_du_times:
                window = series[(series.index >= du_time - pd.Timedelta(minutes=10)) &
                                (series.index <= du_time + pd.Timedelta(minutes=10))]
                matched_values.extend(window.dropna().values)
    
            matched_values = np.array(matched_values)
    
            if len(matched_values) > 0:
                stats = {
                    'count': len(matched_values),
                    'mean': np.nanmean(matched_values),
                    'median': np.nanmedian(matched_values),
                    'std': np.nanstd(matched_values),
                    'min': np.nanmin(matched_values),
                    'max': np.nanmax(matched_values)
                }
            else:
                stats = {k: np.nan for k in ['count', 'mean', 'median', 'std', 'min', 'max']}
    
            stats_by_resolution[label] = stats
    
        return stats_by_resolution



