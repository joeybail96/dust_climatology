import pandas as pd
import xarray as xr


class grimm_processor:
    def __init__(self):
        pass   
    
    
    def convert_grimm_to_nc(self, grimm, grimm_file):
        # Ensure 'Time_MST' is in datetime format
        grimm['Time_MST'] = pd.to_datetime(grimm['Time_MST'])


        particle_columns = grimm.columns.drop('Time_MST')
        
        # Create the Dataset
        grimm_ds = xr.Dataset(
            data_vars=dict(
                size_dist=(["time_mst", "size"], grimm[particle_columns].values),
            ),
            coords=dict(
                time_mst=("time_mst", grimm['Time_MST'].dt.tz_convert(None).astype('datetime64[s]')),
                size=("size", particle_columns.astype(float)),   # Convert column names to float
            ),
            attrs=dict(description="Particle size distribution data from GRIMM.")
        )

        # particle_size = 0.29
        # particle_data = grimm_ds.particle_dist.sel(particles=particle_size, method='nearest').values
        # time_data = grimm_ds.time_mst.values
        # plt.figure(figsize=(10, 6))
        # plt.plot(pd.to_datetime(time_data), particle_data, marker='o', linestyle='-', markersize=3, linewidth=0.5)
        # plt.xlabel('Time (MST)')
        # plt.ylabel(f'Particle Count for {particle_size} µm')
        # plt.title(f'Particle Size {particle_size} µm vs Time')
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()
        

        # Save to a NetCDF file
        grimm_ds.to_netcdf(grimm_file, mode="w")
        print(f"NetCDF file '{grimm_file}' has been created successfully.")

        return grimm_ds