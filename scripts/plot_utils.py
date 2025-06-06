import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np


class grimm_plotter:
    def __init__(self):
        pass   
    


    def plot_spring_dust_by_year(self, grimm_ds, metar_nc, meso_path=False, wind_plot=False):
    
        # --- Grimm data processing ---
        dust_series = grimm_ds['dust'].to_series()
        dust_series = dust_series[dust_series.index.month.isin([3, 4, 5])]
        
        df = dust_series.to_frame(name='raw_dust')
        df['year'] = df.index.year
        df['month'] = df.index.month
        
        resampled = {
            'dust_1min': dust_series.resample('1T').mean(),
            'dust_5min': dust_series.resample('5T').mean(),
            'dust_10min': dust_series.resample('10T').mean(),
            'dust_1hr': dust_series.resample('1H').mean()
        }
    
        # --- METAR DU event processing ---
        metar_du_times = pd.to_datetime(metar_nc['du_event_time'].values)
        metar_du_times = metar_du_times[metar_du_times.month.isin([3, 4, 5])]
    
        if meso_path:
            # --- Wind data ---
            wind_df = pd.read_csv(meso_path, skiprows=11)
            wind_df.columns = ['Station_ID', 'Date_Time', 'wind_speed_set_1', 'wind_direction_set_1']
            wind_df = wind_df.dropna(subset=['wind_speed_set_1', 'wind_direction_set_1'])
            wind_df['Date_Time'] = pd.to_datetime(wind_df['Date_Time'], errors='coerce')
            wind_df['Date_Time'] = wind_df['Date_Time'].dt.tz_localize(None)  # <--- Fix here
            wind_df = wind_df.dropna(subset=['Date_Time'])
            wind_df['year'] = wind_df['Date_Time'].dt.year
            wind_df['month'] = wind_df['Date_Time'].dt.month
    
        plot_order = [
            ('raw_dust', {'color': 'lightgray', 'label': 'Raw Dust', 'marker': '.', 'linestyle': 'None', 'alpha': 0.3, 'zorder': 1, 'markersize': 3}),
            ('dust_1min', {'color': 'tab:blue', 'label': '1 min avg', 'linestyle': '-', 'linewidth': 1.2, 'alpha': 0.8, 'zorder': 2}),
            ('dust_5min', {'color': 'tab:orange', 'label': '5 min avg', 'linestyle': '-', 'linewidth': 1.5, 'alpha': 0.85, 'zorder': 3}),
            ('dust_10min', {'color': 'tab:green', 'label': '10 min avg', 'linestyle': '-', 'linewidth': 1.7, 'alpha': 0.9, 'zorder': 4}),
            ('dust_1hr', {'color': 'tab:red', 'label': '1 hr avg', 'linestyle': '-', 'linewidth': 2.0, 'alpha': 0.95, 'zorder': 5}),
        ]
    
        for (year, month), sub_df in df.groupby(['year', 'month']):
            fig, (ax_dust, ax_wind) = plt.subplots(nrows=2, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
            start_of_month = pd.Timestamp(year=year, month=month, day=1)
            elapsed = (sub_df.index - start_of_month).total_seconds() / 86400.0
            
            
    
            # --- Dust plot ---
            ax_dust.scatter(elapsed, sub_df['raw_dust'],
                            color=plot_order[0][1]['color'],
                            label=plot_order[0][1]['label'],
                            marker=plot_order[0][1]['marker'],
                            alpha=plot_order[0][1]['alpha'],
                            s=plot_order[0][1]['markersize'],
                            zorder=plot_order[0][1]['zorder'])
    
            for label, style in plot_order[1:]:
                series = resampled[label]
                sel = series[(series.index.year == year) & (series.index.month == month)]
                elapsed_resampled = (sel.index - start_of_month).total_seconds() / 86400.0
                ax_dust.plot(elapsed_resampled, sel.values,
                             color=style['color'],
                             linestyle=style['linestyle'],
                             linewidth=style['linewidth'],
                             alpha=style['alpha'],
                             label=style['label'],
                             zorder=style['zorder'])
    
            for du_time in metar_du_times:
                if du_time.year == year and du_time.month == month:
                    match_window = sub_df.index[(sub_df.index >= du_time - pd.Timedelta(minutes=90)) &
                                                (sub_df.index <= du_time + pd.Timedelta(minutes=90))]
                    if not match_window.empty:
                        du_elapsed = (du_time - start_of_month).total_seconds() / 86400.0
                        ax_dust.vlines(du_elapsed, 400, 500, colors='black', linestyle='--', alpha=0.7, zorder=10)
                        
            ax_dust.axhline(28.3, color='black', linestyle='-', linewidth=1.5, alpha=0.6, zorder=9)

    
            ax_dust.set_ylabel('Dust Count (log scale)')
            ax_dust.set_yscale('log')
            ax_dust.set_ylim(1, 500)
            ax_dust.set_xlim(0, 31)
            ax_dust.legend()
            ax_dust.set_title(f'Dust and Wind — {year} {month}')
            
            if wind_plot:
                # --- Wind vector subplot ---
                sub_wind_df = wind_df[(wind_df['year'] == year) & (wind_df['month'] == month)].copy()
                if not sub_wind_df.empty:
                    sub_wind_df = sub_wind_df.set_index('Date_Time')
                    resampled_wind = sub_wind_df[['wind_speed_set_1', 'wind_direction_set_1']].resample('3H').mean().dropna()
                    #resampled_wind = sub_wind_df[['wind_speed_set_1', 'wind_direction_set_1']].dropna()
        
                    theta = np.deg2rad(resampled_wind['wind_direction_set_1'])
                    u = 0.5 * resampled_wind['wind_speed_set_1'] * np.sin(theta)
                    v = 0.5 * resampled_wind['wind_speed_set_1'] * np.cos(theta)
        
                    elapsed_wind = (resampled_wind.index - start_of_month).total_seconds() / 86400.0
                    
                    wind_dir = resampled_wind['wind_direction_set_1'].values
                    dir_diff = np.abs(np.diff(wind_dir))
                    dir_diff = np.minimum(dir_diff, 360 - dir_diff)  # Handle wraparound at 360°
                    
                    # Get times where direction change exceeds 45°
                    change_mask = dir_diff > 90
                    change_times = resampled_wind.index[1:][change_mask]
                    change_elapsed = (change_times - start_of_month).total_seconds() / 86400.0
                    
                    # Plot change points as vertical lines
                    for t in change_elapsed:
                        ax_wind.axvline(t, color='purple', linestyle='--', alpha=0.6, linewidth=1.0, zorder=3)
    
        
                    # Use four distinct colors for 4 direction quadrants
                    colors_list = ['red', 'blue', 'orange', 'green']
                    bin_idx = (resampled_wind['wind_direction_set_1'] // 90).astype(int) % 4
                    colors = [colors_list[i] for i in bin_idx]
        
                    ax_wind.quiver(elapsed_wind, [0]*len(u), u, v,
                                   angles='uv', scale=1, scale_units='xy', width=0.003, color=colors)
                    ax_wind.set_ylabel('Wind Vector')
                    ax_wind.set_ylim(-1, 1)
        
                ax_wind.set_xlabel('Days since start of month')
                
            plt.tight_layout()
            plt.show()
            
            
    def plot_wind_alta_timeseries(self, meso_path, scale_factor=0.75):

        # Read CSV and skip metadata lines
        df = pd.read_csv(meso_path, skiprows=11)
        df.columns = ['Station_ID', 'Date_Time', 'wind_speed_set_1', 'wind_direction_set_1']
    
        # Clean and prepare data
        df = df.dropna(subset=['wind_speed_set_1', 'wind_direction_set_1'])
        df['Date_Time'] = pd.to_datetime(df['Date_Time'], errors='coerce')
        df = df.dropna(subset=['Date_Time'])
    
        # Filter for March, April, and May
        df = df[df['Date_Time'].dt.month.isin([3, 4, 5])]
    
        # Extract year and month for grouping
        df['Year'] = df['Date_Time'].dt.year
        df['Month'] = df['Date_Time'].dt.month
        month_names = {3: "March", 4: "April", 5: "May"}
    
        # Group by year and month
        grouped = df.groupby(['Year', 'Month'])
    
        for (year, month), group in grouped:
            if group.empty:
                continue
    
            # Set datetime index
            group = group.set_index('Date_Time')
    
            # Only keep numeric columns before resampling
            numeric_group = group[['wind_speed_set_1', 'wind_direction_set_1']]
    
            # Downsample to 2-hour intervals
            resampled = numeric_group.resample('2H').mean().dropna()
    
            if resampled.empty:
                continue
    
            # Compute wind vector components
            theta = np.deg2rad(resampled['wind_direction_set_1'])
            u = scale_factor * resampled['wind_speed_set_1'] * np.sin(theta)
            v = scale_factor * resampled['wind_speed_set_1'] * np.cos(theta)
    
            # Define your custom 4 distinct colors in order
            four_colors = ['red', 'blue', 'orange', 'green']
            
            # Bin wind direction into 4 quadrants
            direction_deg = resampled['wind_direction_set_1'] % 360
            bin_indices = (direction_deg // 90).astype(int)
            
            # Assign colors accordingly
            colors = [four_colors[i] for i in bin_indices]
            
            # Plot with these discrete colors
            fig, ax = plt.subplots(figsize=(15, 4))
            ax.quiver(resampled.index, [0]*len(resampled), u, v,
                      angles='uv', scale=1, scale_units='xy', width=0.002, color=colors)

    
            ax.set_xlabel('Time')
            ax.set_ylabel('Wind Vector')
            ax.set_title(f'Wind Speed and Direction — {month_names[month]} {year} (2-hr Avg, colored by direction)')
            ax.set_ylim(-1, 1)
            ax.grid(True)
            plt.tight_layout()
            plt.show()       


    def plot_dust_thresholds(self, grimm_ds, threshold):

        # Convert xarray time coordinate to pandas DatetimeIndex for easy grouping
        time_index = pd.to_datetime(grimm_ds['time_utc'].values)
        
        # Extract year and month for grouping
        years = time_index.year
        months = time_index.month
        
        # Create a DataFrame to facilitate grouping and plotting
        df = pd.DataFrame({
            'dust': grimm_ds['dust'].values,
            'time': time_index,
            'year': years,
            'month': months
        })
        
        # Group by year and month
        grouped = df.groupby(['year', 'month'])
        
        for (year, month), group in grouped:
            if month not in [3, 4, 5]:
                continue  # Skip months other than March, April, May
        
            fig, ax = plt.subplots(figsize=(12, 6))
        
            # Below or equal threshold — black markers
            below_thresh = group['dust'] <= threshold
            ax.plot(group.loc[below_thresh, 'time'], group.loc[below_thresh, 'dust'],
                    'k.', markersize=4, label=f'<= {threshold}')
        
            # Above threshold — red markers
            above_thresh = group['dust'] > threshold
            ax.plot(group.loc[above_thresh, 'time'], group.loc[above_thresh, 'dust'],
                    'r.', markersize=4, label=f'> {threshold}')
        
            ax.set_title(f'Dust Concentration for {year}-{month:02d}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Dust Concentration')
        
            ax.set_yscale('log')
            ax.set_ylim(1, 500)
            # For time x-axis, set limits to start and end of month
            ax.set_xlim(group['time'].min(), group['time'].max())
        
            ax.grid(True)
            ax.legend()
            fig.tight_layout()
            plt.show()
            
            
    def plot_dust_days_from_events(self, grimm_ds, grimm_events, threshold):

        # Ensure datetime format
        full_time = pd.to_datetime(grimm_ds['time_utc'].values)
        full_dust = grimm_ds['dust'].values
    
        # Build full DataFrame from grimm_ds
        df = pd.DataFrame({
            'time': full_time,
            'dust': full_dust
        })
    
        # Extract unique dates from grimm_events
        grimm_events['time'] = pd.to_datetime(grimm_events['time'])  # just in case
        unique_days = grimm_events['time'].dt.date.unique()
    
        for day in sorted(unique_days):
            # Mask to all data from the same day in df
            mask = df['time'].dt.date == day
            day_data = df.loc[mask]
    
            if day_data.empty:
                continue  # Skip if no data found (unlikely)
    
            fig, ax = plt.subplots(figsize=(12, 6))
    
            # Below or equal threshold — black
            below_thresh = day_data['dust'] <= threshold
            ax.plot(day_data.loc[below_thresh, 'time'], day_data.loc[below_thresh, 'dust'],
                    'k.', markersize=4, label=f'<= {threshold}')
    
            # Above threshold — red
            above_thresh = day_data['dust'] > threshold
            ax.plot(day_data.loc[above_thresh, 'time'], day_data.loc[above_thresh, 'dust'],
                    'r.', markersize=4, label=f'> {threshold}')
    
            ax.set_title(f'Dust Concentration on {day}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Dust Concentration')
            ax.set_yscale('log')
            ax.set_ylim(1, 500)
            ax.set_xlim(day_data['time'].min(), day_data['time'].max())
            ax.grid(True)
            ax.legend()
            fig.tight_layout()
            plt.show()