import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
import matplotlib.dates as mdates
import os


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
            


    def add_threshold_colored_line(self, ax, times, values, clean_avg, threshold, linewidth=1.5):
        def classify(val):
            if val < clean_avg:
                return 'low'
            elif val > threshold:
                return 'high'
            else:
                return 'mid'
    
        def get_color(class_label):
            return {
                'low': '#7fcdbb',   # blue-green
                'mid': 'black',
                'high': '#d73027'   # red
            }[class_label]
    
        times = mdates.date2num(times)
        segments = []
        colors = []
    
        for i in range(len(times) - 1):
            t0, t1 = times[i], times[i + 1]
            d0, d1 = values[i], values[i + 1]
    
            points = [(t0, d0), (t1, d1)]
    
            # Identify crossings and insert interpolated points
            breaks = []
            for level in [clean_avg, threshold]:
                if (d0 - level) * (d1 - level) < 0:  # crossing occurs
                    frac = (level - d0) / (d1 - d0)
                    tc = t0 + frac * (t1 - t0)
                    breaks.append((tc, level))
    
            # Sort all breakpoints and split segment accordingly
            all_points = [points[0]] + sorted(breaks, key=lambda x: x[0]) + [points[1]]
    
            for j in range(len(all_points) - 1):
                (x0, y0), (x1, y1) = all_points[j], all_points[j + 1]
                class_label = classify((y0 + y1) / 2)
                color = get_color(class_label)
                segments.append([[x0, y0], [x1, y1]])
                colors.append(color)
    
        lc = LineCollection(segments, colors=colors, linewidths=linewidth)
        ax.add_collection(lc)
        ax.set_xlim(times.min(), times.max())
        
    
    
    def plot_dust_days_from_events(self, grimm_ds, grimm_events, avg_period, threshold, clean_avg, wind_csv_path=None, fig_dir=None):
    
        # Ensure datetime format
        full_time = pd.to_datetime(grimm_ds['time_utc'].values)
        full_dust = grimm_ds['dust'].values
    
        # Build full DataFrame from grimm_ds
        df = pd.DataFrame({
            'time': full_time,
            'dust': full_dust
        })
        
        # Localize time to UTC if naive
        if df['time'].dt.tz is None:
            df['time'] = df['time'].dt.tz_localize('UTC')
    
        # Extract unique dates from grimm_events
        grimm_events['event_peak_time'] = pd.to_datetime(grimm_events['event_peak_time'])  # just in case
        unique_days = grimm_events['event_peak_time'].dt.date.unique()
    
        # Load wind data if provided
        if wind_csv_path:
            wind_df = pd.read_csv(wind_csv_path, skiprows=10)
            wind_df.columns = wind_df.columns.str.strip()
            wind_df['Date_Time'] = pd.to_datetime(wind_df['Date_Time'], errors='coerce')
            wind_df = wind_df.dropna(subset=['Date_Time', 'wind_speed_set_1', 'wind_direction_set_1'])
            wind_df['wind_speed_set_1'] = pd.to_numeric(wind_df['wind_speed_set_1'], errors='coerce')
            wind_df['wind_direction_set_1'] = pd.to_numeric(wind_df['wind_direction_set_1'], errors='coerce')
            wind_df = wind_df.dropna()
    
        for day in sorted(unique_days):
            # Define extended time window: from 4 hours before day start to 4 hours after day end
            day_start = pd.Timestamp(day).tz_localize('UTC')  # localize to UTC
            window_start = day_start - pd.Timedelta(hours=4)
            window_end = day_start + pd.Timedelta(days=1) + pd.Timedelta(hours=4)
    
            # Filter dust data within this window
            mask = (df['time'] >= window_start) & (df['time'] <= window_end)
            day_data = df.loc[mask]
            if day_data.empty:
                continue
    
            wind_day = None
            if wind_csv_path:
                # Filter wind data within the same window
                wind_mask = (wind_df['Date_Time'] >= window_start) & (wind_df['Date_Time'] <= window_end)
                wind_day = wind_df.loc[wind_mask]
                if wind_day.empty:
                    print(f"No wind data for {day} (±4h), skipping wind panel.")
                    wind_day = None
                else:
                    # Resample to 1-hour intervals
                    wind_day = wind_day.set_index('Date_Time')[['wind_speed_set_1', 'wind_direction_set_1']]
                    wind_day = wind_day.resample('1H').mean().dropna()
                    times = wind_day.index
                    directions = wind_day['wind_direction_set_1']
                    speeds = wind_day['wind_speed_set_1']
    
                    # Unit vectors
                    u_unit = np.sin(np.radians(directions))
                    v_unit = np.cos(np.radians(directions))
    
            # Create subplots
            if wind_csv_path and wind_day is not None:
                fig = plt.figure(figsize=(12, 8))
                gs = GridSpec(2, 2, width_ratios=[20, 1], height_ratios=[3, 1], hspace=0.05, wspace=0.05)
                ax1 = fig.add_subplot(gs[0, 0])
                ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
                cax = fig.add_subplot(gs[:, 1])  # colorbar axis (occupies both rows)
            else:
                fig, ax1 = plt.subplots(figsize=(12, 6))
    
            # Plot vertical lines for event start and end
            if 'event_start' in grimm_events.columns and 'event_end' in grimm_events.columns:
                grimm_events['event_start'] = pd.to_datetime(grimm_events['event_start'])
                grimm_events['event_end'] = pd.to_datetime(grimm_events['event_end'])

                day_events = grimm_events[
                    (grimm_events['event_start'].dt.date <= day) &
                    (grimm_events['event_end'].dt.date >= day)
                ]

                for _, row in day_events.iterrows():
                    ax1.axvline(row['event_start'], color='#d73027', linestyle='--', linewidth=1.5, label='Event Start')
                    ax1.axvline(row['event_end'], color='#d73027', linestyle='--', linewidth=1.5, label='Event End')

    
            # Dust plot coloring masks
            below_clean_avg = day_data['dust'] < clean_avg
            between = (day_data['dust'] >= clean_avg) & (day_data['dust'] <= threshold)
            above_threshold = day_data['dust'] > threshold
    
            self.add_threshold_colored_line(
                ax1,
                day_data['time'].to_numpy(),
                day_data['dust'].to_numpy(),
                clean_avg,
                threshold
            )
    
            ax1.plot(day_data.loc[below_clean_avg, 'time'], day_data.loc[below_clean_avg, 'dust'],
                     '.', color='#7fcdbb', markersize=4, label=f'< clean_avg ({clean_avg:.2f})')
            
            ax1.plot(day_data.loc[between, 'time'], day_data.loc[between, 'dust'],
                     '.', color='k', markersize=4, label=f'between clean_avg and threshold')
            
            ax1.plot(day_data.loc[above_threshold, 'time'], day_data.loc[above_threshold, 'dust'],
                     '.', color='#d73027', markersize=4, label=f'> threshold ({threshold})')

    
            ax1.axhline(threshold, color='#d73027', linestyle='-', linewidth=1.5, zorder=10, label=f'Threshold = {threshold}')
            ax1.axhline(clean_avg, color='#7fcdbb', linestyle='-', linewidth=1.5, zorder=10, label=f'Clean Avg = {clean_avg:.2f}')
            
            ax1.set_xlim(window_start - pd.Timedelta(hours=1), window_end + pd.Timedelta(hours=1))
            
            # Set x-axis major ticks to every hour
            ax1.xaxis.set_major_locator(mdates.HourLocator())
            # Format x-axis labels to show hour and minute (e.g., 14:00)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            
            # Optional: rotate labels so they don't overlap
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # If you have ax2 sharing x-axis and want the same for it
            if wind_csv_path and wind_day is not None:
                ax2.xaxis.set_major_locator(mdates.HourLocator())
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
            ax1.set_title(f'{avg_period} min avg - Dust Concentration on {day} (±4 hours)')
            ax1.set_ylabel('Dust Concentration')
            ax1.set_yscale('log')
            ax1.set_ylim(0.01, 500)
            ax1.grid(False)                      # Disable gridlines
            ax1.tick_params(labelbottom=False)  # Hide x-axis tick labels
            ax1.set_xlabel('')                  # Remove x-axis label
    
            # Wind vector plot
            if wind_csv_path and wind_day is not None:

                bins = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

                # Blue to dark blue (colorblind-friendly, distinct)
                blue_to_darkblue = mcolors.LinearSegmentedColormap.from_list(
                    'blue_to_darkblue', ['#7fcdbb', '#0868ac'], N=5)  # light teal to dark blue
                
                # Orange to red (distinct warm colors)
                orange_to_red = mcolors.LinearSegmentedColormap.from_list(
                    'orange_to_red', ['#fdae61', '#d73027'], N=5)  # light orange to dark red
                
                blue_colors = blue_to_darkblue(np.linspace(0, 1, 5))
                orange_colors = orange_to_red(np.linspace(0, 1, 5))
                
                all_colors = np.vstack((blue_colors, orange_colors))
                
                discrete_cmap = mcolors.ListedColormap(all_colors)
                norm = mcolors.BoundaryNorm(bins, discrete_cmap.N)
                
                # Map speeds to discrete colors
                colors = discrete_cmap(norm(speeds))
            
                
                # Quiver and scatter plotting using discrete colors:
                q = ax2.quiver(times, [0]*len(times), u_unit, v_unit,
                               angles='xy', scale_units='xy', scale=30,
                               width=0.003,
                               headwidth=3, headlength=4, headaxislength=3,
                               color=colors)
                
                ax2.scatter(times, np.zeros_like(u_unit), color=colors, s=40, zorder=6)
                
                # Colorbar for discrete colormap
                sm = plt.cm.ScalarMappable(cmap=discrete_cmap, norm=norm)
                sm.set_array([])  # Needed for colorbar
                cbar = fig.colorbar(sm, cax=cax, boundaries=bins, ticks=bins)
                cbar.set_label('Wind Speed (m/s)')

                ax2.set_ylim(-1.5, 1.5)
                ax2.set_ylabel('Wind Unit Vector')
                ax2.set_xlabel('Time')
                ax2.set_yticks([])
                ax2.grid(True)
                
                ax2.set_aspect('equal', adjustable='datalim')  # <- This enforces 1:1 aspect ratio
    
            else:
                ax1.set_xlabel('Time')
    
            fig.tight_layout()
            
            if fig_dir:
                fig_path = os.path.join(fig_dir, f"{day}_{avg_period}min.png")
                fig.savefig(fig_path, dpi=300)
            
            else:
                plt.show()