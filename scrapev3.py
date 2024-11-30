# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 09:08:08 2024

@author: dpasq
"""

import warnings
from pybaseball import statcast
import pandas as pd
import datetime
import os

# Suppress specific warnings from pybaseball
warnings.filterwarnings("ignore", category=FutureWarning, module="pybaseball")

# Define the start and end years
start_year = 2024
end_year = 2024

# Define the chunk size in days
chunk_size = 30

def download_statcast_data(start_date, end_date, chunk_size):
    current_date = start_date
    while current_date <= end_date:
        chunk_end_date = current_date + datetime.timedelta(days=chunk_size - 1)
        if chunk_end_date > end_date:
            chunk_end_date = end_date
        
        print(f"Downloading data for {current_date.strftime('%Y-%m-%d')} to {chunk_end_date.strftime('%Y-%m-%d')}")
        data = statcast(start_dt=current_date.strftime('%Y-%m-%d'), end_dt=chunk_end_date.strftime('%Y-%m-%d'))
        yield data
        
        current_date = chunk_end_date + datetime.timedelta(days=1)

for year in range(start_year, end_year + 1):
    start_date = datetime.datetime.strptime(f"{year}-04-01", '%Y-%m-%d')
    end_date = datetime.datetime.strptime(f"{year}-12-31", '%Y-%m-%d')

    # Create a list to hold data chunks
    all_data = []
    
    # Download data in chunks
    for chunk_data in download_statcast_data(start_date, end_date, chunk_size):
        all_data.append(chunk_data)

    # Concatenate all chunks into a single dataframe
    all_data_df = pd.concat(all_data, ignore_index=True)

    # Create the data directory
    os.makedirs('pitch_data', exist_ok=True)

    # Export the data to a .pkl file for the current year
    pkl_filename = os.path.join('pitch_data', f'pitch_data_{year}.pkl')
    all_data_df.to_pickle(pkl_filename)
    print(f"Data for {year} exported to {pkl_filename}")
