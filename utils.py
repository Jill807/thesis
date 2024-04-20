import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from datetime import datetime, timedelta


class preprocessing:
    def __init__(self, image_path, sensor_file_path):
        self.sensor_data_file = pd.read_csv(sensor_file_path)
        self.image_file = pd.DataFrame(os.listdir(image_path), columns = ['FileName'])
        self.image_path = image_path
        self.image_file['File Path'] = self.image_path + self.image_file['FileName']

    
    def filter_stations(self):
        '''
        Keep stations we are interested in
        '''
        other_stations = ['HLL_hl_device_433', 'HLL_hl_device_442','HLL_hl_device_513','HLL_hl_device_237', 'HLL_hl_device_288', 'LTD_52030', 'HLL_hl_device_317', 'HLL_hl_device_452', 'HLL_hl_device_024']
        return self.sensor_data_file[self.sensor_data_file['station'].isin(other_stations)]
    def match_timestamps(self):
        '''
        This function creates timestamps for the image and sensor file path and returns a dataframe of the label associated with each image.
        return: Dataframe with image filename, timestamp and pm25 value
        '''

        ## Make timestamps: Label File
        self.sensor_data_file['datetime'] = pd.to_datetime(self.sensor_data_file['date'], format='%Y-%m-%d %H:%M:%S')

        # Split 'datetime' column into 'date' and 'time' columns
        self.sensor_data_file['date'] = self.sensor_data_file['datetime'].dt.date
        self.sensor_data_file['time'] = self.sensor_data_file['datetime'].dt.time
        
        ## Make timestamps: Image file
        self.image_file['epoch'] = self.image_file['FileName'].str.slice(stop = -4)

        # Convert epoch values to timestamps
        self.image_file['timestamp'] = self.image_file['epoch'].apply(lambda x: None if not x.isdigit() else datetime.fromtimestamp(int(x)).strftime('%Y-%m-%d %H:%M:%S'))
        self.image_file['timestamp'] = pd.to_datetime(self.image_file['timestamp'], errors='coerce')  # Convert to datetime
        self.image_file['time'] = self.image_file['timestamp'].dt.time
        self.image_file['date'] = self.image_file['timestamp'].dt.date

        # Round to nearest hour
        self.image_file['rounded_datetime'] = self.image_file['timestamp'].apply(self.round_to_hour)


        # Match on Time Stamp
        # print(self.image_file)
        self.labelled_data = pd.merge(self.image_file, self.sensor_data_file, how='left', left_on='rounded_datetime', right_on='datetime')
        # Cleaned data: remove NaNs. If small then downsize dataset to 20%
        self.labelled_data = self.clean_data(self.labelled_data, small = True)

        # Group by 'rounded_time' and 'FileName' and apply aggregation functions to the grouped data
        # self.grouped_max_pm25 = self.labelled_data.groupby(['rounded_datetime']).agg(
        # {
        # 'pm25': 'max',    # Maximum value of 'pm25' for each group
        # 'File Path': 'first'
        # }
        # )

        # Reset the index to make the grouped columns regular columns
        # self.grouped_max_pm25.reset_index(inplace=True)

        # print(self.grouped_max_pm25.head(20))

        return self.labelled_data
    
    def round_to_hour(self, dt):
        '''
        Round to hour function ...
        '''
        # If minute is greater than or equal to 30, round up to the next hour
        if dt.minute >= 30:
            dt += timedelta(hours=1)
        # Round down to the nearest hour by setting minute and second to 0
        return dt.replace(minute=0, second=0, microsecond=0)

    def clean_data(self, data, small = None):
        '''
        Cleans dataset
        '''
        # Keep the columns that I need
        columns_to_keep = ['File Path', 'pm25', 'timestamp', 'rounded_datetime', 'FileName']
        data = data[columns_to_keep]

        ## If I want a very small sample to train on I will take 20% of the data
        if small == True:
            sampled_data = data.sample(frac=0.2, random_state=42)  # Taking 20% of the data with a random state for reproducibility
            return sampled_data.dropna().reset_index(drop=True)
        return data.dropna().reset_index(drop=True)
    
    def small_data(self, data):
        '''
        Downsize data
        '''
        

# print(pd.read_csv('DataEarly2023.csv'))
pp = preprocessing('2024-03-12/', '12032024_sensor.csv')

labelled_data = pp.match_timestamps()



