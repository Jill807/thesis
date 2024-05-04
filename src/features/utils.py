import os
import sys
sys.path.append(os.getcwd())
import pandas as pd
from datetime import datetime, timedelta
import math
import numpy as np


class preprocessing:
    def __init__(self, sensor_file_path):
        # Read in sensor data
        self.sensor_data_file = pd.read_csv(sensor_file_path)

        # Read in iamge data
        self.image_file = self.retrieve_images(parent_directory = '/gpfs/scratch1/shared/jgulbis')
        # self.image_file = pd.DataFrame(os.listdir(image_path), columns = ['FileName'])
        # self.image_path = '2024-03-12/'
        # self.image_file['FilePath'] = self.image_path + self.image_file['FileName']

        # Read in weather data
        self.weather_df = self.weather_preprocessing(pd.read_csv(r'WeatherMarch01April30.txt', delimiter= '\t|,', header = 32, engine = 'python'))
        # For a small dataset, comment out above code and rerun with code below:
        # self.weather_df = self.weather_preprocessing(pd.read_csv(r'WeatherMarch12.txt', delimiter= '\t|,', header = 32, engine = 'python'))


    def main(self):
        """
        Function to preprocess the dataset and read into parquet file
        """
        # Get matching timestamps between the images and sensor times
        timestamps_df = self.match_timestamps()
        ## --------------------TEMPORARY FILTER ------------------
        # Replace with filter_stations after
        # filtered_df = data[data['station'] == 'NL10644']
        filtered_df = self.filter_stations(timestamps_df)

        # Cleaned data: remove NaNs. If small then downsize dataset to 20%
        cleaned_df = self.clean_data(filtered_df, small = False)
        print("CLEANED DF:-----------------------------")
        print(cleaned_df.columns)

        # Get distances from tata steel to the sensors
        distance_data = self.direction_distance(cleaned_df)
        print("processing")

        # Merge weather and sensor data
        merge_weather_sensor = self.merge_weather_sensor(distance_data, self.weather_df)
        print("processing")
        print("Weather sensor", merge_weather_sensor['rounded_datetime'].unique())

        # Get the pm2.5 value from the nearest station i.e. the one with the smallest wind direction difference
        nearest_station = self.nearest_station(merge_weather_sensor)
        print("processing")

        # Create t sequences of images
        sequenced_df = self.sequence_generation(nearest_station, nr_timestamps = 7)
        print("processing")
        print("Length of sequenced df", len(sequenced_df))

        sequenced_df = sequenced_df.drop(columns = ['time', 'date', 'datetime', 'date_sensor', 'time_sensor', 'datetime_weather',
       'date_weather', 'time_weather'])
        sequenced_df = sequenced_df.dropna(axis =0)
        sequenced_df = sequenced_df.reset_index()
        print("done")
        return sequenced_df
    
    def retrieve_images(self, parent_directory):
        
        folder_paths = [os.path.join(root, d) for root, dirs, _ in os.walk(parent_directory) for d in dirs]

        # Initialize an empty list to store file information
        file_info = []

        # Iterate over each folder path
        for folder_path in folder_paths:
            # Iterate over files and directories in the folder
            for entry in os.scandir(folder_path):
                # Check if the entry is a file
                if entry.is_file():
                    # Construct the full file path
                    file_path = os.path.join(folder_path, entry.name)
                    # Append file information to the list
                    file_info.append({'FileName': entry.name, 'FilePath': file_path})

        # Create a DataFrame from the list of file information
        print("Done: RETRIEVED IMAGES")
        return pd.DataFrame(file_info)
    
    def filter_stations(self, df):
        '''
        Keep stations we are interested in
        '''
        other_stations = ['HLL_hl_device_433', 'HLL_hl_device_442','HLL_hl_device_513','HLL_hl_device_237', 'HLL_hl_device_288', 'LTD_52030', 'HLL_hl_device_317', 'HLL_hl_device_452', 'HLL_hl_device_024']
        filtered_df = df[df['station'].isin(other_stations)]
        # grouped_df = filtered_df.groupby(['station', 'datetime'])['pm25_kal'].mean().reset_index(name='average_pm_25kal')
        # grouped_df['date'] = filtered_df['datetime'].dt.date
        # grouped_df['time'] = filtered_df['datetime'].dt.time
        # grouped_df['average_pm_25kal'] = grouped_df['average_pm_25kal'].clip(0,150)
        return filtered_df
    
    def match_timestamps(self):
        '''
        This function creates timestamps for the image and sensor file path and returns a dataframe of the label associated with each image.
        return: Dataframe with image filename, timestamp and pm25 value
        '''

        ## Make timestamps: Label File
        self.sensor_data_file['datetime'] = pd.to_datetime(self.sensor_data_file['date'], format='mixed')
        print("processing: timestamps")

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
        # print(np.unique(self.sensor_data_file['station']))

        # Match on Time Stamp
        self.labelled_data = pd.merge(self.image_file, self.sensor_data_file, how='left', left_on='rounded_datetime', right_on='datetime',  suffixes=('', '_sensor'))
        print("Done: MATCH TIMESTAMPS")
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

        data = data.drop(columns = ['wd', 'ws', 'temp', 'rh', 'pm10', 'pm10_kal', 'pm25_kal'])
        # columns_to_keep = ['FilePath', 'pm25', 'timestamp', 'rounded_datetime', 'FileName', 'datetime', 'station', 'lat', 'lon']
        # data = data[columns_to_keep]
        # Change timestamp to datetime column
        ## If I want a very small sample to train on I will take 20% of the data
        if small == True:
            sampled_data = data.sample(frac=0.2, random_state=42)  # Taking 20% of the data with a random state for reproducibility
            return sampled_data.dropna().reset_index(drop=True)
        return data.dropna().reset_index(drop=True)
    
    def sequence_generation(self, data, nr_timestamps):
        """
        """
        data = data.sort_values(by = 'timestamp')
        data['frame_sequence'] = data['FilePath']
        for i in range(1,nr_timestamps+1):
            data[f't-{i}'] = data['FilePath'].shift(i)
            data['frame_sequence'] += " " + data[f't-{i}'] 
        data['frame_sequence'] = data['frame_sequence'].apply(lambda x: x.split() if isinstance(x, str) else pd.NA)
        return data 
    
    def direction_distance(self, sensor_df):
        """
        Create Distance and Direction features from Tata Steel to the sensors
        """
        print("processing: Direction distance")
        # Store tata lat and long
        sensor_df['tata_lat'] = 52.480234
        sensor_df['tata_lon'] = 4.607623

        # Function to calculate the distance between two points
        def calculate_distance(sensor_x, sensor_y, point_x, point_y):
            print("Dtype", sensor_x.dtype)
            print("Dtype", point_x.dtype)
            return np.sqrt((sensor_x - point_x)**2 + (sensor_y - point_y)**2)

        # Function to calculate the angle between two points
        def calculate_angle(sensor_x, sensor_y, point_x, point_y):
            angles = np.degrees(np.arctan2(sensor_y - point_y, sensor_x - point_x))
            angles[angles < 0] += 360  # Adjust negative angles
            return angles
            
        # Sensor coordinates
        sensor_coordinates= sensor_df[['lat', 'lon']].values # lat and lon of the sensors
        lat = sensor_coordinates[:,0]
        lon = sensor_coordinates[:,1]

        # Tata coordinates
        tata_coordinates = sensor_df[['tata_lat', 'tata_lon']].values  # lat and lon of Tata Steel 
        tata_lat = tata_coordinates[:,0]
        tata_lon = tata_coordinates[:,1]

        # Calculate distance and direction
        distance = calculate_distance(lat, lon, tata_lat, tata_lon)
        direction = calculate_angle(lat, lon, tata_lat, tata_lon)

        # Store in dataframe
        sensor_df['distance'] = distance
        sensor_df['direction'] = direction
        print("Done: DIRECTION DISTANCE")
        return sensor_df
    
    def weather_preprocessing(self, weather_df):
        """
        """
        # Preprocess the weather column names (get rid of white space)
        weather_df.columns = [i.strip() for i in weather_df.columns]
        weather_df.columns

        # 257 and 225 are two different weather stations next to each other.
        # They measure different variables which is why I will be joining the columns to get one table with all the weather values
        station_257 = weather_df[weather_df['# STN'] == 257]
        station_225 = weather_df[weather_df['# STN'] == 225]

        # Keep relevant columns from station 225
        # STN = Station Number
        # YYYYMMDD = date
        # HH = Hour of day
        # DD = Mean wind direction (in degrees) during the 10-minute period preceding the time of observation 
        # (360=north; 90=east; 180=south; 270=west; 0=calm 990=variable)
        # FH = Hourly mean wind speed (in 0.1 m/s)
        # FX = Mean wind speed (in 0.1 m/s) during the 10-minute period preceding the time of observation
        # FF = Maximum wind gust (in 0.1 m/s) during the hourly division
        station_225 = station_225[["# STN", "YYYYMMDD", "HH", "DD", "FH", "FX", "FF"]]

        # Keep relevant columns from station 257 
        # STN = Station Number
        # YYYYMMDD = date
        # T    = Temperature (in 0.1 degrees Celsius) at 1.50 m at the time of observation
        # TD   = Dew point temperature (in 0.1 degrees Celsius) at 1.50 m at the time of observation
        # SQ   = Sunshine duration (in 0.1 hour) during the hourly division; calculated from global radiation 
        # (-1 for <0.05 hour)
        # Q    = Global radiation (in J/cm2) during the hourly division
        # DR   = Precipitation duration (in 0.1 hour) during the hourly division
        # RH   = Hourly precipitation amount (in 0.1 mm) (-1 for <0.05 mm)
        # U    = Relative atmospheric humidity (in percents) at 1.50 m at the time of observation
        station_257 = station_257[["# STN", "YYYYMMDD", "HH", "T", "TD", "SQ", "Q", "DR", "RH", "U"]]

        # Join both dataframes on date and time
        weather_station_df = pd.merge(station_225, station_257, on=["YYYYMMDD", "HH"], suffixes=('_225', '_257'))
        # Convert columns to numeric
        weather_features = ["DD", "FH", "FX", "FF", "T", "TD", "SQ", "Q", "DR", "RH", "U"]

        # Create name dictionary to decipher abbreviations
        name_dict = {"DD": "mean_wind_direction", "FH": "hourly_mean_wind_speed", "FX": "mean_wind_speed", 
                    'FF': "max_wind_gust", "T": "temperature", 
                    "TD": "dew_point_temperature","SQ": "sunshine_duration",
                    "Q": "global_radiation", "DR": "precipitation_duration", 
                    "RH": "hourly_precipitation_amount", 
                    "U": "relative_atmospheric_humidity"}

        # Concver to numeric
        for i in weather_features:
            weather_station_df[i] = pd.to_numeric(weather_station_df[i])

        # Convert values to 0.1 values. For this I copy the list to be
        # converted and remove DD, U and Q because they don't need to be multiplied by 0.1
        w_conversion_df = weather_features.copy()
        w_conversion_df.remove("DD")
        w_conversion_df.remove("U")
        w_conversion_df.remove("Q")

        # Convert into right format
        for i in w_conversion_df:
            weather_station_df[i] = weather_station_df[i]*0.1

        # Convert 'datetime' column to datetime type
        weather_station_df['datetime'] = pd.to_datetime(weather_station_df['YYYYMMDD'], format = '%Y%m%d')

        # Split 'datetime' column into 'date' and 'time' columns
        weather_station_df['date'] = weather_station_df['datetime'].dt.date
        weather_station_df['time'] = pd.to_datetime(weather_station_df['HH'], format= "%H", errors="coerce").dt.time

        # Convert 'date' column to datetime object
        weather_station_df['date'] = pd.to_datetime(weather_station_df['date'])

        # Convert 'time' column to timedelta object
        weather_station_df['time'] = pd.to_timedelta(weather_station_df['time'].astype(str))

        # Add 'date' and 'time' columns to get the combined datetime
        weather_station_df['datetime'] = weather_station_df['date'] + weather_station_df['time']

        return weather_station_df
    
    def merge_weather_sensor(self, sensor_df, weather_df):
        """
        """
        print(sensor_df.columns)
        print(weather_df)
        wind_direction = pd.merge(sensor_df, weather_df, left_on='rounded_datetime', right_on='datetime', how='inner', suffixes = ['', '_weather'])

        return wind_direction
    
    def nearest_station(self, wind_direction):
        """
        """
        # Calculate direction difference
        wind_direction['direction_difference'] = abs(wind_direction['DD'].astype(float)- wind_direction['direction'])

        idx_min_direction = wind_direction.groupby(['rounded_datetime'])['direction'].idxmin()
        print(idx_min_direction)


        # Select the rows with the smallest direction difference for each unique date and time
        result = wind_direction.loc[idx_min_direction]
        print("result", result)

        return result.reset_index()
    
# print(pd.read_csv('DataEarly2023.csv'))
pp = preprocessing('VelsenMarch01April30.csv')
labelled_data = pp.main()
labelled_data.to_parquet('output_file.parquet', engine='fastparquet')
with open("dataset_info.txt", "a") as f:
    print("Dataset columns", labelled_data.columns, file = f)
    print("-----------------Dataset------------------", file = f)
    print(labelled_data[['FileName', 'rounded_datetime', 'timestamp', 'pm25', 'direction_difference', 'frame_sequence']], file = f)
    print("Datasize shape:", labelled_data.shape, file = f)
    print("Frame sequence", labelled_data['frame_sequence'][2], file = f)

