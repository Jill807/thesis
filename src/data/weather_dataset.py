import sys
import os
sys.path.append(os.getcwd())
import torch
# from src.features.utils import preprocessing
from PIL import Image
import pandas as pd
import ast
from torchvision import transforms




class createWeatherDataset(torch.utils.data.Dataset):
    """Creates Tata Steel Dataset"""

    def __init__(self, image_path, sensor_file, transform=None, target_transform = None, regression = True):
        """
        Arguments:
            image_path: path to image folder
            sensor_file: csv file with sensor data
            transform: the transformation to apply to the images and target
            regression: if the dataset should be made for a regression or classification task
        """
        self.image_path = image_path
        self.transform = transform
        self.target_transform = target_transform
        self.regression = regression
        print("THE DATASET LABEL IS REGRESSION:", self.regression)


        # Instantiate preprocessing steps
        # self.pp = preprocessing(image_path, sensor_file)
        
        # #Clean the dataset before making it a tensor. The small parameter downsizes the dataset considerably for testing purposes
        # self.data = self.pp.main()

        # self.data = pd.read_parquet('labelled_dataset2.parquet', engine = 'fastparquet')
        self.data = pd.read_parquet('labelled_dataset2.parquet', engine = 'fastparquet')


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_paths = self.data.loc[idx, 'frame_sequence']
        # img_paths = ast.literal_eval(img_paths.decode('utf-8'))

        FH = torch.tensor(self.data.loc[idx, 'FH_sequence'], dtype = torch.float32)
        SQ = torch.tensor(self.data.loc[idx, 'SQ_sequence'], dtype = torch.float32)
        Q = torch.tensor(self.data.loc[idx, 'Q_sequence'], dtype = torch.float32)
        FX = torch.tensor(self.data.loc[idx, 'FX_sequence'], dtype = torch.float32) 

        weatherfeatures = torch.stack((FH,SQ,Q,FX), dim = 0)
        weatherfeatures = weatherfeatures.view(8,4)
        # weatherfeatures = weatherfeatures.unsqueeze(dim = 2)
        # print(weatherfeatures.shape)
        # weather_features = self.data.loc[idx, ['FH_sequence', 'SQ_sequence', 'Q_sequence', 'FX_sequence']]
        
        # print(weatherfeatures)
        # Retrieve image path and labels
        if self.regression:
            label = self.data.loc[idx, 'pm25']
        else:
            label = self.data.loc[idx, 'pm25_binary']

        # Read in the image
        li = []
        features = []
        for i, im in enumerate(img_paths):
            image = Image.open(im)
            # Apply transforms
            image = self.transform(image)
            image = torch.squeeze(image)
            li.append(image)
        # Stack to get multiple time tensors (shape: T,H,W,C)
        seq = torch.stack(li, dim = 0)



        # Convert label to torch and float
        
        label = torch.tensor(label, dtype=torch.float32)

        return seq, weatherfeatures, label

# transform = transforms.Compose([transforms.Resize(size = (224,224)), transforms.ToTensor()]) 

# cD = createWeatherDataset(image_path='2024-03-12/',
#                                     sensor_file='12032024_sensor.csv', transform = transform, target_transform=None, regression = True)

# im, feature, label = cD.__getitem__(5)
# print(feature[0])
# print(im.size())
# print("Success!")