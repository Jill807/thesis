import sys
import os
sys.path.append(os.getcwd())
import torch
# from src.features.utils import preprocessing
from PIL import Image
import pandas as pd
import ast
from torchvision import transforms
import PIL
import numpy as np




class pm25Dataset(torch.utils.data.Dataset):
    """Creates Tata Steel Dataset"""

    def __init__(self, transform):
        """
        Arguments:
            transform: the transformation to apply to the images and target
        """
        self.transform = transform
        self.data = pd.read_parquet('src/data/dataset_without_corrupted_images.parquet', engine = 'fastparquet')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
    
        #Image
        try:
            image = Image.open(self.data.loc[idx, 'FilePath'])
            image = self.transform(image)
            image = torch.squeeze(image)
    
        except (OSError, PIL.UnidentifiedImageError):
            print("Corrupted")

        # Additional features
        FH            = self.data.loc[idx, 'FH']
        SQ            = self.data.loc[idx, 'SQ']
        Q             = self.data.loc[idx, 'Q']
        FX            = self.data.loc[idx, 'FX']
        SIN_HOUR      = self.data.loc[idx, 'hour_sin']
        COS_HOUR      = self.data.loc[idx, 'hour_cos']
        DATETIME      = self.data.loc[idx, 'rounded_datetime'].timestamp()
        WEIGHTED_PM25 = self.data.loc[idx, 'weighted_pm25']

        features = np.stack((FH,SQ,Q,FX, SIN_HOUR, COS_HOUR, DATETIME, WEIGHTED_PM25), axis = 0)

        label = self.data.loc[idx, 'pm25']
        
        return image, features, label

# transform = transforms.Compose([transforms.Resize(size = (224,224)), transforms.ToTensor()]) 

# pm25_dataset = pm25Dataset(transform = transform)

# im, wfeatures, label = pm25_dataset.__getitem__(5)
# print(features[0])
# print(im.size())
# print("Success!")