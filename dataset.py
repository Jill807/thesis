import torch
import os
from utils import preprocessing
from PIL import Image
import cv2
from torchvision.io import read_image
from torchvision import transforms



class createDataset(torch.utils.data.Dataset):
    """Tata Steel Dataset"""

    def __init__(self, image_path, sensor_file, transform=None, target_transform = None):
        """
        Arguments:
            image_path
            sensor_file
            transform
        """
        self.image_path = image_path
        self.transform = transform
        self.target_transform = target_transform

        # Instantiate preprocessing steps
        self.pp = preprocessing(image_path, sensor_file)
        
        #Clean the dataset before making it a tensor. The small parameter downsizes the dataset considerably for testing purposes
        self.data = self.pp.match_timestamps()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Retrieve image path and labels
        img_paths, label = self.data.loc[idx, 'frame_sequence'], self.data.loc[idx, 'pm25']

        # Read in the image
        li = []
        for i in img_paths:
            image = Image.open(i)
            # Apply transforms
            image = self.transform(image)
            image = torch.squeeze(image)
            li.append(image)
        seq = torch.stack(li, dim = 0)

        # Convert label to torch and float
        label = torch.tensor(label, dtype=torch.float32)

        return seq, label

transform = transforms.Compose([
    transforms.Resize(size = (224,224))
    , transforms.ToTensor()]) 

cD = createDataset(image_path='2024-03-12/',
                                    sensor_file='12032024_sensor.csv', transform = transform, target_transform=None)

ex, label = cD.__getitem__(5)
print(ex.size())