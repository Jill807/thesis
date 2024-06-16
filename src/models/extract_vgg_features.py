import sys
import os
sys.path.append(os.getcwd())
import torch
import torch.nn           as nn
from torchvision          import transforms
import torchvision.models as models
from src.data.fulldataset import pm25Dataset
from torchvision          import transforms
import pandas             as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

class vgg_extractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pre-trained VGG16 model with imagenet weights
        self.vgg16 = models.vgg16(weights = 'DEFAULT')

        # Freeze all the parameters in VGG16
        for param in self.vgg16.parameters():
            param.requires_grad = False
        
        # Extract feature layers without fully connected layers from VGG16
        self.features        = self.vgg16.features
        self.avgpool         = self.vgg16.avgpool

        # The output dimension of the cnnn
        self.vgg16_output_dimension = 512*7*7

        self.score_arrays = []
    
    def forward(self, image):
        # No training for VGG16 feature extraction
        with torch.no_grad():
            frame_feature = self.features(image)
            frame_feature = frame_feature.view(-1, self.vgg16_output_dimension)
        # Convert to numpy
        np_image = frame_feature.detach().numpy()
        return np_image
    

# Create transforms
transform           = transforms.Compose([transforms.Resize(size = (224,224))
                                            ,transforms.ToTensor()]) 
target_transform    = transforms.Compose([ transforms.ToTensor()]) 

# Initialise model
model = vgg_extractor()

# Initialise Dataframe
nr_cols      = 25088
col_names    = [ "X" + str(idx) for idx in range(nr_cols)]
cnn_features = pd.DataFrame(columns = col_names)
cnn_labels   = pd.DataFrame(columns = ["PM2.5"])
aux_features = pd.DataFrame(columns = ["FH", "SQ", "Q", "FX", "SIN_HOUR", "COS_HOUR", "DATETIME", "WEIGHTED_PM25"])

# Create dataset
pm25_images = pm25Dataset(transform=transform)

# Store image features and labels
for i in range(1, len(pm25_images)):
    image, additional_features, label  = pm25_images[i]
    features                           = model(image)
    cnn_features.loc[i, :]             = features
    cnn_labels.loc[i, "PM2.5"]         = label
    aux_features.loc[i, :]             = additional_features
    if i%50 ==0:
        print(i)

cnn_labels["PM2.5"] = cnn_labels["PM2.5"].astype(float)
cnn_dataframe = pd.concat([cnn_features, aux_features, cnn_labels], axis = 1)
cnn_dataframe.to_parquet('src/data/cnn_features.parquet', engine='fastparquet')

