
import torch
import torch.nn as nn
import torchvision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.data import DataLoader, TensorDataset
import tensorflow as tf
from tataDataset import createDataset
from torchvision import transforms
import torchvision.models as models
from torchvision.models import VGG16_Weights
from tataDataset import createDataset
from torchvision import transforms
import torch.optim as optim
# from torcheval.metrics import R2Score



device = "cuda" if torch.cuda.is_available() else "cpu"

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pre-trained VGG16 model
        self.flatten = nn.Flatten()
        self.vgg16 = models.vgg16(weights= VGG16_Weights.DEFAULT)
        num_features = self.vgg16.classifier[-1].in_features
        # Single output neuron for regression
        self.vgg16.classifier[-1] = torch.nn.Linear(num_features, 1)  


    def forward(self, x):
        x = self.vgg16(x)
        return x
    
## Create transforms, test and train set, load data and training loop

transform = transforms.Compose([
    transforms.Resize(size = (224,224))
    , transforms.ToTensor()]) 
target_transform = transforms.Compose([
    transforms.ToTensor()]) 

cD = createDataset(image_path='2024-03-12/',
                                    sensor_file='12032024_sensor.csv', transform = transform, target_transform=None)

train_set, test_set = torch.utils.data.random_split(cD, [int(0.8*len(cD)), len(cD) - int(0.8*len(cD))])

trainloader = DataLoader(train_set, batch_size=4,
                        shuffle=True, num_workers=0)

testloader = DataLoader(test_set, batch_size=4,
                        shuffle=True, num_workers=0)




model = Model().to(device)


# Initialize the loss function
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr=0.0001)
learning_rate = 1e-3
batch_size = 64
epochs = 5

## Model

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs.squeeze())

        # print(outputs, labels.dtype)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0