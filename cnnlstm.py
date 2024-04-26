
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import createDataset
from torchvision import transforms
import torchvision.models as models
from dataset import createDataset
from torchvision import transforms
import torch.optim as optim
# from torcheval.metrics import R2Score



device = "cuda" if torch.cuda.is_available() else "cpu"

class CNN_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim):
        super().__init__()

        # Load pre-trained VGG16 model with imagenet weights
        self.vgg16 = models.vgg16(weights = 'DEFAULT')
        # Freeze all the parameters in VGG16
        for param in self.vgg16.parameters():
            param.requires_grad = False
        # Extract feature layers without fully connected layers from VGG16
        self.features = self.vgg16.features
        self.avgpool = self.vgg16.avgpool

        # Initialise hidden layer and layer dimension for LSTM
        self.H = hidden_dim
        self.L = layer_dim

        # LSTM model
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layer_dim,
            batch_first=True)
        
        # Fully connected layers with one output for regression
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128,1)

    def forward(self, x):
        # No training for VGG16 feature extraction
        with torch.no_grad():
            # Extract dimensions of x
            batch_size, timesteps, C, H, W = x.size()
            # Reshape to match input dimensions of VGG16
            c_in = x.view(batch_size*timesteps, C, H, W)
            # Retrieve features
            c_out = self.features(c_in)
            # Pass through average pooling layer
            a_out = self.avgpool(c_out)
            print("A_out size:", a_out.size())
            # Reshape to fit input dimensions of LSTM
            a_out = a_out.view(batch_size, timesteps, -1)
            print("C_out size reshaped:", a_out.size())
        
        # Initialisation of first hidden and cell nodes
        # Num layers, batch size, hidden layers
        h0 = torch.zeros(self.L, a_out.size(0), self.H).to(device)
        c0 = torch.zeros(self.L, a_out.size(0), self.H).to(device)

        #Forward propagate through LSTM
        r_out, _ = self.rnn(a_out, (h0.detach(), c0.detach()))

        # Decode hidden state of last time step
        f_out = self.fc1(r_out[:, -1, :]) 
        # Pass through final fully connected layer
        f2_out = self.fc2(f_out)

        return f2_out.squeeze()
    
transform = transforms.Compose([
    transforms.Resize(size = (224,224))
    , transforms.ToTensor()]) 
target_transform = transforms.Compose([
    transforms.ToTensor()]) 

cD = createDataset(image_path='2024-03-12/',
                                    sensor_file='12032024_sensor.csv', transform = transform, target_transform=None)


train_set, test_set = torch.utils.data.random_split(cD, [int(0.8*len(cD)), len(cD) - int(0.8*len(cD))])
train_set[0][0].size()


trainloader = DataLoader(train_set, batch_size=4,
                        shuffle=True, num_workers=0)

testloader = DataLoader(test_set, batch_size=4,
                        shuffle=True, num_workers=0)



model = CNN_LSTM(512*7*7,2,1).to(device)
print(model)

# Initialize the loss function
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr=0.0001)
learning_rate = 1e-3
batch_size = 64
epochs = 5


for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        print("----------------------------------")
        print("Labels:", labels.size())
        print("Inputs:", inputs.size())
        # forward + backward + optimize
        outputs = model(inputs)
        print("Output size:", outputs.size())
        print("Outputs:", outputs, labels.dtype)
        loss = loss_fn(outputs, labels)
        print("Loss:", loss)
        # loss.backward()
        # optimizer.step()

        # # print statistics
        # running_loss += loss.item()
        # if i % 2000 == 1999:    # print every 2000 mini-batches
        #     print('[%d, %5d] loss: %.3f' %
        #           (epoch + 1, i + 1, running_loss / 2000))
        #     running_loss = 0.0
