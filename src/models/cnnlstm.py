
import sys
import os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
from src.data.dataset import createDataset
from torchvision import transforms
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.regression import R2Score



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
 
    def train_model(self, trainloader, model, loss_fn, optimizer, num_epochs, checkpoint_epoch):
        """
        Train a PyTorch model and print model performance metrics.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            The dataloader object.
        model : torch.nn.Module
            The PyTorch model that we want to train.
        loss_fn : torch.nn.modules.loss._Loss
            The loss function.
        optimizer : torch.optim.Optimizer
            The optimization algorithm.
        num_epochs : int
            Number of epochs for training the model.
        """
        model.train(True)
        output_file = "losses.txt"

        # if start_epoch > 0:
        #     resume_epoch = start_epoch - 1
        #     self.resume(model, f"epoch-{resume_epoch}.pth")
        for epoch in range(checkpoint_epoch, num_epochs):  # loop over the dataset multiple times
            print("Epoch:", epoch)
            running_loss = 0.0
            score_arrays= [] # add the field for appending the loss
            for i, data in enumerate(trainloader, 0):
                print("Minibatch:", i)
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                print("----------------------------------")
                print("Labels:", labels.size())
                print("Inputs:", inputs.size())

                # forward + backward + optimize
                outputs = model(inputs)
                print("Output size:", outputs.size())
                print("Outputs:", outputs, labels.dtype)
                loss = loss_fn(outputs, labels)
                print("Loss:", loss)
                loss.backward()
                optimizer.step()
                score_arrays.append(loss.item()) # append the loss
                # print statistics
                with open(output_file, "a") as f:
                    f.write(f"Epoch {epoch + 1}, Loss: {loss}\n")  # Writing each loss value on a separate line

                running_loss += loss.item()
                if i % 10 == 0:    # print every 10 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 10))
                    running_loss = 0.0
            self.checkpoint(model, f"epoch-{epoch}.pth")
    
    def evaluate_model(self, dataloader, model, num_epochs):
        """
        Evaluate a PyTorch regression model and print model performance metrics.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            The dataloader object.
        model : torch.nn.Module
            The PyTorch model that we want to evaluate.
        """
        model.eval() # set the model to evaluation mode
        r2score = R2Score().to(device)
        y_predlist = []
        ylist = []
        # Since we do not want to train the model, make sure that we deactivate the gradients
        with torch.no_grad():
            metrics = {'mse': [], 'mae': [], 'r2': [], 'rmse': []} # initialize empty lists for metrics
            # Loop through the dataloader
            for x, y in dataloader:
                x = x.to(device)
                y = y.to(device)
                y_pred = model(x) # make predictions using the model
                ylist.append(y)
                y_predlist.append(y_pred)
                # Compute evaluation metrics
                mse = torch.mean((y_pred - y)**2).item() # mean squared error
                mae = torch.mean(torch.abs(y_pred - y)).item() # mean absolute error
                rmse = torch.sqrt(torch.tensor(mse))
                # r2 = r2score(y_pred, y)
                # Append the evaluation metrics to the lists
                metrics['mse'].append(mse)
                metrics['mae'].append(mae)
                # metrics['r2'].append(r2)
                metrics['rmse'].append(rmse)
            y_pred_tensor = torch.cat(y_predlist, dim=0)
            y_tensor = torch.cat(ylist, dim=0)
            r2 = r2score(y_pred_tensor, y_tensor)
            # Print the averaged evaluation metrics
            with open("metrics.txt", "w") as f:
                # Write the metrics to the file
                f.write("MSE: {}\n".format(mse))
                f.write("MAE: {}\n".format(mae))
                f.write("RMSE: {}\n".format(rmse.item()))  
                f.write("R2 Score: {}\n".format(r2.item()))  

        print(metrics['mse'])
        print(len(metrics))
        print("="*40)
        print("Evaluate model performance:")
        for metric_name, values in metrics.items():
            print("Averaged %s: %.4f" % (metric_name.upper(), np.mean(values)))

        #---------------------PLOTTING----------------------------#
        # Plot MSE
        plt.plot(range(1,len(metrics['mse'])+1), metrics['mse'], label='MSE', color='blue')
        # Plot MAE
        plt.plot(range(1,len(metrics['mae'])+1), metrics['mae'], label='MAE', color='red')

        # Add labels and title
        plt.title('MSE and MAE Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('foo.png')
        print("Plot updated")
        
    def checkpoint(self, model, filename):
        torch.save({
            'optimizer': optimizer.state_dict(),
            'model': model.state_dict(),
        }, filename)
    
    def resume(self, model, filename): 
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

transform = transforms.Compose([
    transforms.Resize(size = (224,224))
    , transforms.ToTensor()]) 
target_transform = transforms.Compose([
    transforms.ToTensor()]) 

cD = createDataset(image_path='2024-03-12/',
                                    sensor_file='12032024_sensor.csv', transform = transform, target_transform=None)


train_set, test_set = torch.utils.data.random_split(cD, [int(0.8*len(cD)), len(cD) - int(0.8*len(cD))])
train_set[0][0].size()


trainloader = DataLoader(train_set, batch_size=16,
                        shuffle=True, num_workers=0)
testloader = DataLoader(test_set, batch_size=4,
                        shuffle=True, num_workers=0)



model = CNN_LSTM(512*7*7 , 8, 1).to(device)
print(model)
# Try commit
# Define the path to the saved model state dictionary
# path_to_saved_model = "epoch-49.pth"  # Replace with the actual path to your saved model

# Load the model state dictionary
# checkpoint = torch.load(path_to_saved_model, map_location=torch.device('cpu'))

# Load weights into the model
# model.load_state_dict(checkpoint['model'])
# Optionally, move the model to the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Initialize the loss function
loss_fn = nn.MSELoss()  # mean square error
learning_rate = 1e-4
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
checkpoint_epoch = 0
epochs = 50

# model.train_model(trainloader, model, loss_fn, optimizer, epochs, checkpoint_epoch)
model.evaluate_model(testloader, model, epochs)

