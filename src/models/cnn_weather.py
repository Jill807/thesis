
import sys
import os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
from src.data.dataset import createDataset
from src.data.weather_dataset import createWeatherDataset
from torchvision import transforms
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.regression import R2Score
import torch.nn.functional as F
from src.features.histogramsmoothing import smoothing
from sklearn.metrics import confusion_matrix



device = "cuda" if torch.cuda.is_available() else "cpu"

class CNN_WEATHER(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, regression = True):
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

        # LSTM model: Not relevant in this model!!

        # self.rnn = nn.LSTM(
        #     input_size=input_dim,
        #     hidden_size=hidden_dim,
        #     num_layers=layer_dim,
        #     batch_first=True)


        # Fully connected layers with one output for regression
        # 6 additional input dimensions: 4 weather features 2 time encoded features
        print("INPUT DIM + 6", input_dim+6)
        self.fc1 = nn.Linear(input_dim+6, 32)
        self.fc2 = nn.Linear(32,1)
        # self.fc3 = nn.Linear(128,32)
        # self.fc4 = nn.Linear(32,1)

        self.score_arrays = []

    def forward(self, x, features):
        # No training for VGG16 feature extraction
        with torch.no_grad():
            # Extract dimensions of x
            batch_size, timesteps, C, H, W = x.size()
            #Initialise empty torch matrix
            frameFeatures = torch.empty(size=(batch_size, timesteps, 512*7*7)).to(device)

            # This step is only important for the LSTM. It creates a stack of t timesteps
            # In this case it is not needed.
            for t in range(0, x.size()[1]):
                frame = x[:, t, :, :, :]
                frame_feature = self.features(frame)
                # print("Size frame feature", frame_feature.size())
                frame_feature = frame_feature.view(-1, 25088)
                # print(frame_feature.shape)
                frameFeatures[:, t, :] = frame_feature

        # Get last timestep for fully connected layer (only relevant when using LSTM)
        last_timestep = frameFeatures[:,-1,:]

        # Decode hidden state of last time step
        c_out = torch.cat((features, last_timestep), dim = 1)

        # Fully connected layers
        f1_out = F.relu(self.fc1(c_out))
        f2_out = F.relu(self.fc2(f1_out))

        # Squeeze output for prediction with y_true
        return f2_out.squeeze()

 
    def train_model(self, trainloader, validationloader, model, optimizer, num_epochs, checkpoint_epoch):
        """
        Train a PyTorch model and print model performance metrics.

        Parameters
        ----------
        trainloader and validationloader: split dataset in train and val set 
        model : CNN with two fully connected layers
        optimizer : Adam
        num_epochs : number of epochs to train by
        checkpoint_epoch: if there's a checkpoint, train from there
        """
        # Set training to true
        model.train(True)
        # Store the epoch losses in a list for plotting
        self.epoch_loss = []
        # Best validation loss
        best_vloss = 100000.0


        for epoch in range(checkpoint_epoch, num_epochs):  # loop over the dataset multiple times
            print("Epoch:", epoch)
            # Initialise losses
            running_loss = 0.0
            last_loss = 0.0
            # Number of batches per epoch
            n_batches = len(trainloader)
            score_arrays= [] # add the field for appending the loss
            for i, data in enumerate(trainloader, 0):
                print("----------------------------------")
                print("Minibatch:", i)
                # get the inputs, features and labels
                inputs, features, labels = data
                # Read them to device [cpu, gpu]
                inputs = inputs.to(device)
                features = features.to(device)
                labels = labels.to(device)

                # print("Labels:", labels.size())
                # print("Inputs:", inputs.size())

                # forward + backward + optimize
                outputs = model(inputs, features)
                loss = self.weighted_loss(outputs, labels)
                loss.backward()
                optimizer.step()
                print("Loss:", loss)

                # Add loss to running loss per epoch
                running_loss += loss.item()
                if i % 10 == 0:    # print every 10 mini-batches
                    last_loss = running_loss / 10
                    print('  batch {} loss: {}'.format(i + 1, last_loss))
                self.score_arrays.append(last_loss) # append the loss

                # Print statistics
                with open("only_cnn/minibatch_losses.txt", "a") as f:
                    f.write(f"Epoch {epoch + 1}, Minibatch: {i} Loss: {loss}\n")  # Writing each loss value on a separate line

            # PER EPOCH
            # Add checkpoint
            self.checkpoint(model, f"epoch_onlycnn/epoch-{epoch}.pth")

            # Print loss per epoch and store result
            print("Epoch loss:", running_loss/n_batches)
            self.epoch_loss.append(running_loss/n_batches) 

            # Set the model to evaluation mode
            model.eval()
            running_vloss = 0.0
            with torch.no_grad():
                for i, vdata in enumerate(validationloader):
                    # Read in data
                    vinputs, vfeatures, vlabels = vdata
                    # Put to device
                    vinputs = vinputs.to(device)
                    vfeatures = vfeatures.to(device)
                    vlabels = vlabels.to(device)
                    # Predict output
                    voutputs = model(vinputs, vfeatures)
                    # Calculate loss
                    vloss = self.weighted_loss(voutputs, vlabels)
                    # Append to runningvloss
                    running_vloss += vloss
            #Print vloss
            avg_vloss = running_vloss / (i + 1)
            print(f"Epoch {epoch + 1}, Training loss: {running_loss/n_batches} Valid loss: {best_vloss} \n")

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                # model_path = f"epoch_onlycnn/epoch-{epoch}.pth"
                # model_path = 'model_{}_{}'.format(timestamp, epoch_number)
                # torch.save(model.state_dict(), model_path)
                            # print statistics

            # Write final statistics to loss txt file
            with open("only_cnn/epoch_loss.txt", "a") as f:
                f.write(f"---------------\n Epoch {epoch + 1}, Training loss: {running_loss/n_batches} Valid loss: {best_vloss} \n")  # Writing each loss value on a separate line

    
    def evaluate_model(self, dataloader, model, num_epochs):
        """
        Evaluate a PyTorch regression model and print model performance metrics.

        Parameters
        ----------
        dataloader : testd data
        model : CNN
        """
        model.eval()
        # Initialise torch R2Score class
        r2score = R2Score().to(device)
        # Initialise arrays
        y_predlist = []
        ylist = []
        # Since we do not want to train the model, make sure that we deactivate the gradients
        with torch.no_grad():
            # Initialize empty lists for metrics
            metrics = {'mse': [], 'mae': [], 'r2': [], 'rmse': []} 

            # Loop through the dataloader
            for x, features, y in dataloader:
                # Move to device
                x = x.to(device)
                features = features.to(device)
                y = y.to(device)

                # make predictions using the model
                y_pred = model(x, features) 

                # Append to list
                ylist.append(y)
                y_predlist.append(y_pred)

                # Compute evaluation metrics
                mse = torch.mean((y_pred - y)**2).item() # mean squared error
                mae = torch.mean(torch.abs(y_pred - y)).item() # mean absolute error
                rmse = torch.sqrt(torch.tensor(mse))

                # Append the evaluation metrics to the lists
                metrics['mse'].append(mse)
                metrics['mae'].append(mae)
                metrics['rmse'].append(rmse)
            # Change to tensor    
            y_pred_tensor = torch.cat(y_predlist, dim=0)
            y_tensor = torch.cat(ylist, dim=0)

            # Calculate scores
            r2 = r2score(y_pred_tensor, y_tensor)
            metrics['r2'].append(r2)


            # Print the averaged evaluation metrics
            with open("only_cnn/only_cnn_metrics.txt", "w") as f:
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
            if metric_name != 'r2':
                print("Averaged %s: %.4f" % (metric_name.upper(), np.mean(values)))

        #---------------------PLOTTING----------------------------#
        # Plot MSE
        # plt.plot(range(1,len(metrics['mse'])+1), metrics['mse'], label='MSE', color='blue')
        # # Plot MAE
        # plt.plot(range(1,len(metrics['mae'])+1), metrics['mae'], label='MAE', color='red')

        # # Add labels and title
        # plt.title('MSE and MAE Over Epochs')
        # plt.xlabel('Minibatch')
        # plt.ylabel('Loss')
        # plt.legend()
        # plt.grid(True)
        # name = 'weighted_loss.png'
        # plt.savefig('only_cnn/'+name)
        # print(f"{name} plot updated")
        # plt.show()
        print("LEN SCORE ARRAYS", len(self.score_arrays))
        #-----------------PRINT SCORE ARRAY LOSS --------------------
        plt.plot(range(1, len(self.epoch_loss)+1), self.epoch_loss)
        plt.title('Loss over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig("only_cnn/only_cnn_loss_plot.png")
        plt.show()

        
    def checkpoint(self, model, filename):
        """
        Retrieves checkpoints of epochs
        """
        torch.save({
            'optimizer': optimizer.state_dict(),
            'model': model.state_dict(),
        }, filename)
    
    def resume(self, model, filename): 
        """
        Resumes checkpoint of epochs
        """
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    def weighted_loss(self, output, target):
        """
        Custom loss function which uses histogram smoothing class to give
        weights to certain bins of the target distribution.
        High PM2.5: More weight
        Low PM2.5: Lower weight
        """
        output_cpu = output.cpu().detach().numpy()

        s = smoothing(10,2)
        weights, bins = s.smoothingfunction()

        weights_tensor = torch.tensor(weights)
        weights_tensor_cuda = weights_tensor.to(output.device)


        index = np.digitize(output_cpu, bins)

        weight = weights_tensor_cuda[index]
        loss = torch.mean(weight*(output - target)**2)
        return loss
    


# Create transforms
transform = transforms.Compose([
    transforms.Resize(size = (224,224))
   , transforms.ToTensor()]) 
target_transform = transforms.Compose([
    transforms.ToTensor()]) 

# Set regression to true or false for task
regression = True
print("REGRESSION:", regression)

# Create dataset
cD = createWeatherDataset(image_path='2024-03-12/',
                                    sensor_file='12032024_sensor.csv', transform = transform, target_transform=None, regression=regression)

# Make indexes of dataset
dataset_indices = list(range(len(cD)))

# Create indexes
train_idx = int(0.7*len(cD))
val_idx = int(0.1*len(cD))

# Retrieve sets
train_set = dataset_indices[:train_idx]
val_set = dataset_indices[train_idx:train_idx+val_idx]
test_set = dataset_indices[train_idx+val_idx:]

# Make sample
train_sampler = torch.utils.data.SequentialSampler(train_set)
val_sampler = torch.utils.data.SequentialSampler(val_set)
test_sampler = torch.utils.data.SequentialSampler(test_set)
# train_set, val_set, test_set = torch.utils.data.random_split(cD, [int(0.8*len(cD)), len(cD) - int(0.8*len(cD))])
# train_set[0][0].size()

# Load data
trainloader = DataLoader(cD, batch_size=32,
                        shuffle=False, num_workers=0, sampler = train_sampler)
validationloader = DataLoader(cD, batch_size=32,
                        shuffle=False, num_workers=0, sampler = val_sampler)
testloader = DataLoader(cD, batch_size=32,
                        shuffle=False, num_workers=0, sampler = test_sampler)

print("Success with dataloading")

model = CNN_WEATHER(512*7*7 , 2, 1, regression = regression).to(device)
print("MODEL", model)

 #Define the path to the saved model state dictionary
# path_to_saved_model = "epoch_onlycnn/epoch-24.pth"

# Load the model state dictionary
# checkpoint = torch.load(path_to_saved_model, map_location=torch.device('cpu'))

# Load weights into the model
# model.load_state_dict(checkpoint['model'])
# Optionally, move the model to the GPU if available
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


learning_rate = 1e-5
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
checkpoint_epoch = 24
epochs = 25

model.train_model(trainloader, validationloader, model, optimizer, epochs, checkpoint_epoch)
model.evaluate_model(testloader, model, epochs)

