
import sys
import os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
from src.data.dataset import createDataset
from src.data.weather_dataset2 import createWeatherDataset
from torchvision import transforms
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.regression import R2Score
import torch.nn.functional as F
from src.features.histogramsmoothing import smoothing
from sklearn.metrics import confusion_matrix
import torch.optim.lr_scheduler as lr_scheduler
import pandas as pd
from torch.utils.data import Dataset
import warnings




warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"

class LIGHT_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(25089, 128)
        self.fc2 = nn.Linear(128,1)

        self.score_arrays = []

    def forward(self, cnn_features, weatherfeatures, lagfeatures):

        # Fully connected layers
        f1_out = F.relu(self.fc1(cnn_features))
        f2_out = F.relu(self.fc2(f1_out))

        # Squeeze output for prediction with y_true
        return f2_out.squeeze()
    
 
    def train_model(self, trainloader, validationloader, model, optimizer, num_epochs, checkpoint_epoch, scheduler, learning_rate):
        # Store the epoch losses in a list for plotting
        self.epoch_loss = []
        # Best validation loss
        best_vloss = 100000.0
        loss_fn = nn.MSELoss()

        for epoch in range(checkpoint_epoch, num_epochs):
            print("Epoch:", epoch)
            # Initialise losses
            running_loss = 0.0
            last_loss = 0.0
            # Number of batches per epoch
            n_batches = len(trainloader)
            score_arrays= []
            # Set training to true
            model.train(True)

            for i, data in enumerate(trainloader, 0):
                print("----------------------------------")
                print("Minibatch:", i)

                # Get the inputs, features and labels
                cnn_features, weatherfeatures, lagfeatures, labels = data

                # Read them to device [cpu, gpu]
                cnn_features = cnn_features.to(device)
                weatherfeatures = weatherfeatures.to(device)
                lagfeatures = lagfeatures.to(device)
                labels = labels.to(device)

                # forward + backward + optimize
                outputs = model(cnn_features, weatherfeatures, lagfeatures)
                loss = loss_fn(outputs, labels)
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
                with open("lightcnn/minibatch_losses.txt", "a") as f:
                    f.write(f"---------------\nEpoch {epoch + 1}, Minibatch: {i} Loss: {loss}\n")  # Writing each loss value on a separate line

            # PER EPOCH
            # Add checkpoint
            self.checkpoint(model, f"epoch_lightcnn/epoch-1.pth")

            # Print loss per epoch and store result
            print("Epoch loss:", running_loss/n_batches)
            self.epoch_loss.append(running_loss/n_batches) 

            # Set the model to evaluation mode
            model.eval()
            running_vloss = 0.0
            with torch.no_grad():
                for i, vdata in enumerate(validationloader):
                    # Read in data
                    vinputs, vlagfeatures, vlabels = vdata
                    # Put to device
                    vinputs = vinputs.to(device)
                    vweatherfeatures = vweatherfeatures.to(device)
                    vlagfeatures = vlagfeatures.to(device)
                    vlabels = vlabels.to(device)
                    # Predict output
                    voutputs = model(vinputs, vweatherfeatures, vlagfeatures)
                    # Calculate loss
                    vloss = self.weighted_loss(voutputs, vlabels)
                    # Append to runningvloss
                    running_vloss += vloss

            # Take step of scheduler
            scheduler.step()
            last_lr = scheduler.get_last_lr()
            #Print vloss
            avg_vloss = running_vloss / (i + 1)
            print(f"Epoch {epoch + 1}, Training loss: {running_loss/n_batches} Valid loss: {best_vloss} Learning Rate: {last_lr}\n")



            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                # model_path = f"epoch_lightcnn/epoch-{epoch}.pth"
                # model_path = 'model_{}_{}'.format(timestamp, epoch_number)
                # torch.save(model.state_dict(), model_path)
                            # print statistics
            # Write final statistics to loss txt file
            with open("lightcnn/epoch_loss.txt", "a") as f:
                f.write(f"---------------\n Epoch {epoch + 1}, Training loss: {running_loss/n_batches} Valid loss: {best_vloss} Learning Rate: {last_lr}\n")  # Writing each loss value on a separate line

    
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
            for x, weatherfeatures, lagfeatures, y in dataloader:
                # Move to device
                x = x.to(device)
                weatherfeatures = weatherfeatures.to(device)
                lagfeatures = lagfeatures.to(device)
                y = y.to(device)

                # make predictions using the model
                y_pred = model(x, weatherfeatures, lagfeatures) 

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
            with open("lightcnn/metrics.txt", "w") as f:
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
        # plt.savefig('lightcnn/'+name)
        # print(f"{name} plot updated")
        # plt.show()
        print("LEN SCORE ARRAYS", len(self.score_arrays))
        #-----------------PRINT SCORE ARRAY LOSS --------------------
        plt.plot(range(1, len(self.epoch_loss)+1), self.epoch_loss)
        plt.title('Loss over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig("lightcnn/loss_plot.png")
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

class LIGHT_DATASET(Dataset):
    def __init__(self, transform):
        self.df = self.retrieve_data()
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        cnn_features = torch.tensor(self.df.iloc[idx, :25089].values, dtype=torch.float32)
        lag_features = torch.tensor(self.df.loc[idx,['lag1', 'lag2']].values, dtype=torch.float32)
        weather_features = torch.tensor(self.df.loc[idx, ['FH','SQ','Q','FX', 'SIN_HOUR', 'COS_HOUR']])
        labels = torch.tensor(self.df.loc[idx,'PM2.5'], dtype= torch.float32)

        return cnn_features, weather_features, lag_features, labels
    
    def retrieve_data(self):
        df1 = pd.read_csv('src/data/all_features_part1.csv')
        df2 = pd.read_csv('src/data/all_features_part2.csv')
        labels = pd.read_csv('src/data/all_labels.csv')
        weatherfeatures = pd.read_csv('src/data/all_weatherfeatures.csv')

        concatenated = pd.concat([df1, df2, labels, weatherfeatures], axis = 1)

        concatenated['lag1'] = concatenated['PM2.5'].shift(1)
        concatenated['lag2'] = concatenated['PM2.5'].shift(2)
        concatenated = concatenated.dropna().reset_index()
        return concatenated


# Create transforms
transform = transforms.Compose([
    transforms.Resize(size = (224,224))
   , transforms.ToTensor()]) 
target_transform = transforms.Compose([
    transforms.ToTensor()]) 

# # Set regression to true or false for task
regression = True
print("REGRESSION:", regression)

# # Create dataset
cD = LIGHT_DATASET(transform)

# # Make indexes of dataset
dataset_indices = list(range(len(cD)))

# # Create indexes
train_idx = int(0.7*len(cD))
val_idx = int(0.1*len(cD))

# # Retrieve sets
train_set = dataset_indices[:train_idx]
val_set = dataset_indices[train_idx:train_idx+val_idx]
test_set = dataset_indices[train_idx+val_idx:]

# # Make sample
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
model = LIGHT_CNN().to(device)
print("MODEL", model)

 #Define the path to the saved model state dictionary
# path_to_saved_model = "epoch_lightcnn/epoch-0.pth"

# Load the model state dictionary
# checkpoint = torch.load(path_to_saved_model, map_location=torch.device('cpu'))
# checkpoint = torch.load(path_to_saved_model)


# Load weights into the model
# model.load_state_dict(checkpoint['model'])
# Optionally, move the model to the GPU if available
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)


learning_rate = 0.001

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.MultiStepLR(optimizer = optimizer, milestones=[12, 27], gamma = 0.5)
checkpoint_epoch = 0
epochs = 1

model.train_model(trainloader, validationloader, model, optimizer, epochs, checkpoint_epoch, scheduler, learning_rate)
model.evaluate_model(testloader, model, epochs)



