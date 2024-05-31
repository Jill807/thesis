
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

        # LSTM model
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layer_dim,
            batch_first=True)
        # print(self.)
        # Fully connected layers with one output for regression
        print("INPUT DIM + 6", input_dim+6)
        self.fc1 = nn.Linear(input_dim+6, 32)
        self.fc2 = nn.Linear(32,1)
        # self.fc3 = nn.Linear(128,32)
        # self.fc4 = nn.Linear(32,1)

        # If regression is true, then we change the forward pass
        self.regression = regression

        self.score_arrays = []

    def forward(self, x, features):
        # No training for VGG16 feature extraction
        with torch.no_grad():
            # Extract dimensions of x
            batch_size, timesteps, C, H, W = x.size()
            frameFeatures = torch.empty(size=(batch_size, timesteps, 512*7*7)).to(device)


            for t in range(0, x.size()[1]):
                frame = x[:, t, :, :, :]
                frame_feature = self.features(frame)
                # print("Size frame feature", frame_feature.size())
                frame_feature = frame_feature.view(-1, 25088)
                # print(frame_feature.shape)
                frameFeatures[:, t, :] = frame_feature

        last_timestep = frameFeatures[:,-1,:]

        # Decode hidden state of last time step
        c_out = torch.cat((features, last_timestep), dim = 1)

        # Fully connected layers
        f1_out = F.relu(self.fc1(c_out))
        f2_out = F.relu(self.fc2(f1_out))


        return f2_out.squeeze()

 
    def train_model(self, trainloader, validationloader, model, optimizer, num_epochs, checkpoint_epoch):
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
        output_file = "only_cnn/epoch_loss.txt"

        best_vloss = 100000.0


        for epoch in range(checkpoint_epoch, num_epochs):  # loop over the dataset multiple times
            print("Epoch:", epoch)
            running_loss = 0.0
            last_loss = 0.0
            n_batches = len(trainloader)
            score_arrays= [] # add the field for appending the loss
            for i, data in enumerate(trainloader, 0):
                print("----------------------------------")
                print("Minibatch:", i)
                # get the inputs; data is a list of [inputs, labels]
                inputs, features, labels = data
                inputs = inputs.to(device)
                features = features.to(device)
                labels = labels.to(device)

                # print("Labels:", labels.size())
                # print("Inputs:", inputs.size())

                # forward + backward + optimize
                outputs = model(inputs, features)
                # print("Output size:",x outputs.size())
                # print("Outputs:", outputs, labels.dtype)
                loss = self.weighted_loss(outputs, labels)
                print("Loss:", loss)
                loss.backward()
                optimizer.step()


                running_loss += loss.item()
                if i % 10 == 0:    # print every 100 mini-batches
                    last_loss = running_loss / 10
                    print('  batch {} loss: {}'.format(i + 1, last_loss))
                    # print('[%d, %5d] loss: %.3f' %
                        # (epoch + 1, i + 1, running_loss / 100))
                self.score_arrays.append(last_loss) # append the loss

                # print statistics
                with open("only_cnn/minibatch_losses.txt", "a") as f:
                    f.write(f"Epoch {epoch + 1}, Minibatch: {i} Loss: {loss}\n")  # Writing each loss value on a separate line
            # PER EPOCH
            self.checkpoint(model, f"epoch_onlycnn/epoch-{epoch}.pth")

            print("Epoch loss:", running_loss/n_batches)

            running_vloss = 0.0
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            model.eval()

        # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i, vdata in enumerate(validationloader):
                    vinputs, vfeatures, vlabels = vdata
                    vinputs = vinputs.to(device)
                    vfeatures = vfeatures.to(device)
                    vlabels = vlabels.to(device)
                    voutputs = model(vinputs, vfeatures)
                    vloss = self.weighted_loss(voutputs, vlabels)
                    running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            print('LOSS training: {} || valididation: {}'.format(last_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        # writer.add_scalars('Training vs. Validation Loss',
        #                 { 'Training' : last_loss, 'Validation' : avg_vloss },
        #                 epoch_number + 1)
        # writer.flush()

        # Track best performance, and save the model's state

            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = f"epoch_onlycnn/epoch-{epoch}.pth"
                # model_path = 'model_{}_{}'.format(timestamp, epoch_number)
                torch.save(model.state_dict(), model_path)
                            # print statistics
            
            with open(output_file, "a") as f:
                f.write(f"---------------\n Epoch {epoch + 1}, Training loss: {running_loss/n_batches} Valid loss: {best_vloss} \n")  # Writing each loss value on a separate line

    
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
                                
            y_pred_tensor = torch.cat(y_predlist, dim=0)
            y_tensor = torch.cat(ylist, dim=0)

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
        plt.plot(range(1, len(self.score_arrays)+1), self.score_arrays)
        plt.title('Loss over epochs')
        plt.xlabel('Minibatch')
        plt.ylabel('Loss')
        plt.savefig("only_cnn/only_cnn_loss_plot.png")
        plt.show()

        
    def checkpoint(self, model, filename):
        torch.save({
            'optimizer': optimizer.state_dict(),
            'model': model.state_dict(),
        }, filename)
    
    def resume(self, model, filename): 
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    def weighted_loss(self, output, target):
        """
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
trainloader = DataLoader(cD, batch_size=64,
                        shuffle=False, num_workers=0, sampler = train_sampler)
validationloader = DataLoader(cD, batch_size=64,
                        shuffle=False, num_workers=0, sampler = val_sampler)
testloader = DataLoader(cD, batch_size=64,
                        shuffle=False, num_workers=0, sampler = test_sampler)

print("Success with dataloading")

model = CNN_WEATHER(512*7*7 , 2, 1, regression = regression).to(device)
print("MODEL", model)

 #Define the path to the saved model state dictionary
# path_to_saved_model = "epoch_onlycnn/epoch-2.pth"  # Replace with the actual path to your saved model

# Load the model state dictionary
# checkpoint = torch.load(path_to_saved_model, map_location=torch.device('cpu'))

# Load weights into the model
# model.load_state_dict(checkpoint['model'])
# Optionally, move the model to the GPU if available
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


learning_rate = 1e-5
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
checkpoint_epoch = 0
epochs = 8

model.train_model(trainloader, validationloader, model, optimizer, epochs, checkpoint_epoch)
model.evaluate_model(testloader, model, epochs)

