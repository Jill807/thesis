import sys
import os
sys.path.append(os.getcwd())

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn               import linear_model as LinearModel
import numpy as np
import torch.nn as nn
import torch
from copy import deepcopy
from torchmetrics.regression import R2Score
from src.features.smoothed_labels import label_smoothing
# import seaborn as sns

# from pytorchtools import EarlyStopping



warnings.filterwarnings('ignore')
device = "cuda" if torch.cuda.is_available() else "cpu"

class weighted_lstm_model(nn.Module):
    def __init__(self, nr_input_features, timestamps, lstm_hidden_size,
                 fc_layer_size, activation_fn, saved_model_path):
        super().__init__()
        self.lstm = nn.LSTM(input_size = nr_input_features,
                            hidden_size=lstm_hidden_size, num_layers=1,
                            batch_first=True)
        self.fc1          = nn.Linear(in_features = lstm_hidden_size, out_features = fc_layer_size)
        self.output_layer = nn.Linear(in_features = fc_layer_size, out_features = 1)
        self.activation   = activation_fn
        self.saved_params_path = saved_model_path

    def forward(self, x):
        x, _ = self.lstm(x)

        # Get last timestep
        last_timestep = x[:, -1, :] 
        # Fully connected layers
        fc1_out = self.activation(self.fc1(last_timestep))
        output  = self.output_layer(fc1_out)
        return output

    def save_model(self, path = None):
      if path == None:
        path = self.saved_params_path
      torch.save(deepcopy(self.state_dict()), self.saved_params_path)

    def load_model(self, path):
      self.load_state_dict(torch.load(path))

    def train_model(self, model, n_epochs, train_X, train_y, 
                    train_w, val_X, val_y, val_w, test_X, test_y, 
                    test_w, optimizer, loss_fn, metrics):
        best_val_mse      = np.inf
        best_train_mse    = np.inf
        eval_frequency    = 20
        patience_counter  = 5
        patience          = 0

        for epoch in range(n_epochs):
            model.train()
            y_pred = model(train_X)
            loss = self.weighted_loss(y_pred, train_y, train_w)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Early stopping
            patience = self.early_stopping(loss, best_train_mse, patience, min_delta = 0.05)
            if patience >= patience_counter:
              print(f"Early stopping epoch: {epoch}")
              break

            # Best Train MSE
            if loss < best_train_mse:
                  best_train_mse = loss
        
            # Validation
            if epoch % eval_frequency != 0:
                continue
            model.eval()
            with torch.no_grad():
                # Train
                y_pred     = model(train_X)
                train_mse  = self.weighted_loss(y_pred, train_y, train_w)
                train_r2   = self.weighted_r2(train_mse, train_y, train_w)
                # Validation
                y_pred     = model(val_X)
                val_mse    = self.weighted_loss(y_pred, val_y, val_w)
                val_r2     = self.weighted_r2(val_mse, val_y, val_w)
                # Test
                y_pred     = model(test_X)
                test_mse   = self.weighted_loss(y_pred, test_y, test_w)
                test_r2    = self.weighted_r2(test_mse, test_y, test_w)

                print("Epoch %d: train RMSE %.4f, val RMSE %.4f" % (epoch, np.sqrt(train_mse), np.sqrt(val_mse)))
                metrics["epoch"].append(epoch)

                # Train
                metrics["train_mse"].append(train_mse)
                metrics["train_r2"].append(train_r2)
                #Validation
                metrics["val_mse"].append(val_mse)
                metrics["val_r2"].append(val_r2)
                # Test
                metrics["test_mse"].append(test_mse)
                metrics["test_r2"].append(test_r2)
                # if this model obtains the best val MSE, save it
                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    model.save_model()
                print("Saved")
        return model, metrics
    
    def test_model(self, model, test_X, test_y, test_w):
        # Set to eval model
        model.eval()

        # Predict on test set
        pred_test = model(test_X)

        # Generate metrics
        test_rmse = np.sqrt(self.weighted_loss(pred_test, test_y, test_w).detach().numpy())      
        test_mae  = nn.L1Loss(pred_test, test_y, test_w).detach().numpy()
        test_mse  = self.weighted_loss(pred_test, test_y, test_w)
        test_r2   = self.weighted_r2(test_mse, test_y, test_w)

        # Print metrics
        print("-"*40)
        print(f"Weighted: test RMSE {test_rmse}, test MAE {test_mae}, test R2 {test_r2}")
        with open("src/data/metrics.txt", "a") as f:
            f.write(f"Weighted: test RMSE {test_rmse}, test MAE {test_mae}, test R2 {test_r2}\n")

        return np.sqrt(test_rmse), test_mae, test_r2
    
    def early_stopping(self, train_loss, best_train_loss, patience, min_delta = 0.1):
          # Early stopping check
        if train_loss < best_train_loss - min_delta:
            patience += 0
        else:
            patience += 1
        return patience
    def weighted_loss(self, output, target, weight):
      """
      Custom loss function which uses histogram smoothing class to give
      weights to certain bins of the target distribution.
      High PM2.5: More weight
      Low PM2.5: Lower weight
      """
      return torch.mean(weight*(output-target)**2)
    
    def weighted_r2(self, mse, y, weights):
      y         = y.detach().numpy()
      mse       = mse.detach().numpy()  
      weights   = weights.detach().numpy()

      weights   = weights[:, 0]
      variance  = np.cov(y[:, 0], aweights = weights)
      r2        = 1 - mse / variance 
      return r2




df = pd.read_parquet("src/data/pca_features.parquet", engine = 'fastparquet')
df = df.iloc[1:, :].reset_index()

feature_columns = df.columns[:-2].values.tolist()
label_column    = "PM2.5"
lookback_window = 48
X  = df.iloc[1:, :-2]
y  = df.loc[1:, "PM2.5"]
w  = df.loc[1:, "WEIGHTED_PM25"]

labels              = df.loc[:, "PM2.5"]
ls                  = label_smoothing(labels, lookback_window)
smooth_y            = ls.label_smooth()

# Train, val, test split
pct_train = 0.7
pct_val   = 0.2
pct_test  = 0.1
train_set = int(len(df)*pct_train)
val_set   = int(len(df)*pct_val)
test_set  = train_set + val_set
print(test_set)

# Retrieve sets
train_X   = X[:train_set]
train_y   = y[:train_set]

scaler = StandardScaler()
scaler.fit(train_X)
X_normalised = scaler.transform(X)
df_X_normalised = pd.DataFrame(data = X_normalised, columns = feature_columns)

np_X = df_X_normalised[feature_columns].values
np_y = df["PM2.5"].values
np_w = df["WEIGHTED_PM25"].values

# np_y = smooth_y

val_X     = X[train_set:train_set+val_set]
val_y     = y[train_set:train_set+val_set]
test_X    = X[test_set:]
test_y    = y[test_set:]


nr_obs, nr_features = np_X.shape

# model specifications
TIMESTAMPS       = 6
LSTM_HIDDEN_SIZE = 64
FC_LAYER_SIZE    = 32
ACTIVATION_FN    = nn.Tanh()
LEARN_RATE       = 0.001
best_model_filename = "epoch/" + "model1.pth"

# set up data
X                = np.zeros(shape = (nr_obs - TIMESTAMPS + 1, TIMESTAMPS, nr_features))
y                = np.zeros(shape = (nr_obs - TIMESTAMPS + 1, 1))
w                = np.zeros(shape = (nr_obs - TIMESTAMPS + 1, 1))

for idx in range(TIMESTAMPS-1, nr_obs, 1):
  feature = np_X[idx - TIMESTAMPS+1:idx+1, :]
  target  = np_y[idx]
  weight  = np_w[idx]
  X[idx - TIMESTAMPS+1, :, :] = feature
  y[idx - TIMESTAMPS+1:]      = target
  w[idx - TIMESTAMPS+1:]      = weight

tensor_X = torch.tensor(X, dtype= torch.float32)
tensor_y = torch.tensor(y, dtype= torch.float32)
tensor_w = torch.tensor(w, dtype = torch.float32)

train_X  = tensor_X[:train_set - TIMESTAMPS+1, :, :]
train_y  = tensor_y[:train_set - TIMESTAMPS+1, :]
train_w  = tensor_w[:train_set - TIMESTAMPS+1, :]

val_X    = tensor_X[train_set - TIMESTAMPS+1: train_set + val_set - TIMESTAMPS +1, :, :]
val_y    = tensor_y[train_set - TIMESTAMPS+1: train_set + val_set - TIMESTAMPS +1, :]
val_w    = tensor_w[train_set - TIMESTAMPS+1: train_set + val_set - TIMESTAMPS +1, :]

test_X   = tensor_X[test_set - TIMESTAMPS+1:, :, :]
test_y   = tensor_y[test_set - TIMESTAMPS+1:, :]
test_w   = tensor_w[test_set - TIMESTAMPS+1:, :]


model = weighted_lstm_model(nr_input_features = nr_features, timestamps = TIMESTAMPS,
                    lstm_hidden_size = LSTM_HIDDEN_SIZE, fc_layer_size = FC_LAYER_SIZE,
                    activation_fn = ACTIVATION_FN, saved_model_path = best_model_filename)

optimizer = torch.optim.Adam(params = model.parameters(), lr = LEARN_RATE)
loss_fn   = nn.MSELoss()
loss_mae  = nn.L1Loss()


metrics = {"epoch": [], "train_mse":[], "train_mae": [], "train_r2": [], "val_mse":[], "val_mae": [], "val_r2":[], "test_mse":[], "test_r2":[]}
n_epochs       = 1600
r2score = R2Score()

model, metrics = model.train_model(model, n_epochs, train_X, train_y, 
                                   train_w, val_X, val_y, val_w, test_X, 
                                   test_y, test_w, optimizer, loss_fn, metrics)
model.test_model(model, test_X, test_y, test_w)


# plot eval history

var_y_train = np.var(train_y.numpy())
var_y_dev   = np.var(val_y.numpy())


# r squared = 1 - mse/var_y

# get r squared values
train_mses       = metrics["train_mse"]
train_r_squared  = [1 - x / var_y_train for x in train_mses]

val_mses         = metrics["val_mse"]
val_r_squared    = [1 - x / var_y_dev for x in val_mses]


# sns.set()
fig = plt.figure(figsize=(14, 6))
plt.plot(metrics["epoch"], metrics["train_r2"], "-bo", color = 'b', label = 'Training')
plt.plot(metrics["epoch"], metrics["val_r2"], "-bo", color = 'r', label = 'Validation')
plt.plot(metrics["epoch"], metrics["test_r2"], "-bo", color = 'green', label = 'Test')
plt.title("R squared on the training, validation and test set")
plt.xlabel("Epoch")
plt.ylabel("R Squared")
plt.legend()
plt.grid()
plt.savefig("src/visualisations/Weighted_TrainingAndValidationLoss.png")
fig.show()


# print(metrics['val_r2'])
# print(metrics['train_r2'])



