import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np


# Read in data
df = pd.read_parquet("src/data/cnn_features.parquet", engine="fastparquet")

# Split into features and labels
features     = df.iloc[:, :-9]
aux_features = df.iloc[:, -9:-1]
labels       = df.iloc[:, -1]

# Get train, validation and test indexes
pct_train  = 0.7
pct_val    = 0.2
pct_test   = 0.1
nrow_df    = len(df)

nrow_train = int(nrow_df * pct_train)
nrow_dev   = int(nrow_df * pct_val)
nrow_test  = nrow_df - (nrow_train + nrow_dev)

train_X   = features[:nrow_train]
train_y   = labels[:nrow_train]

# Scale data
scaler    = StandardScaler()
scaler.fit(train_X) 
train_X_scaled = scaler.transform(train_X)

# Apply PCA
pca_model = PCA()
pca_model.fit(train_X_scaled)

explained_variance_ratios = pca_model.explained_variance_ratio_
cumul_evr                 = np.cumsum(explained_variance_ratios)

pca_features      = pca_model.transform(X = features)
nr_pca            = pca_features.shape[1]         #905
pca_col_names     = ["PC" + str(x) for x in range(1, nr_pca+1)]
df_pca            = pd.DataFrame(data = pca_features, columns = pca_col_names)
df_pca_complete   = pd.concat([df_pca, aux_features, labels], axis = 1)

df_pca_complete  = df_pca_complete.iloc[:-2, :]
df_pca_complete.to_parquet("src/data/pca_features.parquet", engine = "fastparquet") 




