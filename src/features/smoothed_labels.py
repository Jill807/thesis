import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import numpy as np
import pandas as pd

# Histogram smoothing function

class label_smoothing():
    def __init__(self, labels, lookback_window = 48):
        self.labels    = labels
        self.lookback_window = lookback_window


    def label_smooth(self):
        self.smoothed_labels = []
        for idx in range(self.lookback_window, len(self.labels)):
            std_idx      = np.std(labels.iloc[idx - self.lookback_window:idx])
            mean_idx     = np.mean(labels.iloc[idx - self.lookback_window:idx])
            self.smoothed_labels.append((labels[idx]-mean_idx)/std_idx)
        print(len(self.smoothed_labels))
        return self.smoothed_labels

    def visualise_distribution(self):
        pass
        fig = plt.figure(figsize = (16,4))

        # Plot PM2.5 distribution
        plt.subplot(1, 2, 1)
        plt.plot(self.labels)
        plt.title("Distribution of PM2.5 values")
        plt.savefig('src/visualisations/pm25histogram.png')

        # Plot original and smoothed histograms
        plt.subplot(1, 2, 2)
        plt.plot(self.smoothed_labels, color='red', label='Smoothed Histogram')
        plt.xlabel('Bins')
        plt.ylabel('Counts')
        plt.title('Distribution of PM2.5 values with smoothed histogram')
        plt.legend()
        plt.savefig('src/visualisations/ls.png')
        fig.show()


df                  = pd.read_parquet("src/data/pca_features.parquet", engine = "fastparquet")
labels              = df.loc[:, "PM2.5"]
ls                  = label_smoothing(labels)
smoothed_labels     = ls.label_smooth()
ls.visualise_distribution()
