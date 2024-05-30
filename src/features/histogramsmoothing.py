
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

class smoothing:
    def __init__(self, bw, sigma):
        self.data = pd.read_parquet('labelled_dataset.parquet')
        self.bw = bw
        self.sigma =sigma


    def smoothingfunction(self):
        """
        """
        labels = self.data['pm25']

        #Make histogram
        bins = np.linspace(min(labels), max(labels), self.bw)
        histogram_counts, _ = np.histogram(labels, bins=bins)

        # CLip histogram: ADJUSTTTT
        histogram_clipped = np.minimum(histogram_counts, 500)


        # Gaussian filter
        smoothed_hs = gaussian_filter1d(histogram_clipped, self.sigma)

        k = len(smoothed_hs)/np.sum(1/smoothed_hs)

        weights = k*(1/smoothed_hs)
        print(weights)

        # Fit histogram to gaussian distribution 
        # smoothed_histogram = np.convolve(histogram_clipped, np.ones(s), mode='same')
        # smoothed_histogram = np.convolve(smoothed_histogram, np.ones(s), mode='same')
        #Plot original histogram
      
        # plt.hist(self.data['pm25'], bins = 50)
        # plt.savefig('originalhistogram.png')
        # plt.show()
        # # Plot original and smoothed histograms
        # plt.bar(bins[:-1], histogram_clipped, width = 2, color='blue', alpha=0.5, label='Original Histogram')
        # plt.plot(bins[:-1], smoothed_hs, color='red', label='Smoothed Histogram')
        # plt.xlabel('Bins')
        # plt.ylabel('Counts')
        # plt.title('Histogram Smoothing')
        # plt.legend()
        # plt.savefig('hs.png')
        # plt.show(block = False)

        return weights, bins

yi = np.random.normal(loc=50, scale=10, size=1000)
bw = 10  # Bin width
sigma =3
hs = smoothing(bw,sigma)
hs.smoothingfunction()