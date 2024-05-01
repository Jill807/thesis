"""
@author : Jill Gulbis
@date : 2024-06-30
@github : 
"""

import torch

DEVICE = torch.device('cuda:0')
CHECKPOINT_DIR = "./checkpoint"

N_EPOCH = 1000

BATCH_SIZE = 2048
NUM_WORKERS = 8

LEARNING_RATE = 1e-5


IMAGE_PATH = " "
SENSOR_PATH = " "