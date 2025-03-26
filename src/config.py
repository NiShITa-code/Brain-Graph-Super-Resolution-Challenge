# config.py
import torch
import os
import random
import numpy as np

# Set random seeds for reproducibility
random_seed = 42
os.environ['PYTHONHASHSEED'] = str(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data paths
LR_TRAIN_PATH = "../lr_train.csv"
HR_TRAIN_PATH = "../hr_train.csv"
LR_TEST_PATH = "../lr_test.csv"

# Model hyperparameters
IN_CHANNELS = 161
HIDDEN_CHANNELS = 128
EMBEDDING_DIM = 64
OUTPUT_DIM = 35778
DROPOUT = 0.5
NUM_HEADS = 4

# Training hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 80
LEARNING_RATE = 0.001
LAMBDA_TOPO = 0.1  # Weight for topological loss

# Evaluation settings
NUM_FOLDS = 3