import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
from collections import defaultdict

from src.eeg_encoder import EEGEncoderBIOT
from src.fmri_encoder import FMRIEncoder1D
from src.utils.dataset import SimultEEG_fMRI, ContrastiveBatchSampler, collate_fn
from src.utils.utils import multi_positive_clip_loss, count_params
from config import TrainConfig

import lightning as L

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    L.seed_everything(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    #code for main process

