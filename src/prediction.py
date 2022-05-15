import time
from turtle import forward
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd
import preprocessing_normalization as pn
import seq_label as sl
import model as ml
import train as tr
future = 12
window_size = 12

# Add the last window of training values to the list of predictions
preds = pn.train_norm[-window_size:].tolist()

# Set the model to evaluation mode
tr.model.eval()

for i in range(future):
    seq = torch.FloatTensor(preds[--window_size:])
    with torch.no_grad():
        tr.model.hidden = (torch.zeros(2, 1, tr.model.hidden_size),
                           torch.zeros(2, 1, tr.model.hidden_size))
        preds.append(tr.model(seq).item())
