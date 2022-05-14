import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd
import preprocessing_normalization as pn

train_norm = torch.FloatTensor(pn.train_norm).view(-1)

window_size = 12


def input_data(seq, ws):
    output = []

    for i in range(len(seq) - ws):
        window = seq[i:i+ws]
        label = seq[i+ws:i+ws+1]
        output.append((window, label))
    return output


data = input_data(train_norm, window_size)
