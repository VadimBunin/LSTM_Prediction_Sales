import time
from turtle import forward
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd
import preprocessing_normalization as pn
import seq_label as sl

train_data = sl.train_data


input_size = 1
hidden_size = 128
num_layers = 2
output_size = 1
drop_prob = 0.0


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=1, drop_prob=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, dropout=drop_prob)
        self.dropout = nn.Dropout(drop_prob)
        self.linear = nn.Linear(hidden_size, output_size)
        self.hidden = (torch.zeros(num_layers, output_size),
                       torch.zeros(num_layers, output_size))

    def forward(self, seq):
        lstm_out, self.hidden = self.lstm(
            seq.view(len(seq), 1, -1), self.hidden)
        pred = self.linear(lstm_out.view(len(seq), -1))
        return pred[-1]
