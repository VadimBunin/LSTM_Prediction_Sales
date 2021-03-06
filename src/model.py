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


torch.manual_seed(101)
model = LSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
epochs = 5

start_time = time.time()

for epoch in range(epochs):

    # extract the sequence & label from the training data
    for seq, y_train in train_data:

        # reset the parameters and hidden states
        optimizer.zero_grad()
        model.hidden = (torch.zeros(2, 1, model.hidden_size),
                        torch.zeros(2, 1, model.hidden_size))

        y_pred = model(seq)

        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

    # print training result
    print(f'Epoch: {epoch+1:2} Loss: {loss.item():10.8f}')

print(f'\nDuration: {time.time() - start_time:.0f} seconds')

future = 12
window_size = 12

# Add the last window of training values to the list of predictions
train_norm = pn.train_norm
preds = train_norm[-window_size:].tolist()

# Set the model to evaluation mode
model.eval()

for i in range(future):
    seq = torch.FloatTensor(preds[-window_size:])
    with torch.no_grad():
        model.hidden = (torch.zeros(2, 1, model.hidden_size),
                        torch.zeros(2, 1, model.hidden_size))
        preds.append(model(seq).item())
