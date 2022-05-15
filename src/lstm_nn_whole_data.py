import time
from turtle import forward
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import preprocessing_normalization as pn
import seq_label as sl
#import model as ml

df = pd.read_csv('data/Advance Retail Sales Clothing and Clothing Accessory Stores.csv',
                 index_col=0, parse_dates=True)

df.dropna(inplace=True)
df = df[:'2019']
y = df['RSCCASN'].values.astype(float)
window_size = 12

scaler = MinMaxScaler(feature_range=(-1, 1))
train_norm = scaler.fit_transform(y.reshape(-1, 1))
train_norm = torch.FloatTensor(train_norm).view(-1)


def input_data(seq, ws):
    out = []
    L = len(seq)
    for i in range(L-ws):
        window = seq[i:i+ws]
        label = seq[i+ws:i+ws+1]
        out.append((window, label))
    return out


train_data = input_data(train_norm, window_size)
input_dim = 1
hidden_dim = 128
num_layers = 2
output_dim = 1


class LSTMnetwork(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)

        self.linear = nn.Linear(hidden_size, output_size)

        self.hidden = (torch.zeros(2, 1, self.hidden_size),
                       torch.zeros(2, 1, self.hidden_size))

    def forward(self, seq):
        lstm_out, self.hidden = self.lstm(
            seq.view(len(seq), 1, -1), self.hidden)
        pred = self.linear(lstm_out.view(len(seq), -1))
        return pred[-1]


torch.manual_seed(101)
model = LSTMnetwork()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
epochs = 70

start_time = time.time()

for epoch in range(epochs):

    # extract the sequence & label from the training data
    for seq, y in train_data:

        # reset the parameters and hidden states
        optimizer.zero_grad()
        model.hidden = (torch.zeros(2, 1, model.hidden_size),
                        torch.zeros(2, 1, model.hidden_size))

        y_pred = model(seq)

        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

    # print training result
    print(f'Epoch: {epoch+1:2} Loss: {loss.item():10.8f}')

print(f'\nDuration: {time.time() - start_time:.0f} seconds')

future = 12

# Add the last window of training values to the list of predictions
preds = train_norm[-window_size:].tolist()

# Set the model to evaluation mode
model.eval()

for i in range(future):
    seq = torch.FloatTensor(preds[-window_size:])

    with torch.no_grad():
        model.hidden = (torch.zeros(2, 1, model.hidden_size),
                        torch.zeros(2, 1, model.hidden_size))
        preds.append(model(seq).item())

true_predictions = scaler.inverse_transform(
    np.array(preds[window_size:]).reshape(-1, 1))

true_predictions = pd.DataFrame(true_predictions)
true_predictions.index = df['RSCCASN'][-12:].index
true_predictions.columns = ['Forecast']
ax = df['RSCCASN'][-12:].plot(color='r', label='Data', figsize=(12, 8))
true_predictions.plot(ax=ax, label='Forecast')
plt.title('Historical Data and the Forecast')
plt.ylabel('Million $')
plt.legend()
plt.show()
