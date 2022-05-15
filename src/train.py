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

torch.manual_seed(101)
model = ml.LSTM()
print(model)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
epochs = 5

start_time = time.time()

for epoch in range(epochs):
    for seq, y_train in sl.train_data:
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
preds = pn.train_norm[-window_size:].tolist()

# Set the model to evaluation mode
model.eval()

for i in range(future):
    seq = torch.FloatTensor(preds[-window_size:])
    print(seq)
    with torch.no_grad():
        model.hidden = (torch.zeros(2, 1, model.hidden_size),
                        torch.zeros(2, 1, model.hidden_size))
        preds.append(model(seq).item())
