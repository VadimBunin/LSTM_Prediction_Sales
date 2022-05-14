import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd


df = pd.read_csv('data/Advance Retail Sales Clothing and Clothing Accessory Stores.csv',
                 index_col=0, parse_dates=True)

# Divide the data into train and test sets
y = df.values.astype(float)
y = np.squeeze(y)

test_size = 24

train_data = y[:-test_size]
test_data = y[-test_size:]

print(f'train_set {len(train_data)}')
print(f'test_set {len(test_data)}')

# Normalize the data

scaler = MinMaxScaler(feature_range=(-1, 1))
train_norm = scaler.fit_transform(train_data.reshape(-1, 1))
