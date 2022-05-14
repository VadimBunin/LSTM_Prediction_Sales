import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('data/Advance Retail Sales Clothing and Clothing Accessory Stores.csv',
                 index_col=0, parse_dates=True)
print(df.head())
df.dropna(inplace=True)
df = df[:'2019']
plt.figure(figsize=(8, 3), dpi=150)
plt.title('Advance Retail Sales: Retail Trade')
plt.ylabel('Millions of Dollars')
plt.grid(True)
#plt.autoscale(axis='x', tight=True)
plt.plot(df['RSCCASN'])
plt.show()
