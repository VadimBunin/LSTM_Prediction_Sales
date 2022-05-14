import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd
import preprocessing_normalization as pn

train_norm = torch.FloatTensor(pn.train_norm).view(-1)
print(train_norm)
