# %reload_ext autoreload
# autoreload 2
# %matplotlib inline
# tabular ==> tabledata
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "0";

import urllib.request
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)

import deepwrap
from deepwrap import tabledata

train_df = pd.read_csv('/deepwrap/examples/house-prices/train.csv', index_col=0)
# print(train_df)

train_df.drop(['Alley', 'PoolQC', 'MiscFeature', 'Fence', 'FireplaceQu', 'Utilities'], 1, inplace=True)

trn, val, preproc = tabledata.load_from_dataframe(train_df, is_regression=True, label_columns='SalePrice',
                                                  random_state=42)

# Invoking multilayer perceptron NN
model = tabledata.tabular_regression_model('mlp', trn)
learner = deepwrap.get_learner(model, train_data=trn, val_data=val, batch_size=128)

learner.lr_find(show_plot=True, max_epochs=16)
learner.autofit(1e-1)

print(learner.evaluate(test_data=val))
