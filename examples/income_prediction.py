# Income prediction

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

import dltkdl
from dltkdl import tabledata

# training set
# download dataset
urllib.request.urlretrieve('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv',
                           '/Users/nageshsinghchauhan/Documents/Qubit/DLTK-DL/examples/train.csv')

trn, val, preproc = tabledata.load_from_csv('/Users/nageshsinghchauhan/Documents/Qubit/DLTK-DL/examples/train.csv',
                                            label_columns='class', random_state=42)

# Invoking multilayer perceptron
model = tabledata.tabular_classifier('mlp', trn)
learner = dltkdl.get_learner(model, train_data=trn, val_data=val, batch_size=128)

learner.lr_find(show_plot=True)

learner.autofit(1e-3)
learner.validate(class_names=preproc.get_classes())

urllib.request.urlretrieve('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv',
                           '/Users/nageshsinghchauhan/Documents/Qubit/DLTK-DL/examples/test.csv')

test_df = pd.read_csv('/Users/nageshsinghchauhan/Documents/Qubit/DLTK-DL/examples/test.csv')
print(test_df)

learner.evaluate(preproc.preprocess_test(test_df), class_names=preproc.get_classes())

preproc.get_classes()

predictor = dltkdl.get_predictor(learner.model, preproc)
preds = predictor.predict(test_df)

df = test_df.copy()
df['predicted_class'] = preds

print(df)
