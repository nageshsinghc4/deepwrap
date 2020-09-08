# predicting which Titanic passengers survived
# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import random
import tensorflow as tf
import pandas as pd

pd.set_option('display.max_columns', None)

seed_value = 0
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

import dltkdl
from dltkdl import tabledata

train_df = pd.read_csv('/Users/nageshsinghchauhan/Documents/Qubit/DLTK-DL/examples/titanic/train.csv', index_col=0)
print(train_df.head())

train_df = train_df.drop('Name', 1)
train_df = train_df.drop('Ticket', 1)
train_df = train_df.drop('Cabin', 1)

# creating test dataset
np.random.seed(42)
p = 0.1  # 10% for test set
prop = 1 - p
df = train_df.copy()
msk = np.random.rand(len(df)) < prop
train_df = df[msk]
test_df = df[~msk]

print(train_df.shape)

print(test_df.shape)

trn, val, preproc = tabledata.load_from_dataframe(train_df, label_columns=['Survived'], random_state=42)

# Print available models
tabledata.print_tabular_classifiers()

model = tabledata.tabular_classifier('mlp', trn)
learner = dltkdl.get_learner(model, train_data=trn, val_data=val, batch_size=32)

learner.lr_find(show_plot=True, max_epochs=5)

learner.fit_onecycle(5e-3, 10)

learner.evaluate(val, class_names=preproc.get_classes())

predictor = dltkdl.get_predictor(learner.model, preproc)

preds = predictor.predict(test_df, return_proba=True)

print('test accuracy:')
(np.argmax(preds, axis=1) == test_df['Survived'].values).sum() / test_df.shape[0]
"""
df = test_df.copy()[[c for c in test_df.columns.values if c != 'Survived']]
df['Survived'] = test_df['Survived']
df['predicted_Survived'] = np.argmax(preds, axis=1)
df.head()

predictor.explain(test_df, row_index=35, class_id=1)
"""