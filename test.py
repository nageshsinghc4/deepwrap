#Income prediction

#%reload_ext autoreload
#autoreload 2
#%matplotlib inline
#tabular ==> tabledata
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0"; 

import urllib.request
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)


import dltkdl
from dltkdl import tabledata

# training set
#download dataset
urllib.request.urlretrieve('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv', '/Users/nageshsinghchauhan/Documents/Qubit/DLTK-DL//examples/train.csv')


trn, val, preproc = tabledata.tabular_from_csv('/Users/nageshsinghchauhan/Documents/Qubit/DLTK-DL/examples/train.csv', label_columns='class', random_state=42)

#Invoking multilayer perceptron
model = tabledata.tabular_classifier('mlp', trn)
learner = dltkdl.get_learner(model, train_data=trn, val_data=val, batch_size=128)


learner.lr_find(show_plot=True)

learner.autofit(1e-3)
learner.validate(class_names=preproc.get_classes())

urllib.request.urlretrieve('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv', '/Users/nageshsinghchauhan/Documents/Qubit/DLTK-DL/examples/test.csv')


test_df = pd.read_csv('/Users/nageshsinghchauhan/Documents/Qubit/DLTK-DL/examples/test.csv')
print(test_df)

learner.evaluate(preproc.preprocess_test(test_df), class_names=preproc.get_classes())

preproc.get_classes()

predictor = dltkdl.get_predictor(learner.model, preproc)
preds = predictor.predict(test_df)


[12:49] Shakeel Dhada
    c = dltk_ai.DltkAiClient('57bd609e-5d14-4da1-8fab-a1092587f3e4')
response = c.sentiment_analysis('I am feeling good.')


df = test_df.copy()
df['predicted_class'] = preds


print(df)



import ktrain
from ktrain import tabular
import pandas as pd
train_df = pd.read_csv('train.csv', index_col=0)
train_df = train_df.drop(['Name', 'Ticket', 'Cabin'], 1)
trn, val, preproc = tabular.tabular_from_df(train_df, label_columns=['Survived'], random_state=42)
learner = ktrain.get_learner(tabular.tabular_classifier('mlp', trn), train_data=trn, val_data=val)
learner.lr_find(show_plot=True, max_epochs=5) # estimate learning rate
learner.fit_onecycle(5e-3, 10)

# evaluate held-out labeled test set
tst = preproc.preprocess_test(pd.read_csv('heldout.csv', index_col=0))
learner.evaluate(tst, class_names=preproc.get_classes())


