import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "0";

from tensorflow.keras.datasets import mnist
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

import deepwrap
from deepwrap import vision as vis

data_aug = vis.get_data_aug(rotation_range=15,
                            zoom_range=0.1,
                            width_shift_range=0.1,
                            height_shift_range=0.1)
classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

(trn, val, preproc) = vis.images_from_array(x_train, y_train,
                                            validation_data=None,
                                            val_pct=0.1,
                                            random_state=42,
                                            data_aug=data_aug,
                                            class_names=classes)

# Using a LeNet-style classifier
model = vis.image_classifier('default_cnn', trn, val)

learner = deepwrap.get_learner(model, train_data=trn, val_data=val, batch_size=128)

learner.lr_find(show_plot=True, max_epochs=3)

learner.fit_onecycle(1e-3, 3)
learner.validate(class_names=preproc.get_classes())

learner.view_top_losses(n=1)

predictor = deepwrap.get_predictor(learner.model, preproc)

predictor.predict(x_test[0:1])[0]

np.argmax(predictor.predict(x_test[0:1], return_proba=True)[0])

predictor.save('/my_mnist')

p = deepwrap.load_predictor('/my_mnist')

p.predict(x_test[0:1])[0]

predictions = p.predict(x_test)

import pandas as pd

df = pd.DataFrame(zip(predictions, y_test), columns=['Predicted', 'Actual'])
print(df.head())
