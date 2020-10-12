# DeepWrap

**DEEPWRAP** is a lightweight wrapper for the deep learning library TensorFlow and Keras (and other libraries) to help build, train, and deploy neural networks and other machine learning models. Inspired by ML framework extensions like fastai and ludwig, it is designed to make deep learning and AI more accessible and easier to apply for both newcomers and experienced practitioners. With only a few lines of code, deepwrap allows you to easily and quickly.

**Installation Instructions**

Make sure pip is up-to-date with:

Install TensorFlow 2 if it is not already installed (e.g., pip install tensorflow==2.3.0)

Install deepwrap: 
```
git clone https://github.com/nageshsinghc4/deepwrap

cd deepwrap

python setup.py install or pip install .
```

The above should be all you need on Linux systems and cloud computing environments like Google Colab and AWS EC2.

Some important things to note about installation:

If using deepwrap on a local machine with a GPU (versus Google Colab, for example), you'll need to install GPU support for TensorFlow 2.

On Google Colab, TensorFlow 2 should be already installed. You should be able to use deepwrap with any version of TensorFlow 2. However, we currently recommend using TensorFlow 2.3.0 (if possible) due to a TensorFlow bug that will not be fixed until TensorFlow 2.4 that affects the Learning-Rate-Finder.

Since some deepwrap dependencies have not yet been migrated to tf.keras in TensorFlow 2 (or may have other issues), deepwrap is temporarily using forked versions of some libraries. If not installed, deepwrap will complain when a method or function needing either of these libraries is invoked. 

### Available algorithms under deepwrap:

#### Artificial Neural Network : For tabular dataset. 
1. **mlp**: a configurable multilayer perceptron with categorical variable embeddings.

#### Convolutional Neural Network : For image/video dataset
1. **pretrained_resnet50**: 50-layer Residual Network (pretrained on ImageNet)
2. **resnet50**: 50-layer Residual Network (randomly initialized)
3. **pretrained_mobilenet**: MobileNet Neural Network (pretrained on ImageNet)
4. **mobilenet**: MobileNet Neural Network (randomly initialized)
5. **pretrained_inception**: Inception Version 3  (pretrained on ImageNet)
6. **inception**: Inception Version 3 (randomly initialized)
7. **wrn22**: 22-layer Wide Residual Network (randomly initialized)
8. **default_cnn**: a default Convolutional Neural Network

#### Recurrent neural Network : For sequenced datasets like text.
1. **fasttext**: a fastText-like model [http://arxiv.org/pdf/1607.01759.pdf]
2. **logreg**: logistic regression using a trainable Embedding layer
3. **nbsvm**: NBSVM model [http://www.aclweb.org/anthology/P12-2018]
4. **bigru**: Bidirectional GRU with pretrained word vectors [https://arxiv.org/abs/1712.09405]
5. **standard_gru**: simple 2-layer GRU with randomly initialized embeddings
6. **bert**: Bidirectional Encoder Representations from Transformers (BERT) [https://arxiv.org/abs/1810.04805]
7. **distilbert**: distilled, smaller, and faster BERT from Hugging Face [https://arxiv.org/abs/1910.01108]

### Examples: 

You can also chckout step-by step implementation by following python notebooks under tutorial folder. 

Example **Hourse price prediction using mlp**
```
import urllib.request
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
import deepwrap
from deepwrap import tabledata

train_df = pd.read_csv('train.csv', index_col=0)
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
```
Example: **Classifying Images of Dogs and Cats Using a Pretrained ResNet50 model**

```
import deepwrap
from deepwrap import vision as vis

# load data
(train_data, val_data, preproc) = vis.images_from_folder(
                                              datadir='data/dogscats',
                                              data_aug = vis.get_data_aug(horizontal_flip=True),
                                              train_test_names=['train', 'valid'], 
                                              target_size=(224,224), color_mode='rgb')

# load model
model = vis.image_classifier('pretrained_resnet50', train_data, val_data, freeze_layers=80)

# wrap model and data in deepwrap.Learner object
learner = deepwrap.get_learner(model=model, train_data=train_data, val_data=val_data, 
                             workers=8, use_multiprocessing=False, batch_size=64)

# find good learning rate
learner.lr_find()             # briefly simulate training to find good learning rate
learner.lr_plot()             # visually identify best learning rate

# train using triangular policy with ModelCheckpoint and implicit ReduceLROnPlateau and EarlyStopping
learner.autofit(1e-4, checkpoint_folder='/tmp/saved_weights')
```
Example: **Language translation(English to Dutch)**
```
from deepwrap import text 
translator = text.Translator(model_name='Helsinki-NLP/opus-mt-en-nl')
src_text = '''My name is Sarah and I live in London.'''
print(translator.translate(src_text))

Output: Mijn naam is Sarah en ik woon in Londen.
```
Example: **Text classification**
```
# load text data
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
from sklearn.datasets import fetch_20newsgroups

train_b = fetch_20newsgroups(subset='train', categories=categories, shuffle=True)
test_b = fetch_20newsgroups(subset='test', categories=categories, shuffle=True)
(x_train, y_train) = (train_b.data, train_b.target)
(x_test, y_test) = (test_b.data, test_b.target)

# build, train, and validate model (Transformer is wrapper around transformers library)
import deepwrap
from deepwrap import text

MODEL_NAME = 'distilbert-base-uncased'
t = text.Transformer(MODEL_NAME, maxlen=500, class_names=train_b.target_names)
trn = t.preprocess_train(x_train, y_train)
val = t.preprocess_test(x_test, y_test)
model = t.get_classifier()
learner = deepwrap.get_learner(model, train_data=trn, val_data=val, batch_size=6)
learner.fit_onecycle(5e-5, 4)
learner.validate(class_names=t.get_classes())  # class_names must be string values

Output: 
                        precision    recall  f1-score   support

           alt.atheism       0.91      0.93      0.92       319
         comp.graphics       0.98      0.96      0.97       389
               sci.med       0.97      0.96      0.96       396
soc.religion.christian       0.95      0.97      0.96       398

              accuracy                           0.96      1502
             macro avg       0.95      0.95      0.95      1502
          weighted avg       0.96      0.96      0.96      1502

array([[296,   1,   6,  16],
       [ 12, 372,   5,   0],
       [  8,   6, 379,   3],
       [  8,   2,   0, 388]])
```
