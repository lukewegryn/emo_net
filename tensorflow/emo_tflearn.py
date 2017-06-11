# First check the Python version
import sys
if sys.version_info < (3,4):
    print('You are running an older version of Python!\n\n' \
          'You should consider updating to Python 3.4.0 or ' \
          'higher as the libraries built for this course ' \
          'have only been tested in Python 3.4 and higher.\n')
    print('Try installing the Python 3.5 version of anaconda '
          'and then restart `jupyter notebook`:\n' \
          'https://www.continuum.io/downloads\n\n')

# Now get necessary libraries
try:
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.transform import resize
    from skimage import data
    from scipy.misc import imresize
    import IPython.display as ipyd
    import csv
    import shlex
except ImportError:
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.transform import resize
    from skimage import data
    from scipy.misc import imresize
    import IPython.display as ipyd
    print('Done!')

# Import Tensorflow
try:
    import tensorflow as tf
except ImportError:
    print("You do not have tensorflow installed!")
    print("Follow the instructions on the following link")
    print("to install tensorflow before continuing:")
    print("")
    print("https://github.com/pkmital/CADL#installation-preliminaries")

try:
    from libs import utils, gif, datasets, dataset_utils, vae, dft
except ImportError:
    print("Make sure you have started notebook in the same directory" +
          " as the provided zip file which includes the 'libs' folder" +
          " and the file 'utils.py' inside of it.  You will NOT be able"
          " to complete this assignment unless you restart jupyter"
          " notebook inside the directory created by extracting"
          " the zip file or cloning the github repo.")
# We'll tell matplotlib to inline any drawn figures like so:
plt.style.use('ggplot')

def import_csv(filename):
  labels = []
  images = []
  with open(filename,'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
      if row[2] == "Training":
        labels.append(row[0])
        images.append(row[1])
  return labels, images

######## Start actual code ##########

data_file = "/Users/luke/ownCloud/deep_learning/course/final_project/fer2013.csv"
labels,images = import_csv(data_file)
assert(len(labels) == len(images))

#read in the images
imgs = []
for image in images:
    imgs.append(np.fromstring(str(image), dtype=np.uint8,sep=' '))
Xs = imgs
ys = labels
Xs = np.array(imgs).astype(np.uint8)
ys = np.array(ys).astype(np.uint8)
#print(ys)
assert(len(Xs) == len(ys))

ds = datasets.Dataset(Xs,ys,one_hot=True,split=[0.8, 0.1, 0.1])

for i in range(0, 10):
    ds.X[i].shape

from tensorflow.python.framework.ops import reset_default_graph
reset_default_graph()

# We'll have placeholders just like before which we'll fill in later.
n_input = 48*48
n_output = 7
ds_X_reshape = np.reshape(ds.X,(28709, 48, 48, 1))
ds_valid_images_reshape = np.reshape(ds.valid.images,(ds.valid.images.shape[0],48,48,1))

#https://github.com/tflearn/tflearn/blob/master/examples/images/alexnet.py
#pip install tflearn
import tflearn
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

"""
net = tflearn.input_data(shape=[None, 48, 48,1])
net = tflearn.conv_2d(net, 64, 5, activation = 'relu')
net = tflearn.max_pool_2d(net, 3, strides = 2)
net = tflearn.conv_2d(net, 64, 5, activation = 'relu')
net = tflearn.max_pool_2d(net, 3, strides = 2)
net = tflearn.conv_2d(net, 128, 4, activation = 'relu')
net = tflearn.dropout(net, 0.3)
net = tflearn.fully_connected(net, 3072, activation = 'tanh')
net = tflearn.fully_connected(net, 7, activation='softmax')
net = tflearn.regression(net, optimizer='momentum', loss='categorical_crossentropy')
"""

network = tflearn.input_data(shape=[None, 48, 48,1])
network = tflearn.conv_2d(network, 96, 11, strides=4, activation='relu')
network = tflearn.max_pool_2d(network, 3, strides=2)
network = tflearn.local_response_normalization(network)
network = tflearn.conv_2d(network, 256, 5, activation='relu')
network = tflearn.max_pool_2d(network, 3, strides=2)
network = tflearn.local_response_normalization(network)
network = tflearn.conv_2d(network, 384, 3, activation='relu')
network = tflearn.conv_2d(network, 384, 3, activation='relu')
network = tflearn.conv_2d(network, 256, 3, activation='relu')
network = tflearn.max_pool_2d(network, 3, strides=2)
network = tflearn.local_response_normalization(network)
network = tflearn.fully_connected(network, 4096, activation='tanh')
network = tflearn.dropout(network, 0.5)
network = tflearn.fully_connected(network, 4096, activation='tanh')
network = tflearn.dropout(network, 0.5)
network = tflearn.fully_connected(network, 7, activation='softmax')
network = tflearn.regression(network, optimizer='momentum',
                     loss='categorical_crossentropy')

model = tflearn.DNN(network,checkpoint_path='./emo_net/checkpoint_emo_net',max_checkpoints=3)
model.fit(ds_X_reshape, ds.Y, n_epoch=1000, show_metric=True, shuffle=True, validation_set=0.01, batch_size=64, snapshot_step=200, snapshot_epoch=False, run_id='emo_net')
model.save('./emo_net/emotion_recog.tflearn')

pred = model.predict(ds_X_reshape)

def onehot_to_dense(array):
    index = np.argmax(array)
    return index

distribution = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
for i in range(0,len(pred)):
    distribution[onehot_to_dense(pred[i])] += 1
print(distribution)