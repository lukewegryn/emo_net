#!/usr/bin/env python3
 
"""Simple HTTP Server With Upload.

This module builds on BaseHTTPServer by implementing the standard GET
and HEAD requests in a fairly straightforward manner.

see: https://gist.github.com/UniIsland/3346170
"""
 
 
__version__ = "0.1"
__all__ = ["SimpleHTTPRequestHandler"]
__author__ = "bones7456"
__home_page__ = "http://li2z.cn/"
 
import tflearn
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

import os
import cv2
import posixpath
import http.server
import urllib.request, urllib.parse, urllib.error
import cgi
import shutil
import mimetypes
import re
from io import BytesIO
import tflearn
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

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

CASC_PATH = './haarcascade_files/haarcascade_frontalface_default.xml'
cascade_classifier = cv2.CascadeClassifier(CASC_PATH)


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
model = tflearn.DNN(net)
model.load('./practice_model/checkpoint_emo-4041')
emotion_dict = {0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Sad", 5:"Surprise", 6:"Neutral"}

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def onehot_to_dense(array):
    index = np.argmax(array)
    return index

def cropFace(image):
	imagePath = image
	faceCascade = cv2.CascadeClassifier('./haarcascade_files/haarcascade_frontalface_default.xml')
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(
	    gray,
	    scaleFactor=1.1,
	    minNeighbors=5
	)
	if not len(faces) > 0:
		return None
	max_area_face = faces[0]
	for face in faces:
	    if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
	        max_area_face = face
	# Chop image to face
	face = max_area_face
	image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]
	return image

def analyze_file(fn, model):
	image = cropFace(fn)
	image = imresize(rgb2gray(image), (48, 48))
    
	if image == None:
		return "ERROR!"
	else:
		Xs = np.array(image, dtype='uint8')
		print(Xs.shape)
		Xs = np.reshape(Xs,(1, 48, 48, 1))
		print(Xs.shape)
		return model.predict(Xs)

pred = analyze_file('S026_001_00000002.png',model)[0]
prediction = emotion_dict[np.argmax(pred)]

print(prediction)