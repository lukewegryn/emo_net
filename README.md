# Emotion Detection

Author: Luke Wegryn
Date: 6/10/2017

## Scope

This project was created as an education assignment and final project for the Kadenze class [Creative Applications of Deep Learning with Tensorflow](https://www.kadenze.com/courses/creative-applications-of-deep-learning-with-tensorflow-iv)

## Overview

The goal of this project is to create a deep net that can succefully detect the emotion of a user based on a photo taken of the user's face. 

This is acheived using a deep neural network similar to the [Alex-net](https://github.com/tflearn/tflearn/blob/master/examples/images/alexnet.py). To simplify the creation of the network, a high level tensorflow API called [TFLearn](https://github.com/tflearn/tflearn) was used.

## Dataset

The dataset is from [Kraggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/leaderboard). 

Permission to use the dataset must be approved by Kraggle before use, so it is not provided in this repository.

## Models

Final checkpoints of the model are located in `tensorflow/emo_net` and have an input of [None, 48, 48, 1] and an output of [7]. The 7 output emotions include (by this association) `{0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Sad", 5:"Surprise", 6:"Neutral"}`.

## Results

99% accuracy on the data set
50% accuracy on the validation set

While the results don't seem staggeringly positive, the model works relatively well in practice. It does confuse certain emotions such as disgust/anger, anger/sadness, suprise/fear, but in general works well enough to convince the user that it isn't simply guessing! If you smile, it will almost always return "Happy"!

## Usage

There are two ways this repository can be used:

1. To train a emotion recognition model
2. To play with a working model using a webcam (local client/server model)

You must first train the model, and then copy the checkpoint file over to the `server/models/` directory. Then in `server/server.py` you must change the line where the model is loaded to point to your model instead of the model currently in there.

### Training

To train this model:

1. Navigate to the `tensorflow/` directory.
2. Run `python emo_tflearn.py`
3. The resulting model checkpoints will occur every 200 steps, and appear in the `emo_net` folder.

### Playing with the model

**Note: This will ask your browser for permission to use the webcam, but the images you take are sent only to localhost, and are taken only when you click the button**

To play with this model using your webcam:

1. Navigate to the `server/` directory
2. Run `python server.py`
3. Wait for the server to start up (this can take a while)
4. If you run into errors, odds are you are missing one of the million dependencies this relies on. This uses Python 3 and TensorFlow among many other things (numpy, cv2, etc.). Please follow the errors and `pip install` the missing components and use google to track down the missing components.
4. In a web browser, navigate to `http://127.0.0.1:8000` (*Note: http://0.0.0.0:8000 will not work correctly even though the console says that*)
4. Proceed to take images of yourself or others making emotional faces, and see what you get!
5. This has only been tested in Chrome on Mac OSX.
