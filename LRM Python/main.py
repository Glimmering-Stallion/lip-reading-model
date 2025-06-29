# This is an implementation of the model found in this paper:
# https://arxiv.org/pdf/1611.01599.pdf



# 1: Initial Setup

# install dependencies
# pip install command: !pip install opencv-python matplotlib imageio gdown tensorflow
# pip see installed packages command: !pip list

# import dependencies
import os
import cv2
import tensorflow as tf
import numpy as np
from typing import List
from matplotlib import pyplot as plt
import imageio

# prevent out of memory (OOM) errors by setting GPU memory consumption growth
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.xperimental.set_memory_growth(physical_devices[0], True)
except:
    pass



# 2: Build Data Loading Functions

import gdown

# url = 'https://drive.google.com/uc?id=1YlvpDLix3S-U8fd-gqRwPcWXAXm8JwjL'
# output = 'data.zip'
# gdown.download(url, output, quiet = False)
# gdown.extractall('data.zip')

# data loading function (takes in a data path and outputs a list of floats)
def load_video(path: str) -> List[float]:

    cap = cv2.VideoCapture(path) # create a video capture instance

    # loop through each frame and store in an array
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read() # read in video frame data
        frame = tf.image.rgb_to_grayscale(frame) # convert from RGB to grayscale to reduce amount of data to be processed
        frames.append(frame[190:236, 80:220, :]) # static slicing for trimming down frame to isolate to mouth region
    cap.release()

    # standardization of frame data through Z-score normalization
    mean = tf.math.reduce_mean(frames) # calculate mean
    std = tf.math.reduce_std(tf.cast(frames, tf.float32)) # calculate standard deviation
    return tf.cast((frames - mean), tf.float32) / std # return standardized frame data

vocab = [char for char in "abcdefghijklmnopqrstuvwxyz'?!0123456789 "] # create array of every possible character expected to be encountered within annotations

# create a bidirectional mapping between characters and integers to enable easy conversion between character sequences and their corresponding numerical representations
char_to_num = tf.keras.layers.StringLookup(vocabulary = vocab, oov_token = "") # convert characters to numbers
num_to_char = tf.keras.layers.StringLookup(vocabulary = char_to_num.get_vocabulary(), oov_token = "", invert = True) # convert numbers to characters

# debugging
# print(
#     f"The vocabulary is: {char_to_num.get_vocabulary()} "
#     f"(size = {char_to_num.vocabulary_size()})"
# )

# function to load alignments (takes in a data path and outputs a list of strings)
def load_alignments(path: str) -> List[str]:

    # read in lines
    with open(path, 'r') as f:
        lines = f.readlines()

    # loop through each line
    tokens = []
    for line in lines:
        line = line.split() # split line into a list of elements by each space
        # append third element of line to an array (ignore any lines that contain 'sil')
        if line[2] != 'sil':
            tokens = [*tokens, ' ', line[2]]
            
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding = 'UTF-8'), (-1)))[1:] # return the numerical values of characters found in the line elements

# function to load data (takes in a data path and outputs frames and alignments)
def load_data(path: str):
    path = bytes.decode(path.numpy())

    file_name = path.split('/')[-1].split('.')[0] # file name splitting for Mac
    # file_name = path.split('\\')[-1].split('.')[0] # file name splitting for Windows

    video_path = os.path.join('data', 's1', f'{file_name}.mpg')
    alignment_path = os.path.join('data', 'alignments', 's1', f'{file_name}.align')

    frames = load_video(video_path)
    alignments = load_alignments(alignment_path)

    return frames, alignments # return frames (75 frames x 46px height x 140px width x 1 color channel) and alignments (21 tokens)

# debugging
# test_path = '/data/s1/bbal6n.mpg'
# test_frames, test_alignments = load_data(tf.convert_to_tensor(test_path))
# [plt.imshow(test_frames[f_index]) and plt.show() for f_index in range(len(test_frames))] # display frames as plots
# print([bytes.decode(x) for x in num_to_char(test_alignments.numpy()).numpy()]) # display corresponding alignment as characters in an array

# mappable processing function (takes in a data path and outputs frames and alignments as a tuple of tensors)
def mappable_function(path: str) -> List[str]:
    result = tf.py_function(load_data, [path], (tf.float32, tf.int64)) # wrap load_data function as a TensorFlow operation
    return result



# 3: Create Data Pipeline

from matplotlib import pyplot as plt

data = tf.data.Dataset.list_files('./data/s1/*.mpg') # obtain all video files in given directory
data = data.shuffle(500, reshuffle_each_iteration = False) # shuffle an amount of data specified by the cache size
data = data.map(mappable_function) # transforms the dataset of file paths into a new dataset
data = data.padded_batch(2, padded_shapes = ([75, None, None, None], [40])) # create new TensorFlow dataset containing batches of 2 elements (the frames tensor within each batch is padded to have a fixed size of 75 frames)
data = data.prefetch(tf.data.AUTOTUNE) # fetch data from next iteration while current iteration is still being processed which mitigates bottlenecking (the argument 'tf.data.AUTOTUNE' is a special value that allows TensorFlow to automatically tune the prefetching buffer size based on available system resources)

# train/test data partitioning
train = data.take(450)
test = data.skip(450)

# debugging
sample_data = data.as_numpy_iterator().next(); sample_data[0] # obtain frames from data
# imageio.mimsave('./animation.gif', test_data[0][1], duration = 100) # create an animated GIF file from the sequence of frames
tf.strings.reduce_join([num_to_char(word) for word in sample_data[1][0]]) # display second alignment in batch as characters



# 4: Design Deep Neural Network

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

# neural network
model = Sequential() # instantiation of model

# first convolution (128 convolution kernels, 3x3x3 kernel size, frame data shape, padding is same to preserve input shape)
model.add(Conv3D(128, 3, input_shape = (75, 46, 140, 1), padding = 'same'))
model.add(Activation('relu')) # provide non-linearity outputs using ReLU function
model.add(MaxPool3D((1, 2, 2))) # condense output down through max pooling

# second convolution (256 convolution kernels, 3x3x3 kernel size, frame data shape, padding is same to preserve input shape)
model.add(Conv3D(256, 3, padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1, 2, 2)))

# third convolution (75 convolution kernels, 3x3x3 kernel size, frame data shape, padding is same to preserve input shape)
model.add(Conv3D(75, 3, padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1, 2, 2)))

model.add(TimeDistributed(Flatten())) # independently apply flatten operation on every temporal slice of input

# first bidirectional LSTM (128 LSTM hidden units, initialize weights with orthogonal matrices, return a sequence of outputs instead of a single final output)
model.add(Bidirectional(LSTM(128, kernel_initializer = 'Orthogonal', return_sequences = True)))
model.add(Dropout(0.5)) # drop out 50% of the units

# second bidirectional LSTM
model.add(Bidirectional(LSTM(128, kernel_initializer = 'Orthogonal', return_sequences = True)))
model.add(Dropout(0.5))

# fully connected dense layer (number of neurons in output layer equal to vocab size, weights of neurons initialized with he_normal initialization, activation function for output layer as softmax)
model.add(Dense(char_to_num.vocabulary_size(), kernel_initializer = 'he_normal', activation = 'softmax'))

model.summary() # debugging

# debugging
# y_hat = model.predict(sample_data[0])



# 5: Training Process

# function for learning rate scheduling
def scheduler(epoch, lr):
    if epoch < 30:
        return lr # output learning rate if epochs is below certain threshold
    else:
        return lr * tf.math.exp(-0.1) # otherwise reduce the learning rate using exponential function
    
# function for calculating loss between a continuous (unsegmented) time series and a target sequence
def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype = 'int64') # get the number of examples in the batch
    input_len = tf.cast(tf.shape(y_pred)[1], dtype = 'int64') # get the length of the predicted sequence
    label_len = tf.cast(tf.shape(y_true)[1], dtype = 'int64') # get the length of the true sequence

    # expand input_len and label_len to match the batch size
    input_len = input_len * tf.ones(shape = (batch_len, 1), dtype = 'int64')
    label_len = label_len * tf.ones(shape = (batch_len, 1), dtype = 'int64')

    # compute and return the CTC loss using tf.keras.backend.ctc_batch_cost
    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_len, label_len)
    return loss

# custom callback class that generates and prints examples after each epoch during training
class ProduceExample(tf.keras.callbacks.Callback):

    # constructor that initializes callback with a dataset iterator
    def __init__(self, dataset) -> None:
        self.dataset = dataset.as_numpy_iterator()

    # function called at the end of each training epoch to generate and print examples
    def on_epoch_end(self, epoch, logs = None) -> None:
        data = self.dataset.next() # get the next batch of data from dataset
        y_hat = self.model.predict(data[0]) # predict sequences using model
        decoded = tf.keras.backend.ctc_decode(y_hat, [75, 75], greedy = False)[0][0].numpy() # decodes predicted sequences using CTC decoding

        # loop through each example in the batch and print original ground truth sequences along with predicted sequences
        for x in range(len(y_hat)):
            print('Original: ', tf.string.reduce_join([vocab[word] + ' ' for word in data[1][x]])).numpy().decode('utf-8')
            print('Prediction: ', tf.string.reduce_join([vocab[word] + ' ' for word in decoded[x]])).numpy().decode('utf-8')
            print('~' * 100) # print separator line after each example

model.compile(optimizer = Adam(learning_rate = 0.0001), loss = CTCLoss) # compile model

# define callbacks
# checkpoint_callback = ModelCheckpoint(os.path.join('models', 'checkpoint'), monitor = 'loss', save_weights_only = True) # responsible for saving the model's weights during training
checkpoint_callback = ModelCheckpoint('models/model.weights.h5', monitor = 'loss', save_weights_only = True) # responsible for saving the model's weights during training
schedule_callback = LearningRateScheduler(scheduler) # allows for scheduling changes to the learning rate during training
example_callback = ProduceExample(data) # responsible for generating and printing examples at the end of each epoch during training

model.fit(train, validation_data = test, epochs = 100, callbacks = [checkpoint_callback, schedule_callback, example_callback]) # train model on given dataset for specified number of epochs



# 5: Make Predictions

# extract model checkpoints externally
# url = 'https://drive.google.com/uc?id=1vWscXs4Vt0a_1IH1-ct2TCgXAZT-N3_Y'
# output = 'checkpoints.zip'
# gdown.download(url, output, quiet = False)
# gdown.extractall('checkpoints.zip', 'models')

model.load_weights('models/model.weights.h5') # load checkpoints from specified directory

# # debugging
# test_data = test.as_numpy_iterator().next()
# y_hat = model.predict(test_data[0])
# decoded = tf.keras.backend.ctc_decode(y_hat, input_length = [75, 75], greedy = True)[0][0].numpy()

# # actual text
# print('REAL TEXT', '~' * 100)
# [tf.strings.reduce_join([num_to_char(word) for word in sentence]) for sentence in test_data[1]]

# # predicted text
# print('PREDICTIONS', '~' * 100)
# [tf.strings.reduce_join([num_to_char(word) for word in sentence]) for sentence in decoded]



# 6: Testing

sample = load_data(tf.convert_to_tensor('./data/s1/swwi9s.mpg')) # load in new video data
y_hat = model.predict(tf.expand_dims(sample[0], axis = 0)) # perform prediction
decoded = tf.keras.backend.ctc_decode(y_hat, input_length = [75], greedy = True)[0][0].numpy() # decode prediction

# actual text
print('~' * 100)
actual_text = tf.strings.reduce_join([num_to_char(word) for word in sample[1]])
print('ACTUAL TEXT:', actual_text.numpy().decode('utf-8'))

# predicted text
print('~' * 100)
predicted_text = tf.strings.reduce_join([num_to_char(word) for word in decoded])
print('PREDICTED TEXT:', ' '.join(predicted_text.numpy().decode('utf-8').split('9')))