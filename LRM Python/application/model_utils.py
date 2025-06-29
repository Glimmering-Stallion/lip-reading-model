import os
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten

# neural network loader function
def load_model() -> Sequential:
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

    model.add(TimeDistributed(Reshape((-1,)))) # independently apply flatten operation on every temporal slice of input

    # first bidirectional LSTM (128 LSTM hidden units, initialize weights with orthogonal matrices, return a sequence of outputs instead of a single final output)
    model.add(Bidirectional(LSTM(128, kernel_initializer = 'Orthogonal', return_sequences = True)))
    model.add(Dropout(0.5)) # drop out 50% of the units

    # second bidirectional LSTM
    model.add(Bidirectional(LSTM(128, kernel_initializer = 'Orthogonal', return_sequences = True)))
    model.add(Dropout(0.5))

    # fully connected dense layer (number of neurons in output layer equal to vocab size, weights of neurons initialized with he_normal initialization, activation function for output layer as softmax)
    model.add(Dense(41, kernel_initializer = 'he_normal', activation = 'softmax'))

    model.load_weights(os.path.join('..', 'models', 'checkpoint')) # load checkpoints from specified directory
    # model.load_weights(os.path.join(os.path.dirname(__file__), '..', 'models', 'model.weights.h5')) # load checkpoints from specified directory

    return model