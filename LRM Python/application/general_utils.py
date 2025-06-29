import tensorflow as tf
from typing import List
import cv2
import os

vocab = [char for char in "abcdefghijklmnopqrstuvwxyz'?!0123456789 "] # create array of every possible character expected to be encountered within annotations

# create a bidirectional mapping between characters and integers to enable easy conversion between character sequences and their corresponding numerical representations
char_to_num = tf.keras.layers.StringLookup(vocabulary = vocab, oov_token = "") # convert characters to numbers
num_to_char = tf.keras.layers.StringLookup(vocabulary = char_to_num.get_vocabulary(), oov_token = "", invert = True) # convert numbers to characters

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

    video_path = os.path.join('..', 'data', 's1', f'{file_name}.mpg')
    alignment_path = os.path.join('..', 'data', 'alignments', 's1', f'{file_name}.align')

    frames = load_video(video_path)
    alignments = load_alignments(alignment_path)

    return frames, alignments # return frames (75 frames x 46px height x 140px width x 1 color channel) and alignments (21 tokens)