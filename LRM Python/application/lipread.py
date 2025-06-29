# Create virtual environment (Mac):     python3 -m venv [name]
# Activate virtual environment (Mac):   source [name]/bin/activate
# Streamlit run command:                python -m streamlit run lipread.py

# import relevant dependencies
import streamlit as st
import numpy as np
import os
import imageio

import tensorflow as tf
from general_utils import load_data, num_to_char
from model_utils import load_model

# set application page layout
st.set_page_config(layout = 'wide')

# sidebar set up
with st.sidebar:
    st.image('https://thenounproject.com/api/private/icons/2318993/edit/?backgroundShape=SQUARE&backgroundShapeColor=%23000000&backgroundShapeOpacity=0&exportSize=752&flipX=false&flipY=false&foregroundColor=%23000000&foregroundOpacity=1&imageFormat=png&rotation=0')
    st.title('LipReader')
    st.info('This application is originally developed from the LipNet deep learning model.')

st.title('LipNet Full Stack Model')

# set up a drop-down list of video options that can be selected
video_options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Choose video', video_options)

# set up two columns
col1, col2 = st.columns(2)

if video_options:

    with col1:
        st.info('The video below is in mp4 format')
        file_path = os.path.join('..', 'data', 's1', selected_video) # obtain selected video file path
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y') # convert video from mpg to mp4 file

        # render video inside application page
        video = open('test_video.mp4', 'rb')
        video_bytes = video.read()
        st.video(video_bytes)

    with col2:
        st.info('This is all the machine learning model sees when making a prediction')
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        imageio.mimsave('animation.gif', video, format = 'GIF', fps = 10)
        st.image('animation.gif', width = 460)

        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        y_hat = model.predict(tf.expand_dims(video, axis = 0))
        decoded = tf.keras.backend.ctc_decode(y_hat, [75], greedy = True)[0][0].numpy()
        st.text(decoded[0])

        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join([num_to_char(word) for word in decoded])
        st.text(converted_prediction.numpy().decode('utf-8').split('9'))

        print('PREDICTED TEXT:', ' '.join(converted_prediction.numpy().decode('utf-8').split('9')))