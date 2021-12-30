#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import streamlit as st
import tensorflow_hub as hub
import os
import numpy as np
import pandas as pd
from PIL import Image
import wget



breeds_name = np.unique(pd.read_csv('./labels.csv').breed.to_numpy())

model = tf.keras.models.load_model('dog_breeds_mobilenetv2_Adam_v1_.h5', custom_objects={'KerasLayer':hub.KerasLayer})

def image_prediction(image_path):
    """
    Takes an image file path and
    1. Truns into a Tensor
    2. predict the breed
    3. show the accuracy
    """
    img = Image.open(image_path)
    img.save('tmp/tmp.jpg')

    image = tf.io.read_file('tmp/tmp.jpg')
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[224, 224])
    
    prediction = model.predict(np.expand_dims(image, axis=0))
    the_breed = breeds_name[np.argmax(prediction)].upper()
    accuracy = str(round(np.max(prediction)*100, 2))+'%'

    return img, the_breed, accuracy

st.write('''
        # Know my dog bread
        ''')

image = st.file_uploader('Please upload your dog image.', type=['jpg', 'png'])
url = st.text_input('Or, please give a valid link.')

if image is not None:
    for file in os.listdir('tmp'):
        os.remove(os.path.join('tmp', file))
    
    image, breed, acc = image_prediction(image)

    bread_c, acc_c = st.columns(2)
    bread_c.metric('The Breed is:', breed)
    acc_c.metric('Confidance:',acc)

    st.image(image)
elif url is not None:
    for file in os.listdir('tmp'):
        os.remove(os.path.join('tmp', file))
    try: 
        image = wget.download(url,out='tmp')
        image, breed, acc = image_prediction(image)

        bread_c, acc_c = st.columns(2)
        bread_c.metric('The Breed is:', breed)
        acc_c.metric('Confidance:',acc)

        st.image(image)
    except ValueError:
        st.text('Please paste a valid link.')

else:
    st.text('Please upload your dog image or paste a link.')
