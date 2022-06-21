import streamlit as st
import streamlit.components.v1 as components


from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing import image as imp
import matplotlib.pyplot as plt


#Live
import detect
import queue
import time
import pydub
import threading
import asyncio
import logging
import av

from aiortc.contrib.media import MediaPlayer
from streamlit_webrtc import (
VideoTransformerBase,
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)
from typing import List, NamedTuple


from apps.code.thin import *

def main():
    col1, col2 = st.columns(2)
    choices = col1.selectbox("Select the output results ", options=('P.Falciparum detection and Parasite density (%)', 
    'Species identification', 
    'Multi-stage identification'))

    if (choices == "P.Falciparum detection and Parasite density (%)"):
        magni = col2.radio("Select the magnification of the blood smear images.", 
                            options=["x500", "x1000", "Live Detection"], 
                            help="Microscopic magnification used to capture the images")
        if magni == 'x500':
            #display = Image.open('../graphical abstract.jpg')
            #display = np.array(display)
            #col2.image(display, width = 400)        
            #model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=1e-4), loss='categorical_crossentropy',
                          #metrics=['accuracy'])
            file = st.file_uploader("Please upload images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
            imageCarouselComponent = components.declare_component("image-carousel-component", path="frontend/public")
            selectedImageUrl = imageCarouselComponent(imageUrls=file, height=200)
            col1, col2 = st.columns(2)
            if selectedImageUrl is not None:
                st.image(selectedImageUrl)
            if file is not None:
                if st.button('Run test'):
                    model = load_Model()
                    for f in file:
                        image = Image.open(f)
                        image = np.asarray(image)
                        champ = cropChamp(image)
                        col1.success("Image uploaded")
                        col1.image(image, use_column_width=True)
                        col2.success("Smear Detected")
                        col2.image(champ, use_column_width=True)
                        with st.spinner('Blood cells analysis...'):
                            champ_f, grp, p = exam(champ, model=model)
                            col3, col4 = st.columns(2)
                            with col3:
                                st.metric(label="PARs (%)", value=p)
                                st.image(champ_f,use_column_width=True)
                                plt.imshow(champ_f)
                                plt.show()
                            with col4:
                                st.image(grp, width=100, channels='RGB')                    
                    
        elif magni=='x1000':    
            model = load_Model()
            #model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=1e-4), loss='categorical_crossentropy',
                          #metrics=['accuracy'])
            file = st.file_uploader("Please upload an image file", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
            col1, col2 = st.columns(2)
            if file is not None:
                for f in file:
                    image = Image.open(f)
                    image = np.asarray(image)
                    champ = cropChamp(image)
                    col1.success("Image uploaded")
                    col1.image(image, use_column_width=True)
                    col2.success("Smear Detected")
                    col2.image(champ, use_column_width=True)
                    with st.spinner('Blood cells analysis...'):
                        thresh, champ_f, grp, p = exam_x1000(champ, model=model)
                        #st.image(thresh,width=500)
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(label="PARs (%)", value=p)
                            st.image(champ_f,use_column_width=True)
                        with col2:
                            st.image(grp, width=100, channels='RGB')
        elif magni=='Live Detection':
            webrtc_streamer(key="example", 
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=VideoTransformer,
            media_stream_constraints={
            "video": True,
            "audio": False
            }   )
            """ctx = webrtc_streamer(key="example",
            #video_processor_factory=VideoProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={
            "video": True,
            "audio": False
            }   
            )"""

                        
    elif (choices == "Species identification"):
        model = st.file_uploader("Please upload your model", type=["h5"])

        #model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=1e-4), loss='categorical_crossentropy',
                      #metrics=['accuracy'])
        if model is None:
            st.text("Please upload a model")
        else:
            model = tf.keras.models.load_model(model,compile=False)
            file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
            if file is None:
                st.text("Please upload an image")
            else:
                image = Image.open(file)
                image = np.asarray(image)
                champ = cropChamp(image)
                st.image(image, use_column_width=True)
                st.success("Smear Detected")
                st.image(champ, use_column_width=True)
                champ_f, grp, p = exam(champ,model=model)
                st.image(champ_f)
                st.text("Charge parasitaire :")
                st.write(p)
                st.image(grp, width=100)

if __name__ == "__main__":
    main()

def main():

    imageCarouselComponent = components.declare_component("image-carousel-component", path="frontend/public")
    imageUrls = [
        "https://images.unsplash.com/photo-1522093007474-d86e9bf7ba6f?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=764&q=80",
        "https://images.unsplash.com/photo-1610016302534-6f67f1c968d8?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=1075&q=80",
        "https://images.unsplash.com/photo-1516550893923-42d28e5677af?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=872&q=80",
        "https://images.unsplash.com/photo-1541343672885-9be56236302a?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=687&q=80",
        "https://images.unsplash.com/photo-1512470876302-972faa2aa9a4?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=870&q=80",
        "https://images.unsplash.com/photo-1528728329032-2972f65dfb3f?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=870&q=80",
        "https://images.unsplash.com/photo-1557744813-846c28d0d0db?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=1118&q=80",
        "https://images.unsplash.com/photo-1513635269975-59663e0ac1ad?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=870&q=80",
        "https://images.unsplash.com/photo-1595867818082-083862f3d630?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=870&q=80",
        "https://images.unsplash.com/photo-1622214366189-72b19cc61597?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=687&q=80",
        "https://images.unsplash.com/photo-1558180077-09f158c76707?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=764&q=80",
        "https://images.unsplash.com/photo-1520106212299-d99c443e4568?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=687&q=80",
        "https://images.unsplash.com/photo-1534430480872-3498386e7856?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=870&q=80",
        "https://images.unsplash.com/photo-1571317084911-8899d61cc464?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=870&q=80",
        "https://images.unsplash.com/photo-1624704765325-fd4868c9702e?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=764&q=80",
    ]
    selectedImageUrl = imageCarouselComponent(imageUrls=imageUrls, height=200)

    if selectedImageUrl is not None:
        st.image(selectedImageUrl)


main()