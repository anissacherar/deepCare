import streamlit as st
import streamlit.components.v1 as components

#for anonyms id 
import uuid
id = uuid.uuid1()

from PIL import Image, ImageOps
import numpy as np
import pandas as pd
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
        st.image(file, width=150, channels='RGB') 
        #imageCarouselComponent = components.declare_component("image-carousel-component", path="frontend/public")
        #selectedImageUrl = imageCarouselComponent(imageUrls=file, height=200)
        col1, col2 = st.columns(2)            
        #st.image(selectedImageUrl)
        def handle_click():
            st.session_state.test = False
        if st.button('Run test', on_click=handle_click) and file is not None:
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
                    champ_f, grp, p, grn = exam(champ, model=model)
                    col3, col4 = st.columns(2)
                    with col3:
                        st.metric(label="PARs (%)", value=p)
                        st.image(champ_f, use_column_width=True)
                    with col4:
                        st.image(grp, width=100, channels='RGB')
                        if st.button('Deep Quality') and len(grp)>0:
                            st.image(grp, width=100, channels='RGB')
                report=pd.DataFrame({"Patient":id.hex, 
                                    "Data":f.name,
                                    "PARS (%)":p, 
                                    "PARs":len(grp), 
                                    "Uninfected":grn,
                                    "RBCs":grn+len(grp),
                                    }, index=[1])
            #st.dataframe(report["PARS (%)"].style.highlight_max(axis=0))
            #st.download_button(
            #label='Export report as CSV file', 
            #data = report, 
            #file_name ='report.csv')
                                    
    elif magni=='x1000':    
        model = load_Model()
        #model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=1e-4), loss='categorical_crossentropy',
                        #metrics=['accuracy'])
        file = st.file_uploader("Please upload images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
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
        #model = tf.keras.models.load_model(model,compile=False)
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


