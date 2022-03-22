import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.morphology import  binary_closing, disk, area_opening, area_closing
from keras.preprocessing import image as imp
import matplotlib.pyplot as plt
import time
import csv
import shutil
import os
from skimage import exposure
from skimage.exposure import match_histograms
from pathlib import Path

#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
st.set_page_config(
     page_title="MALARIS",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded",
)

#multipages
from multipage import MultiPage
from pages import thin_smear_analysis, thick_smear_analysis

# Create an instance of the app 
app = MultiPage()
col1, col2 = st.columns(2)

# Title of the main page
display = Image.open('logo.JPG')
display = np.array(display)
st.image(display, width = 120)
st.title("MALARIS : The web app for malaria diagnosis")

#col1.image(display, width = 120)

#col2.title("")

# Add all your application here
app.add_page("THIN SMEAR", thin_smear_analysis.app)
app.add_page("THICK SMEAR", thick_smear_analysis.app)

# The main app
app.run()


