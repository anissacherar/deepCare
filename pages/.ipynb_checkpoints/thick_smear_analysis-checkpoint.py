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


def app():
    choose_model = st.sidebar.selectbox("IN PROGRESS", ["MALARIS model", "Upload your model"])