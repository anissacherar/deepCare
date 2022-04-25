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
import os
from skimage import exposure
from pathlib import Path
from argparse import Namespace

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



#load the model 
local="./model/model_5_ResNet.h5"
#reference=np.asarray(Image.open("2846_IMG_9391_431.jpg"))
dim = (50, 50)

"""def app_object_detection():

    MODEL_LOCAL_PATH = "./model/best_BCCM.pt"


    CLASSES = [
        "background",
        "RBC",
        "WBC",
    ]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


    DEFAULT_CONFIDENCE_THRESHOLD = 0.5

    class Detection(NamedTuple):
        name: str
        prob: float

    class MobileNetSSDVideoProcessor(VideoProcessorBase):
        confidence_threshold: float
        result_queue: "queue.Queue[List[Detection]]"

        def __init__(self) -> None:
            self._net = cv2.dnn( str(MODEL_LOCAL_PATH))
            self.confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
            self.result_queue = queue.Queue()

        def _annotate_image(self, image, detections):
            # loop over the detections
            (h, w) = image.shape[:2]
            result: List[Detection] = []
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > self.confidence_threshold:
                    # extract the index of the class label from the `detections`,
                    # then compute the (x, y)-coordinates of the bounding box for
                    # the object
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    name = CLASSES[idx]
                    result.append(Detection(name=name, prob=float(confidence)))

                    # display the prediction
                    label = f"{name}: {round(confidence * 100, 2)}%"
                    cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(
                        image,
                        label,
                        (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        COLORS[idx],
                        2,
                    )
            return image, result

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="bgr24")
            blob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
            )
            self._net.setInput(blob)
            detections = self._net.forward()
            annotated_image, result = self._annotate_image(image, detections)

            # NOTE: This `recv` method is called in another thread,
            # so it must be thread-safe.
            self.result_queue.put(result)

            return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=MobileNetSSDVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    confidence_threshold = st.slider(
        "Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05
    )
    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.confidence_threshold = confidence_threshold

    if st.checkbox("Show the detected labels", value=True):
        if webrtc_ctx.state.playing:
            labels_placeholder = st.empty()
            # NOTE: The video transformation with object detection and
            # this loop displaying the result labels are running
            # in different threads asynchronously.
            # Then the rendered video frames and the labels displayed here
            # are not strictly synchronized.
            while True:
                if webrtc_ctx.video_processor:
                    try:
                        result = webrtc_ctx.video_processor.result_queue.get(
                            timeout=1.0
                        )
                    except queue.Empty:
                        result = None
                    labels_placeholder.table(result)
                else:
                    break

class VideoTransformer(VideoTransformerBase):
            webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=MobileNetSSDVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )"""
classes={0:'RBC', 1 :'WBC', 2: 'platelet'}
def define_options(split_path) :
    return Namespace(agnostic_nms=False, \
                augment=False, \
                classes=classes, 
                conf_thres=0.4, 
                device='', 
                img_size=640, 
                iou_thres=0.5, 
                #output='inference/output', 
                save_txt=False, 
                source=split_path, 
                update=False, 
                view_img=True, 
                weights=['best_BCCM.pt'])
class VideoTransformer(VideoTransformerBase):
    """def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

        return img"""
    def transform(self, frame):
    
        img = frame.to_ndarray(format="bgr24")

        #image gray
        #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        options = define_options(img)
        boxes = detect.detect(options, print_msg = True)
        for (x, y, w, h) in boxes:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(255, 0, 0), thickness=2)
            #cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return img
#old 1awdgaKTdrhk3U5cUbWr5ilaS21XVHNQ7
#old 2 https://drive.google.com/file/d/1m90YsqJYROdASP2utrNeqfegwbw3c7NP/view?usp=sharing
GOOGLE_DRIVE_FILE_ID="14ZBhwCAZGNCf1ZfAILGn-IFjv9SGJd1Y"
 
@st.cache(allow_output_mutation=True)
def load_model():
    
	# path to file
	filepath = "model/"
	# folder exists?
	if not os.path.exists('model'):
		# create folder
		os.mkdir('model')
	
	# file exists?
	if not os.path.exists(filepath):
		# download file
		from GD_download import download_file_from_google_drive
		download_file_from_google_drive(id=GOOGLE_DRIVE_FILE_ID, destination=filepath)
	
	# load model
	model = tf.keras.models.load_model(local)
	return model

class MyImage:
    def __init__(self, img_name):
        self.img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)
        self.__name = img_name

    def __str__(self):
        return self.__name

#fonction de suppression du fond image -thresh= fond
def background_subtract_area(image, area_threshold=7000, light_bg=True):
    #  default area_threshold is similar to the area of a disk with radius 50, which is ImageJ's default
    if light_bg:
        return area_closing(image, area_threshold) - image
    else:
        return image - area_opening(image, area_threshold)

def posouneg(img,model):
    img_r = cv2.resize(img, dim)
    img_r = imp.img_to_array(img_r)
    img_r = np.expand_dims(img_r, axis=0)
    img_r /= 255.

    res=np.argmax(model.predict(img_r),axis=1)
    if res==0:
        prediction = 'GRN'
    elif res==1:
        prediction='GRP'

    return prediction

def plot_image_grid(images, ncols=None):
    '''Plot a grid of images'''
    if not ncols:
        factors = [i for i in range(1, len(images)+1) if len(images) % i == 0]
        ncols = factors[len(factors) // 2] if len(factors) else len(images) // 4 + 1
    nrows = int(len(images) / ncols) + int(len(images) % ncols)
    imgs = [images[i] if len(images) > i else None for i in range(nrows * ncols)]
    f, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 2*nrows))
    axes = axes.flatten()[:len(imgs)]
    for img, ax in zip(imgs, axes.flatten()):
        if np.any(img):
            if len(img.shape) > 2 and img.shape[2] == 1:
                img = img.squeeze()
            ax.imshow(img)


def cropChamp(image):
    e = 40  # envorion 1 cm 70 c'est beacoup ça dépend de l'oculaire du microscope
    plt.imshow(image)
    plt.show()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)) #à enlever ou laisser*************************************************************
    # closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    # centre de gravité du champ pour évaluer la segmentation
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    # print(cX,cY)
    rayon = cv2.pointPolygonTest(c, (cX, cY), True)
    ima = image.copy()
    # cv2.circle(ima, (cX, cY), 10, (255, 0, 0), -1)#centre de gravité
    cv2.circle(ima, (cX, cY), int(rayon - e), (255, 0, 0), 1)  # nouveau grand champ
    # ima=ima[cY-r:cY+r, cX-r:cX+r]
    # draw filled circle in white on black background as mask
    mask = np.zeros_like(image)
    mask = cv2.circle(mask, (cX, cY), int(rayon - e), (255, 255, 255), -1)

    # apply mask to image
    result = cv2.bitwise_and(ima, mask)
    # cv2.imwrite("resultats/evaluation_images/cercle_champ_"+(str(image).split('/')[-1]).split('.')[0]+".jpg",ima)
    # cv2.imwrite("resultats/evaluation_images/reducti_"+(str(image).split('/')[-1]).split('.')[0]+".jpg",result)

    ret, thresh = cv2.threshold(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY), 100, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    # cv2.rectangle(image.img, (x, y), (x + w, y + h), (0, 255, 0), 4)
    # cv2.imwrite("detected_forms.jpg",image.img)
    champ = result[y:y + h, x:x + w]

    if champ.shape[0] >= 4000 or champ.shape[1] >= 4000:
        #champ = cv2.resize(champ, (int(champ.shape[1] * 0.4), int(champ.shape[0] * 0.4)), interpolation=cv2.INTER_CUBIC)
        champ= champ.resize((int(champ.shape[1] * 0.4), int(champ.shape[0] * 0.4)), Image.ANTIALIAS)
        print("nouvelle dim champ trop grand :", champ.shape)
        # plt.imshow(champ)
        # plt.show()
    else:
        print("Champ correctement détecté !")
        print("W,H :", champ.shape)

    # cv2.imwrite("resultats/evaluation_images/odm/" + (str(image).split('/')[-1]).split('.')[0] + ".jpg", champ)
    return champ

#@st.cache(allow_output_mutation=True, max_entries=10, ttl=3600) #online uniquement
def exam(champ,model):
    # champ2 = cv2.fastNlMeansDenoisingColored(champ, None, 10, 10, 21, 7)
    plt.figure(figsize=(10, 10))
    plt.imshow(champ)
    plt.show()
    champ_seg = champ.copy()  # nouveau champ pour dessiner les bbox et le périmêtre d'analyse
    # champ=cv2.medianBlur(champ, 3)
    # cv2.imwrite("thresh2.jpg",thresh2)
    ####################################################kmeans##########################################################
    # result_image = kmeans_color_quantization(champ, clusters=4)#combien de couleurs ?
    start_time = time.time()
    #####################################################binarize#########################################################
    gray = cv2.cvtColor(champ, cv2.COLOR_BGR2GRAY)
    gray = background_subtract_area(gray)
    # gray = cv2.filter2D(gray, -1, kernel_sharpening)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # cv2.imwrite("resultats/gray_equa.jpg", gray)
    # gray[gray != 0] = 255
    # cv2.imwrite("resultats/gray_whiteall.jpg", gray)
    # gray=cv2.medianBlur(gray, 5)
    # cv2.imwrite("resultats/gray_whiteall_sans_bruit.jpg",gray)
    # print("#######################thresh 1#####################")
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(thresh, kernel, iterations=3)  # 3 ou 2 max ********************************
    thresh = cv2.dilate(erosion, kernel, iterations=2)
    # on cherche le contour des hématies blanches
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    ####################on ferme les trous dans les GR pour que watershed marche bien###################################
    for cnt in contours:
        # cv2.drawContours(opened, [cnt], 0,255,-1, cv2.LINE_8)
        cv2.drawContours(thresh, [cnt], 0, 255, -1)

    # plt.imshow(thresh, cmap='gray')
    # plt.show()
    # cv2.imwrite("resultats/thresh.jpg", thresh)

    #################################watershed########################################################################
    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=10, labels=thresh)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)
    # print("[INFO] {} segments retrouvés".format(len(np.unique(labels)) - 1))

    vraiesCellules = []  # une liste ou on va stocker les vrais GR
    grp = []
    idx = 1
    #shutil.rmtree("resultats/pos/", ignore_errors=True)
    #os.mkdir("resultats/pos/")
    #shutil.rmtree("resultats/neg/", ignore_errors=True)
    #os.mkdir("resultats/neg/")

    for label in np.unique(labels):
        # si label ==0 on regarde le fond à ignorer
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255

        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
        c = max(cnts, key=cv2.contourArea)

        ##############################################################################################################
        x, y, w, h = cv2.boundingRect(c)
        # if w<=1.5*h and h<=1.5*w and 20 <= w <= 100 and 20 <= h <= 100 :#and dist>=30 and area >=200 :
        if 30 <= w <= 100 and 30 <= h <= 100:  # 20 si c'est du x100 ou oculaire x 12 30 si c'est du X50 champ large
            gr = champ[y-5:y + h+5, x-5:x + w+5]
            #gr = cv2.cvtColor(gr, cv2.COLOR_BGR2RGB)
            #gr= match_histograms(gr, reference)
            vraiesCellules.append(gr)
            res = posouneg(gr,model)
            if res == 'GRP':
                # cv2.circle(champ_seg, (cX, cY), 3, (0,255 , 0), -1)
                cv2.rectangle(champ_seg, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # cv2.putText(champ_seg, 'GRP',  (x + w + 10, y + h), 0, 0.3, (255, 0, 0))
                #cv2.imwrite("resultats/pos/" + str(idx) + ".jpg", gr)
                grp.append(gr)

            elif res == 'GRN':
                # cv2.circle(champ_seg, (cX, cY), 3, (0,255 , 0), -1)
                cv2.rectangle(champ_seg, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # cv2.putText(champ_seg, 'GRN',  (x + w + 10, y + h), 0, 0.3, (0, 255, 0))
                #cv2.imwrite("resultats/neg/" + str(idx) + ".jpg", gr)

        idx += 1
    grn=len(vraiesCellules)-len(grp)
    p = '%03.03f%%' % ((len(grp) / len(vraiesCellules) * 100))
    print("Nombre de GRP : ", len(grp))
    print("Nombre de GRN : ", grn)
    print("charge parasitaire du champ :")
    print(p)
    print("###########################################################################")
    print("Analyse de " + str(len(vraiesCellules)) + " hématies en  --- %s secondes ---" % ((time.time() - start_time)))
    print("###########################################################################")

    return champ_seg, grp, p

def exam_x1000(champ,model):
    # champ2 = cv2.fastNlMeansDenoisingColored(champ, None, 10, 10, 21, 7)
    champ_seg = champ.copy()  # nouveau champ pour dessiner les bbox et le périmêtre d'analyse
    # champ=cv2.medianBlur(champ, 3)
    # cv2.imwrite("thresh2.jpg",thresh2)
    ####################################################kmeans##########################################################
    # result_image = kmeans_color_quantization(champ, clusters=4)#combien de couleurs ?
    start_time = time.time()
    #####################################################binarize#########################################################
    gray = cv2.cvtColor(champ, cv2.COLOR_BGR2GRAY)
    gray = background_subtract_area(gray)
    # gray = cv2.filter2D(gray, -1, kernel_sharpening)
    gray = cv2.equalizeHist(gray)
    #gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # cv2.imwrite("resultats/gray_equa.jpg", gray)
    # gray[gray != 0] = 255
    # cv2.imwrite("resultats/gray_whiteall.jpg", gray)
    # gray=cv2.medianBlur(gray, 5)
    # cv2.imwrite("resultats/gray_whiteall_sans_bruit.jpg",gray)
    # print("#######################thresh 1#####################")
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(thresh, kernel, iterations=2)  # 3 ou 2 max ********************************
    thresh = cv2.dilate(erosion, kernel, iterations=2)
    # on cherche le contour des hématies blanches
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    ####################on ferme les trous dans les GR pour que watershed marche bien###################################
    for cnt in contours:
        # cv2.drawContours(opened, [cnt], 0,255,-1, cv2.LINE_8)
        cv2.drawContours(thresh, [cnt], 0, 255, -1)

    # plt.imshow(thresh, cmap='gray')
    # plt.show()
    # cv2.imwrite("resultats/thresh.jpg", thresh)

    #################################watershed########################################################################
    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=10, labels=thresh)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)
    # print("[INFO] {} segments retrouvés".format(len(np.unique(labels)) - 1))

    vraiesCellules = []  # une liste ou on va stocker les vrais GR
    grp = []
    idx = 1
    #shutil.rmtree("resultats/pos/", ignore_errors=True)
    #os.mkdir("resultats/pos/")
    #shutil.rmtree("resultats/neg/", ignore_errors=True)
    #os.mkdir("resultats/neg/")

    for label in np.unique(labels):
        # si label ==0 on regarde le fond à ignorer
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255

        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
        c = max(cnts, key=cv2.contourArea)

        ##############################################################################################################
        x, y, w, h = cv2.boundingRect(c)
        # if w<=1.5*h and h<=1.5*w and 20 <= w <= 100 and 20 <= h <= 100 :#and dist>=30 and area >=200 :
        if 45 <= w <= 200 and 45 <= h <= 200:  # >40 min si c'est du x100 ou oculaire x 12 30 si c'est du X50 champ large
            gr = champ[y-5:y + h+5, x-5:x + w+5]
            #print(gr.shape)
            #gr = cv2.cvtColor(gr, cv2.COLOR_BGR2RGB)
            #gr= match_histograms(gr, reference)
            vraiesCellules.append(gr)
            res = posouneg(gr,model)
            if res == 'GRP':
                # cv2.circle(champ_seg, (cX, cY), 3, (0,255 , 0), -1)
                cv2.rectangle(champ_seg, (x-5, y-5), (x + w+5, y + h+5), (255, 0, 0), 2)
                # cv2.putText(champ_seg, 'GRP',  (x + w + 10, y + h), 0, 0.3, (255, 0, 0))
                #cv2.imwrite("resultats/pos/" + str(idx) + ".jpg", gr)
                grp.append(gr)

            elif res == 'GRN':
                # cv2.circle(champ_seg, (cX, cY), 3, (0,255 , 0), -1)
                cv2.rectangle(champ_seg, (x-5, y-5), (x + w+5, y + h+5), (0, 255, 0), 2)
                # cv2.putText(champ_seg, 'GRN',  (x + w + 10, y + h), 0, 0.3, (0, 255, 0))
                #cv2.imwrite("resultats/neg/" + str(idx) + ".jpg", gr)

        idx += 1
    grn=len(vraiesCellules)-len(grp)
    p = '%03.03f%%' % ((len(grp) / len(vraiesCellules) * 100))
    print("Nombre de GRP : ", len(grp))
    print("Nombre de GRN : ", grn)
    print("charge parasitaire du champ :")
    print(p)
    print("###########################################################################")
    print("Analyse de " + str(len(vraiesCellules)) + " hématies en  --- %s secondes ---" % ((time.time() - start_time)))
    print("###########################################################################")

    return thresh,champ_seg, grp, p


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
def app():
    col1, col2 = st.columns(2)
    choices = col1.selectbox("Select the output results ", options=('Parasite density (%)', 'Species identification', 'Multi-stage identification'))

    if (choices == "Parasite density (%)"):
        magni = col2.radio("Select the magnification of the blood smear images.", 
                            options=["x500", "x1000", "Live Detection"], 
                            help="Microscopic magnification used to capture the images")
        if magni == 'x500':
            display = Image.open('graphical abstract.jpg')
            display = np.array(display)
            col2.image(display, width = 400)        
            model = load_model()
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
                        champ_f, grp, p = exam(champ, model=model)
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(label="PARs (%)", value=p)
                            st.image(champ_f,use_column_width=True)
                        with col2:
                            st.image(grp, width=100, channels='RGB')
        elif magni=='x1000':    
            model = load_model()
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
            video_transformer_factory=VideoTransformer,
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

if __name__ == "__app__":
    app()