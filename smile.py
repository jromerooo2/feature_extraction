from keras.models import load_model
from PIL import Image, ImageOps 
import numpy as np
import os, os.path
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import time

def loop_video(vid_path):
    v= cv2.VideoCapture(vid_path)
    i=1
    while True:
        ret,frame = v.read()
        # print(frame)
        detect_smile(frame,i)
        i+=1
        if ret == False:
            break

def detect_smile(frame=None,i=0):

    np.set_printoptions(suppress=True)
    model = load_model("keras_Model.h5", compile=False)
    class_names = open("labels.txt", "r").readlines()
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    detected_face = DeepFace.detectFace(frame, detector_backend = 'opencv')
    if i >= 34:
        plt.imshow(detected_face)
        plt.title("figure " + str(i))
        plt.show()
    image = Image.fromarray(np.uint8(detected_face*255)).convert("RGB")
    image = image.resize((64,64))
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)

    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    print("Class:", class_name[2:], end="")
    print("Frame number" + str(i))
    print("Confidence Score:", confidence_score)

# detect_smile()
loop_video('./testing/diego.mp4')