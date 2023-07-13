from feat import Detector
from feat.utils.io import get_test_data_path
from feat.plotting import imshow
import os
from smile import loop_video
import pandas as pd


#initialize the detector
detector = Detector(
    face_model="retinaface",
    landmark_model="mobilefacenet",
    au_model='xgb',
    emotion_model="resmasknet",
    facepose_model="img2pose",
)

img_folder = './samples/'
all_img = [f for f in os.listdir(img_folder) if f.endswith('.mp4')]
all_img.sort()

for file in all_img:
    face_video = os.path.join('./samples/' + file)
    #returns a dataframe that will be used to save later
    pred = detector.detect_video(face_video)

    #returns an array of arrays -> [[class_name (smile or no smile),confidence score ]]
    #idea is to save the new data as new columns in the original dataframe
    smile_frame = loop_video(face_video)
    column_smile = []
    column_landmarks = []
    for array in smile_frame:
        column_smile.append(array[0])
        column_landmarks.append(array[1])
    pred = pd.DataFrame(pred)
    pred["Class"] = column_smile
    pred["3D_landmark"] = column_landmarks
    pred.to_pickle('./output/video_{}.pkl'.format(file))
    pred.to_csv('./output/video_{}.csv'.format(file))

    # print(pred)
