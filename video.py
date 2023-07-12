from feat import Detector
from feat.utils.io import get_test_data_path
from feat.plotting import imshow
import os
from smile import detect_smile

#initialize the detector
detector = Detector(
    face_model="retinaface",
    landmark_model="mobilefacenet",
    au_model='xgb',
    emotion_model="resmasknet",
    facepose_model="img2pose",
)

img_folder = 'samples'
all_img = [f for f in os.listdir(img_folder) if f.endswith('.mp4')]
all_img.sort()

for file in all_img:
    face = os.path.join('samples',file)
    pred = detector.detect_video(face)
    print(pred)