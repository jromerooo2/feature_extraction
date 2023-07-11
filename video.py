from feat import Detector
from feat.utils.io import get_test_data_path
from feat.plotting import imshow
import os

detector = Detector(
    face_model="retinaface",
    landmark_model="mobilefacenet",
    au_model='xgb',
    emotion_model="resmasknet",
    facepose_model="img2pose",
)


audio_folder = 'samples'
all_audio = [f for f in os.listdir(audio_folder) if f.endswith('.mp4')]
all_audio.sort()

for file in all_audio:
    face = os.path.join('samples',file)
    pred = detector.detect_video(face)
    print(pred)