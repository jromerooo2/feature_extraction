import numpy as np
import cv2
import os, os.path

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

DIR = './GENKI4K/files'
OUTPUT_DIR = './cropped'
numPics = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])

for pic in sorted(os.listdir(DIR)):

    image_file = os.path.join(DIR, pic)
    try:
      img = cv2.imread(image_file);
      

      height = img.shape[0]
      width = img.shape[1]
      size = height * width

      if size > (500^2):
          r = 500.0 / img.shape[1]
          dim = (500, int(img.shape[0] * r))
          img2 = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
          img = img2

      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      faces = face_cascade.detectMultiScale(gray, 1.1, 5)
      # print faces
      eyesn = 0

      for (x,y,w,h) in faces:
          imgCrop = img[y:y+h,x:x+w]

          roi_gray = gray[y:y+h, x:x+w]
          roi_color = img[y:y+h, x:x+w]

          eyes = eye_cascade.detectMultiScale(roi_gray)
          for (ex,ey,ew,eh) in eyes:
              eyesn = eyesn +1
          if eyesn >= 2:
              cv2.imwrite(os.path.join(OUTPUT_DIR, pic), imgCrop)

      imgCrop = cv2.resize(imgCrop, (64,64))
      cv2.imwrite(os.path.join(OUTPUT_DIR, pic), imgCrop)
      
      print("Image"+image_file+" has been processed and cropped")
      k = cv2.waitKey(30) & 0xff
      if k == 27:
          break
    except IOError as e:
      print('Could not read: ', image_file, ' : ', e)

#cap.release()
print("All images have been processed!!!")
cv2.destroyAllWindows()
cv2.destroyAllWindows()