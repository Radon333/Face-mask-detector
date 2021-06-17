import cv2
import numpy as np
import math
from tensorflow.keras.models import load_model
model=load_model("face_mask_detection.h5")
labels_dict={0:'without_mask',1:'with_mask'}
color_dict={0:(0,0,255),1:(0,255,0)}

size = 4


# We load the xml file
classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
class VideoCamera(object):
    def __init__(self):
        # capturing video
        self.webcam = cv2.VideoCapture(0)

    def __del__(self):
        # releasing camera
        self.webcam.release()

    def get_frame(self):
        (rval, im) = self.webcam.read()
        im = cv2.flip(im, 1, 1)  # Flip to act as a mirror

        # Resize the image to speed up detection
        mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))

        # detect MultiScale / faces
        faces = classifier.detectMultiScale(mini)

        # Draw rectangles around each face
        for f in faces:
            (x, y, w, h) = [v * size for v in f]  # Scale the shapesize backup
            # Save just the rectangle faces in SubRecFaces
            face_img = im[y:y + h, x:x + w]
            resized = cv2.resize(face_img, (180, 180))
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1, 180, 180, 3))
            reshaped = np.vstack([reshaped])
            result = model.predict(reshaped)
            result = result + 0.3
            print(result)
            label = math.floor(result[0][0])
            # print(label)
            cv2.rectangle(im, (x, y), (x + w, y + h), color_dict[label], 2)
            cv2.rectangle(im, (x, y - 40), (x + w, y), color_dict[label], -1)
            cv2.putText(im, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            break
        ret, png = cv2.imencode('.png', im)
        return png.tobytes()

