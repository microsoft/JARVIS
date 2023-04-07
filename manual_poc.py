# Proof-of-concept
import cv2
import sys
import os
from constants import *
from emotion_recognition import EmotionRecognition
import numpy as np


def format_image(image):
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    faces = cv2.CascadeClassifier(CASC_PATH).detectMultiScale(
        image,
        scaleFactor=1.3,
        minNeighbors=5
    )
    # None is we don't found an image
    if not len(faces) > 0:
        return None
    max_area_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
            max_area_face = face
    # Chop image to face
    face = max_area_face
    image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]

    # Resize image to network size
    try:
        image = cv2.resize(image, (SIZE_FACE, SIZE_FACE),
                           interpolation=cv2.INTER_CUBIC) / 255.
        while True:
            cv2.imshow("frame", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception:
        print("[+] Problem during resize")
        return None
    # cv2.imshow("Lol", image)
    # cv2.waitKey(0)
    return image


# Load Model
network = EmotionRecognition()
network.build_network()

files = []

for f in os.listdir("./"):
    ext = os.path.splitext(f)[1]
    if ext.lower() in [".jpg"]:
        files.append(f)

for f in files:
    frame = cv2.imread(f)
    # Predict result with network
    result = network.predict(format_image(frame))

    if result is not None:
        for index, emotion in enumerate(EMOTIONS):
            print(emotion, ': ', result[0][index])

    print("Emotion: of ", f, "-", EMOTIONS[np.argmax(result[0])])
