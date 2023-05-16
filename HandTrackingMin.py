import cv2
import mediapipe as mp
import time

import os

cap = cv2.VideoCapture(1)

while True:
    sucess, img = cap.read()

    cv2.imshow("Image", img)
    cv2.waitKey(1)