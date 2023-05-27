import cv2
import mediapipe as mp
mp_face_detection=mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.7)
mp_drawing=mp.solutions.drawing_utils

webcam=cv2.VideoCapture(0)

while webcam.isOpened():
    success,img=webcam.read()

    # face detection using MediaPipe
    img=cv2.cvtColor(cv2.flip(img, 1),cv2.COLOR_BGR2RGB)
    results=mp_face_detection.process(img)

    # draw the face detection annotations on the image
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(img,detection)
        print("Face find")
    else:
        print("Not founded")

    cv2.imshow("Face recognitier",img)
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

webcam.release()
cv2.destroyAllWindows()
