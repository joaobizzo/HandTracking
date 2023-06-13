import cv2
import mediapipe as mp
import os
import webbrowser
import time

#  Setting detection parameters
num_hands = 2
min_detection_confidence = 0.8
min_tracking_confidence = 0.5

#  Drawing parameters

# --colors in BGR: // TODO: create a function to put rgb and convert to bgr
line_color = (224, 208, 64)
ball_color = (255, 255, 255)
keyborad_color = (194, 178, 2)
WHITE = (255, 255, 255)
BLACK = (15, 15, 15)


mp_hands = mp.solutions.hands
mp_draws = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)

#  By default, 0 or 1
#  Mac users can use their phone's camera by testing both options (0 and 1)
cam = cv2.VideoCapture(0)

#  Image resolution
resolution_x = 1280
resolution_y = 720
cam.set(cv2.CAP_PROP_FRAME_WIDTH, resolution_x)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution_y)

# special parameters
rock_and_roll = False

keys = [['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
        ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
        ['Z', 'X', 'C', 'V','B', 'N', 'M', ',', '.', '']]
offset = 80


#  Function to verify if the camera was successfully initialized
def verify_cam_success(camera_success):
    if camera_success == True:
        message = "Camera initialized"
        print(message)
    else:
        message = "Error initializing camera"
    print(message)

#  Function to inform whether the hand was correctly detected
def success_terminal(success):
    if success == True:
        message = "Hand detected"
    else:
        message = "Hand not detected"
    print(message)

#  Main function to find hand coordinates
def find_hand_coordinates(img, mirror=False):
    if mirror:
        #  Mirror the image
        img = cv2.flip(img, 1)
    #  Convert from RGB to BGR (OpenCV default)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    all_hands = []
    if result.multi_hand_landmarks:
        for hand_side, hand_landmarks in zip(result.multi_handedness, result.multi_hand_landmarks):
            hand_info = {}
            coordinates = []
            for landmark in hand_landmarks.landmark:
                coord_x, coord_y, coord_z = int(landmark.x * resolution_x), int(landmark.y * resolution_y), int(landmark.z * resolution_x)
                coordinates.append([coord_x, coord_y, coord_z])
                # print(coord_x, coord_y, coord_z)
            hand_info["coordinates"] = coordinates
            hand_info["side"] = hand_side.classification[0].label

            all_hands.append(hand_info)
            mp_draws.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                    mp_draws.DrawingSpec(color=ball_color, thickness=2, circle_radius=2),
                                    mp_draws.DrawingSpec(color=line_color, thickness=2, circle_radius=2))
        success = True
    else:
        success = False

    return img, all_hands, success

#  Function to determine if the hand is open or closed
def count_raised_fingers(hand):
    fingers = []
    for finger_tip in 8, 12, 16, 20:
        if hand['coordinates'][finger_tip][1] < hand['coordinates'][finger_tip - 2][1]:
          fingers.append(True)
        else:
            fingers.append(False)
    return fingers


def print_keyboard(img, position, letter, size=50, color=WHITE):
    cv2.rectangle(img, position, (position[0]+size, position[1]+size), keyborad_color, cv2.FILLED)
    cv2.rectangle(img, position, (position[0]+size, position[1]+size), BLACK, 1)
    cv2.putText(img, letter, (position[0]+15, position[1]+30), cv2.FONT_HERSHEY_PLAIN, 2, WHITE, 3)
    return img

#  Main code loop
while cam.isOpened():
    camera_success, img = cam.read()
    # verify_cam_success(camera_success)

    img, all_hands, success = find_hand_coordinates(img, mirror=True)

    cv2.rectangle(img, (50, 50), (100, 100), keyborad_color, 2)
    cv2.putText(img, 'X', (65, 85), cv2.FONT_HERSHEY_PLAIN, 2, WHITE, 3)


    if len(all_hands) == 1:
        hand1_finger_info = count_raised_fingers(all_hands[0])

        if all_hands[0]['side'] == 'Left':
            for index_line, keyboard_line in enumerate(keys):
                for index, key in enumerate(keyboard_line):
                    img = print_keyboard(img, (offset+index*60, offset+index_line*60), key)
        
        if all_hands[0]['side'] == 'Right':
            if hand1_finger_info == [True, False, False, True] and rock_and_roll == False:
                rock_and_roll = True
                print("Rock and roll baby")
                #time.sleep(2)
                #webbrowser.open("https://www.youtube.com/watch?v=pAgnJDJN4VA")
            


        
    cv2.imshow("Image", img)
    # success_terminal(success)

    

    key = cv2.waitKey(1)
    if key == 27: #esc key
        break
