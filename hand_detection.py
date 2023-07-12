import cv2
import mediapipe as mp
import os
import datetime


#defining detection parameters
min_detection_confidence = 0.8
min_tracking_confidence = 0.5
#drawing parameters
line_color = (224, 208, 64)
ball_color = (255, 255, 255)
purple = (255, 0, 255)
black = (10, 10, 10)

mp_hands = mp.solutions.hands
mp_draws = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)

#initialize the previous verification time
prev_verification_time = datetime.datetime.now()

#by default, 0 or 1
#Mac users can use their phone camera by testing both options (0 and 1)
cam = cv2.VideoCapture(0)

#image resolution
resolution_x = 1280
resolution_y = 720
cam.set(cv2.CAP_PROP_FRAME_WIDTH, resolution_x)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution_y)

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

# function to inform if the hand was detected correctly
def terminal_sucess(success):
    if success:
        terminal = "Hand detected"
    else:
        terminal = "Hand not detected"
    print(terminal)


# main function to find hand coordinates
def find_hand_coordinates(img, mirror=False):
    if mirror:
        # mirror the image
        img = cv2.flip(img, 1)
    # Convert from RGB to BGR (OpenCV default)
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
            hand_info["coordinates"] = coordinates

            all_hands.append(hand_info)
            mp_draws.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                    mp_draws.DrawingSpec(color=ball_color, thickness=2, circle_radius=2),
                                    mp_draws.DrawingSpec(color=line_color, thickness=2, circle_radius=2))
        sucess = True
    else:
        sucess = False
    
    return img, all_hands, sucess


# function to determine if the hand is open or closed
def fingers_raised(hand):
    fingers = []
    for finger_tip in 8, 12, 16, 20:
        if hand['coordinates'][finger_tip][1] < hand['coordinates'][finger_tip - 2][1]:
            fingers.append(True)
        else:
            fingers.append(False)
    return fingers


# main loop of the cod
while cam.isOpened():
    sucess_camera, img = cam.read()
    img, all_hands, sucess = find_hand_coordinates(img, mirror=True)
    
    if len(all_hands) == 1:
        thumb_tip = all_hands[0]['coordinates'][4]
        index_finger_tip = all_hands[0]['coordinates'][8]
        # Calculate the Euclidean distance between thumb and index finger tips
        distance = ((thumb_tip[0] - index_finger_tip[0])**2 + (thumb_tip[1] - index_finger_tip[1])**2)**0.5

        hand_fingers_info = fingers_raised(all_hands[0])

        #calculate the center of the hand
        hand_x_center = all_hands[0]['coordinates'][9][0]
        hand_y_center = all_hands[0]['coordinates'][9][1]
        
        cv2.circle(img, (hand_x_center, hand_y_center), 10, purple, -1)
        
        
    
    clear()
    # Calculate the response time
    current_time = datetime.datetime.now()
    response_time = (current_time - prev_verification_time).total_seconds()
    prev_verification_time = current_time

    print("Response time:", response_time)
    cv2.imshow("Image", img)
    
    key = cv2.waitKey(1)
    if key == 27: #  esc
        break

cam.release()
cv2.destroyAllWindows()
