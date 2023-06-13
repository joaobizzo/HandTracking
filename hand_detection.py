import cv2
import mediapipe as mp

# defining detection parameters
num_hands = 2
min_detection_confidence = 0.8
min_tracking_confidence = 0.5

# drawing parameters
line_color = (224, 208, 64)
ball_color = (255, 255, 255)

mp_hands = mp.solutions.hands
mp_draws = mp.solutions.drawing_utils
hands = mp_hands.Hands(num_hands=num_hands, min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)

# by default, 0 or 1
# Mac users can use their phone camera by testing both options (0 and 1)
cam = cv2.VideoCapture()

# image resolution
resolution_x = 1280
resolution_y = 720
cam.set(cv2.CAP_PROP_FRAME_WIDTH, resolution_x)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution_y)

# function to verify if the camera was initialized successfully
def verify_cam_success(camera_success):
    if camera_success == True:
        terminal = "Camera initialized"
        print(terminal)
    else:
        terminal = "Error initializing camera"
        print(terminal)

# function to inform if the hand was detected correctly
def terminal_success(success):
    if success == True:
        terminal = "Hand detected"
        print(terminal)
    else:
        terminal = "Hand not detected"
        print(terminal)

# main function to find hand coordinates
def find_hand_coords(img, mirror=False):
    if mirror: # flip image
        img = cv2.flip(img, 1)
    # convert from RGB to BGR (OpenCV default)
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
            if mirror:
                if hand_side.classification[0].label == "Left":
                    hand_info["side"] = "Right"
                else:
                    hand_info["side"] = "Left"
            else:
                hand_info["side"] = hand_side.classification[0].label

            print(hand_info["side"])

            all_hands.append(hand_info)
            mp_draws.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                    mp_draws.DrawingSpec(color=ball_color, thickness=2, circle_radius=2),
                                    mp_draws.DrawingSpec(color=line_color, thickness=2, circle_radius=2))
        success = True
    else:
        success = False

    return img, hand_side, all_hands, success

# function to determine if the hand is open or closed
def fingers_raised(hand):
    fingers = []
    for finger_tip in [8, 12, 16, 20]:
        if hand['coordinates'][finger_tip][1] < hand['coordinates'][finger_tip - 2][1]:
            fingers.append(True)
        else:
            fingers.append(False)
        return fingers

# main loop of the code
while cam.isOpened():
    camera_success, img = cam.read()
    verify_cam_success(camera_success)
    img, hand_side, all_hands, success = find_hand_coords(img, mirror=True)

    if len(all_hands) != 0:
        hand1_finger_info = fingers_raised(all_hands[0])
        print(hand1_finger_info)

    cv2.imshow("Image", img)
    terminal_success(success)

    key = cv2.waitKey(1)
    if key == 27: # esc key
        break
