import cv2
import mediapipe as mp
import pyautogui
import os
import time

# Definindo parâmetros da detecção
min_detection_confidence = 0.8
min_tracking_confidence = 0.5

# Parâmetros de desenho
line_color = (224, 208, 64)
ball_color = (255, 255, 255)
purple = (255, 0, 255)
black = (10, 10, 10)

mp_hands = mp.solutions.hands
mp_draws = mp.solutions.drawing_utils
maos = mp_hands.Hands(min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)

# Inicialize o tempo de verificação anterior
prev_verification_time = time.time()

# Por padrão, 0 ou 1
# Usuários Mac podem usar a câmera do celular, basta testar as duas opções (0 e 1)
cam = cv2.VideoCapture(0)

# Resolução da imagem
resolution_x = 640
resolution_y = 480
cam.set(cv2.CAP_PROP_FRAME_WIDTH, resolution_x)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution_y)

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

# Função principal para encontrar as coordenadas das mãos
def encontra_coord_maos(img, espelho=False):
    if espelho:
        # Espelhar imagem
        img = cv2.flip(img, 1)
    # Converter de RGB para BGR (padrão OpenCV)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resultado = maos.process(img_rgb)
    todas_maos = []

    if resultado.multi_hand_landmarks:
        for lado_mao, marcacoes_maos in zip(resultado.multi_handedness, resultado.multi_hand_landmarks):
            info_mao = {}
            coordenadas = []
            for marcacao in marcacoes_maos.landmark:
                coord_x, coord_y, coord_z = int(marcacao.x * resolution_x), int(marcacao.y * resolution_y), int(marcacao.z * resolution_x)
                coordenadas.append([coord_x, coord_y, coord_z])
            info_mao["coordenadas"] = coordenadas

            todas_maos.append(info_mao)
            mp_draws.draw_landmarks(img, marcacoes_maos, mp_hands.HAND_CONNECTIONS,
                                    mp_draws.DrawingSpec(color=ball_color, thickness=2, circle_radius=2),
                                    mp_draws.DrawingSpec(color=line_color, thickness=2, circle_radius=2))
        sucesso = True
    else:
        sucesso = False
    
    return img, todas_maos, sucesso


# Função para definir se a mão está aberta ou fechada
def dedos_levantados(mao):
    dedos = []
    for ponta_dedo in 8, 12, 16, 20:
        if mao['coordenadas'][ponta_dedo][1] < mao['coordenadas'][ponta_dedo - 2][1]:
            dedos.append(True)
        else:
            dedos.append(False)
    return dedos


# Loop principal do código
frame_count = 0

while cam.isOpened():
    sucesso_camera, img = cam.read()
    img, todas_maos, sucesso = encontra_coord_maos(img, espelho=True)
    
    frame_count += 1
    if frame_count % 5 == 0:  # Desenhar apenas a cada 5 quadros
        if len(todas_maos) == 1:
            thumb_tip = todas_maos[0]['coordenadas'][4]
            index_finger_tip = todas_maos[0]['coordenadas'][8]
            # Calculate the Euclidean distance between thumb and index finger tips
            distance = ((thumb_tip[0] - index_finger_tip[0])**2 + (thumb_tip[1] - index_finger_tip[1])**2)**0.5

            info_dedos_mao1 = dedos_levantados(todas_maos[0])

            # Calculate the center of the hand
            hand_x_center = todas_maos[0]['coordenadas'][9][0]
            hand_y_center = todas_maos[0]['coordenadas'][9][1]
            
            cv2.circle(img, (hand_x_center, hand_y_center), 10, purple, -1)

            x, y = hand_x_center, hand_y_center
            current_x, current_y = pyautogui.position()
            pyautogui.moveTo(x, y, duration=0.8, tween=pyautogui.easeInOutQuad)
            if distance < 30:
                pyautogui.click()
                cv2.circle(img, (hand_x_center, hand_y_center), 10, black, -1)
    
    clear()
    # Calculate the response time
    current_time = time.time()
    response_time = current_time - prev_verification_time
    prev_verification_time = current_time

    print("Response time:", response_time) if response_time > 0.5 else print("Response time: OK")
    cv2.imshow("Imagem", img)
    
    tecla = cv2.waitKey(1)
    if tecla == 27:  # Tecla Esc
        break

cam.release()
cv2.destroyAllWindows()
