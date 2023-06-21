import cv2
import mediapipe as mp
import pyautogui


#definindo parametros da deteccao
num_hands = 2
min_detection_confidence = 0.8
min_tracking_confidence = 0.5
#parametros desenho
line_color = (224, 208, 64)
ball_color = (255, 255, 255)
purple = (255, 0, 255)
black = (10, 10, 10)



mp_hands = mp.solutions.hands
mp_draws = mp.solutions.drawing_utils
maos = mp_hands.Hands(min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)



#por padrao, 0 ou 1
#usuarios mac, podem usar a camera do celular, basta testar as duas opcoes (0 e 1)
cam = cv2.VideoCapture(0)

#resolucao da imagem
resolution_x = 1280
resolution_y = 720
cam.set(cv2.CAP_PROP_FRAME_WIDTH, resolution_x)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution_y)


# funcao para verificar se a camera foi iniciada
def verify_cam_sucess(sucesso_camera):
    if sucesso_camera == True:
        terminal = "Camera iniciada"
        print(terminal)
    else:
        terminal = "Erro ao iniciar camera"
        print(terminal)

# funcao para informar se a mao foi detectada corretamente
def terminal_sucesso(sucesso):
    if sucesso == True:
        terminal = "Mao detectada"
        print(terminal)
    else:
        terminal = "Mao nao detectada"
        print(terminal)



# funcao principal encontrar as coordenadas das maos
def encontra_coord_maos(img, espelho = False):
    if espelho:
        #espelhar imagem
        img = cv2.flip(img, 1) 
    #converter de RGB para BGR(padrao opencv)
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
                #print(coord_x, coord_y, coord_z)
            info_mao["coordenadas"] = coordenadas

            todas_maos.append(info_mao)
            mp_draws.draw_landmarks(img, marcacoes_maos, mp_hands.HAND_CONNECTIONS,
                                    mp_draws.DrawingSpec(color=ball_color, thickness=2, circle_radius=2),
                                    mp_draws.DrawingSpec(color=line_color, thickness=2, circle_radius=2))
        sucesso = True
    else:
        sucesso = False
    
    return img, todas_maos, sucesso


# funcao para definir se a mao esta aberta ou fechada
def dedos_levantados(mao):
    dedos = []
    for ponta_dedo in 8, 12, 16, 20:
        if mao['coordenadas'][ponta_dedo][1] < mao['coordenadas'][ponta_dedo - 2][1]:
            dedos.append(True)
        else:
            dedos.append(False)
    return dedos



#loop principal do codigo
while cam.isOpened():

    sucesso_camera, img = cam.read()
    #verify_cam_sucess(sucesso_camera)
    img, todas_maos, sucesso = encontra_coord_maos(img, espelho=True)

    

    if len(todas_maos) == 1:

        thumb_tip = todas_maos[0]['coordenadas'][4]
        index_finger_tip = todas_maos[0]['coordenadas'][8]
        # Calculate the Euclidean distance between thumb and index finger tips
        distance = ((thumb_tip[0] - index_finger_tip[0])**2 + (thumb_tip[1] - index_finger_tip[1])**2)**0.5

        # Print the distance
        print("Distance between thumb and index finger:", distance)


        info_dedos_mao1 = dedos_levantados(todas_maos[0])

        #calculate the center of the hand
        hand_x_center = todas_maos[0]['coordenadas'][9][0]
        hand_y_center = todas_maos[0]['coordenadas'][9][1]
        
        if info_dedos_mao1 == [True, True, True, True]:
            cv2.circle(img, (hand_x_center, hand_y_center), 10, purple, -1)

            x, y = hand_x_center, hand_y_center
            current_x, current_y = pyautogui.position()
            delta_x = x - current_x
            delta_y = y - current_y
            acceleration = 1.5  # Adjust this value to control the acceleration
            
            while abs(delta_x) > 1 or abs(delta_y) > 1:
                delta_x *= acceleration
                delta_y *= acceleration
                x = current_x + delta_x
                y = current_y + delta_y
                pyautogui.moveTo(x, y, duration=0.3)
                current_x, current_y = pyautogui.position()
                delta_x = x - current_x
                delta_y = y - current_y
            if distance < 30:
                pyautogui.click()
        

      
            
        
    
    cv2.imshow("Imagem", img)
    #terminal_sucesso(sucesso)

    tecla = cv2.waitKey(1)
    if tecla == 27: # tecla esc
        break
    
