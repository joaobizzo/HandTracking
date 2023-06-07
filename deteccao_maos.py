#not working

import cv2
import mediapipe as mp


mp_maos = mp.solutions.hands
mp_desenho = mp.solutions.drawing_utils
maos = mp_maos.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)

cor_linha = (224, 208, 64)
cor_bola = (255, 255, 255)

#por padrao, 0 ou 1
#usuarios mac, podem usar a camera do celular, basta testar as duas opcoes (0 e 1)
camera = cv2.VideoCapture()

#resolucao da imagem
resolucao_x = 1280
resolucao_y = 720
camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolucao_x)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolucao_y)


terminal = "Camera iniciada"
print(terminal)


def encontra_coord_maos(img):
    #espelhar imagem
    img = cv2.flip(img, 1) 
    #converter de RGB para BGR(padrao opencv)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resultado = maos.process(img_rgb)

    if resultado.multi_hand_landmarks:
        for marcacoes_maos in resultado.multi_hand_landmarks:
            for marcacao in marcacoes_maos.landmark:
                coord_x, coord_y, coord_z = int(marcacao.x * resolucao_x), int(marcacao.y * resolucao_y), int(marcacao.z * resolucao_x)
                #print(coord_x, coord_y, coord_z)
            mp_desenho.draw_landmarks(img, marcacoes_maos, mp_maos.HAND_CONNECTIONS,
                                    mp_desenho.DrawingSpec(color=cor_bola, thickness=2, circle_radius=2),
                                    mp_desenho.DrawingSpec(color=cor_linha, thickness=2, circle_radius=2))
        terminal = "Mao encontrada"
    else:
        terminal = "Mao nao encontrada"
    
    return img





#loop principal do codigo
while camera.isOpened():

    sucesso, img = camera.read()
    img = encontra_coord_maos(img)
    
    cv2.imshow("Imagem", img)

    tecla = cv2.waitKey(1)
    if tecla == 27:
        break
