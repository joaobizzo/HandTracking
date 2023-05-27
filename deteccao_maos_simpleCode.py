import cv2
import mediapipe as mp


mp_maos = mp.solutions.hands
mp_desenho = mp.solutions.drawing_utils
maos = mp_maos.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)

cor_linha = (224, 208, 64)
cor_bola = (255, 255, 255)

#por padrao, 0 ou 1
#usuarios mac, podem usar a camera do celular, basta testar as duas opcoes (0 e 1)
camera = cv2.VideoCapture(0)

#resolucao da imagem
resolucao_x = 1280
resolucao_y = 720
camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolucao_x)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolucao_y)
print("camera iniciada")
terminal = "camera nao encontrada"



while camera.isOpened():

    sucesso, img = camera.read()

    #espelhar imagem
    img = cv2.flip(img, 1) 

    #converter de RGB para BGR(padrao opencv)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    resultado = maos.process(img_rgb)

    if resultado.multi_hand_landmarks:
        for marcacao_maos in resultado.multi_hand_landmarks:
            mp_desenho.draw_landmarks(img, marcacao_maos, mp_maos.HAND_CONNECTIONS,
                                    mp_desenho.DrawingSpec(color=cor_bola, thickness=2, circle_radius=2),
                                    mp_desenho.DrawingSpec(color=cor_linha, thickness=2, circle_radius=2))
        if terminal != "mao encontrada":
            print("mao encontrada")
            terminal = "mao encontrada"
    else:
        terminal = "mao nao encontrada"
        print(terminal)
    
    cv2.imshow("Imagem", img)

    tecla = cv2.waitKey(1)
    if tecla == 27:
        break
