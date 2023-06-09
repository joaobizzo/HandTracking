#importando bibliotecas
import cv2
import mediapipe as mp

#definindo parametros da deteccao
num_hands = 2
min_detection_confidence = 0.8
min_tracking_confidence = 0.5
#parametros desenho
cor_linha = (224, 208, 64)
cor_bola = (255, 255, 255)



mp_maos = mp.solutions.hands
mp_desenho = mp.solutions.drawing_utils
maos = mp_maos.Hands(num_hands = num_hands, min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)



#por padrao, 0 ou 1
#usuarios mac, podem usar a camera do celular, basta testar as duas opcoes (0 e 1)
camera = cv2.VideoCapture()

#resolucao da imagem
resolucao_x = 1280
resolucao_y = 720
camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolucao_x)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolucao_y)



def verificar_sucesso_camera(sucesso_camera):
    if sucesso_camera == True:
        terminal = "Camera iniciada"
        print(terminal)
    else:
        terminal = "Erro ao iniciar camera"
        print(terminal)

def terminal_sucesso(sucesso):
    if sucesso == True:
        terminal = "Mao detectada"
        print(terminal)
    else:
        terminal = "Mao nao detectada"
        print(terminal)

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
                coord_x, coord_y, coord_z = int(marcacao.x * resolucao_x), int(marcacao.y * resolucao_y), int(marcacao.z * resolucao_x)
                coordenadas.append([coord_x, coord_y, coord_z])
                #print(coord_x, coord_y, coord_z)
            info_mao["coordenadas"] = coordenadas
            if espelho:
                if lado_mao.classification[0].label == "Left":
                    info_mao["lado"] = "Right"
                else:
                    info_mao["lado"] = "Left"
            else:
                lado_mao["lado"] = lado_mao.classification[0].label
            
            print(info_mao["lado"])

            todas_maos.append(info_mao)
            mp_desenho.draw_landmarks(img, marcacoes_maos, mp_maos.HAND_CONNECTIONS,
                                    mp_desenho.DrawingSpec(color=cor_bola, thickness=2, circle_radius=2),
                                    mp_desenho.DrawingSpec(color=cor_linha, thickness=2, circle_radius=2))
        sucesso = True
    else:
        sucesso = False
    
    return img, lado_mao, todas_maos, sucesso

#loop principal do codigo
while camera.isOpened():

    sucesso_camera, img = camera.read()
    verificar_sucesso_camera(sucesso_camera)
    img, lado_mao, todas_mao, sucesso = encontra_coord_maos(img, espelho=True)
    
    cv2.imshow("Imagem", img)
    terminal_sucesso(sucesso)

    tecla = cv2.waitKey(1)
    if tecla == 27:
        break
