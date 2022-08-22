# mediaPipe-tablero
Tablero e implementacion de mediapipe.
Hola el dia de hoy implementaremos un tablero con mediapipe, este es uno de los muchos video que desarrollaremos usando esta herramienta avanzada en vision artificial e inteligencia Artificial.

No olvides suscribirte en Ingenieria JHR https://www.youtube.com/c/INGENIER%C3%8DAJHR, gracias por tu apoyo y tu tiempo.

![image](https://user-images.githubusercontent.com/66834393/185817351-a587fbce-c6f3-45a7-befd-c5e3cb5abe79.png)
![image](https://user-images.githubusercontent.com/66834393/185817367-da6889dc-ee0d-44b7-8768-13ec0cec4d40.png)
![image](https://user-images.githubusercontent.com/66834393/185817369-847310bf-0a38-4024-8092-4cf60a49953d.png)


1) Implementacion MediaPipe.
 ```python
 #container generic detect hands in the mediapipe
import mediapipe as mp
import cv2

# inicializamos la clase Hands y almacenarla en una variable
handsMp = mp.solutions.hands
# cargamos componente con las herramientas que nos permitira dibujar mas adelante
drawingMp = mp.solutions.drawing_utils
# cargamos los estilos en la variable mp_drawing_styles
mp_drawing_styles = mp.solutions.drawing_styles
# iniciamos una captura de video en la camara 1
cap = cv2.VideoCapture(1)
# save height and width image
height, width = [0, 0]


# ejecutamos del bloque de deteccion
with handsMp.Hands(static_image_mode=False,
                   max_num_hands=1,
                   min_detection_confidence=0.5,
                   min_tracking_confidence=0.5) as hands:
    # mientras la camara este en ejecucion
    while cap.isOpened():
        # guardamos en la variable succes el estado de la captura y en image la captura
        success, image = cap.read()
        if not success:
            print("camara vacia")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        # get shape image
        if height == 0:
            # return height, width and channels
            height, width, _ = image.shape
            print(height, width)

        # convertimos la imagen de bgr a rgb debido a que la funcion hands process acepta rgb
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Procesa una imagen RGB y devuelve los puntos de referencia de la mano y la destreza de cada mano detectada
        results = hands.process(image)
        # convertimos la imagen rgb a bgr
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # si obtenemos puntos de referencia multiples
        if results.multi_hand_landmarks is not None:
            # recorremos esos puntos multiples de referencia
            for hand_landmarks in results.multi_hand_landmarks:
                # dibujamos los puntos de referencia (imagen,puntos referencia de la mano,describe las conexiones
                # de los puntos de referencia,
                drawingMp.draw_landmarks(
                    image,
                    hand_landmarks,
                    handsMp.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                #*************************************
                #aqui todas las funciones que considereimos necesarias
                #*************************************
        # Voltee la imagen horizontalmente para obtener una vista de selfie.
        cv2.imshow('MediaPipeJHR', cv2.flip(image, 1))
        # en caso de teclear la letra q suspendemos la operacion
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
# Cierra el archivo de video o el dispositivo de captura.
cap.release()
  ```
  
2) tablero MediaPipe
 ```python
 import math
import mediapipe as mp
import cv2

# inicializamos la clase Hands y almacenarla en una variable
handsMp = mp.solutions.hands
# cargamos componente con las herramientas que nos permitira dibujar mas adelante
drawingMp = mp.solutions.drawing_utils
# cargamos los estilos en la variable mp_drawing_styles
mp_drawing_styles = mp.solutions.drawing_styles
# iniciamos una captura de video en la camara 1
cap = cv2.VideoCapture(1)
# save height and width image
height, width = [0, 0]
# array save line point
arrayLine = []


def verifiResetArray(x1, y1, image):
    cv2.rectangle(image, (0, 0), (70, 70), (255, 0, 0), 4)
    if x1 and y1 < 70:
        arrayLine.clear()


def drawDisplay(hand_landmarks, image):
    # indice
    x1 = int(hand_landmarks.landmark[handsMp.HandLandmark.INDEX_FINGER_TIP].x * width)
    y1 = int(hand_landmarks.landmark[handsMp.HandLandmark.INDEX_FINGER_TIP].y * height)

    # pulgar
    x2 = int(hand_landmarks.landmark[handsMp.HandLandmark.THUMB_TIP].x * width)
    y2 = int(hand_landmarks.landmark[handsMp.HandLandmark.THUMB_TIP].y * height)

    verifiResetArray(x1, y1, image)

    # center position pulgar, indice
    center = int((x1 + x2) / 2), int((y1 + y2) / 2)
    #formula distance two points
    distanceTwoPoint = math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))
    #if distance two point < 30
    if distanceTwoPoint < 30:
        #add point in the array
        arrayLine.append(center)
        # draw circle bgr, img, position, radio,color,espesor
        cv2.circle(image, center, 3, (0, 0, 255), 2)
    # size array
    sizelist = len(arrayLine)
    #iniciamos variable i
    i = 0
    while i < sizelist:
        #if exist un dato adicional que seria endpoint accedemos a dibujar linea
        if i + 1 < sizelist:
            #get start point
            startPoint = arrayLine[i]
            #get end point
            endPoint = arrayLine[i + 1]
            # draw line in position array
            cv2.line(image, startPoint, endPoint, (0, 0, 255), 3)
        #i++
        i += 1


# ejecutamos del bloque de deteccion
with handsMp.Hands(static_image_mode=False,
                   max_num_hands=1,
                   min_detection_confidence=0.5,
                   min_tracking_confidence=0.5) as hands:
    # mientras la camara este en ejecucion
    while cap.isOpened():
        # guardamos en la variable succes el estado de la captura y en image la captura
        success, image = cap.read()
        if not success:
            print("camara vacia")
            # If loading a video, use 'break' instead of 'continue'.
            continue
         #get shape image
        if height == 0:
            #return height, width and channels
            height, width, _ = image.shape
            print(height, width)

        # convertimos la imagen de bgr a rgb debido a que la funcion hands process acepta rgb
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Procesa una imagen RGB y devuelve los puntos de referencia de la mano y la destreza de cada mano detectada
        results = hands.process(image)
        # convertimos la imagen rgb a bgr
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # si obtenemos puntos de referencia multiples
        if results.multi_hand_landmarks is not None:
            # recorremos esos puntos multiples de referencia
            for hand_landmarks in results.multi_hand_landmarks:
                # dibujamos los puntos de referencia (imagen,puntos referencia de la mano,describe las conexiones
                # de los puntos de referencia,
                drawingMp.draw_landmarks(
                    image,
                    hand_landmarks,
                    handsMp.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                #draw display fun paint
                drawDisplay(hand_landmarks, image)
        # Voltee la imagen horizontalmente para obtener una vista de selfie.
        cv2.imshow('MediaPipeJHR', cv2.flip(image, 1))
        # en caso de teclear la letra q suspendemos la operacion
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
# Cierra el archivo de video o el dispositivo de captura.
cap.release()
 
  ```
