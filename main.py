import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        h, w, _ = image.shape
        pontos = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
                #Fazendo a contagem de dedos levantados
                for id, cord in enumerate(hand_landmarks.landmark):
                    cx, cy = int(cord.x*h), int(cord.y*w)
                    #cv2.putText(image, str(id), (cx, cy +10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),2)
                    pontos.append((cx, cy))
                    
            # Inserindo a qtd de dedos na tela
            dedos = [8,12,16,20]
            contador = 0
            if pontos[4][0] > pontos[3][0]:
                contador += 1
            if hand_landmarks:
                for x in dedos:
                    if pontos[x][1] < pontos[x - 2][1]:
                        contador += 1
            cv2.putText(image, str(contador),(100,100), cv2.FONT_HERSHEY_SIMPLEX, 4 ,(255,0,0), 5)
            
        image = cv2.flip(image, 1)

        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()