import cv2
import mediapipe as mp

# Inicializar MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Inicializar a captura de vídeo.
cap = cv2.VideoCapture(0)

# Variáveis para armazenar a posição anterior do dedo indicador.
prev_x, prev_y = 0, 0

# Cria uma imagem em branco para desenhar.
canvas = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip e converte a imagem para RGB.
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processa a imagem e detecta as mãos.
    results = hands.process(frame_rgb)

    # Inicializa o canvas na primeira execução.
    if canvas is None:
        canvas = frame.copy()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Desenha as marcações das mãos na imagem original.
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Obtém a posição do dedo indicador.
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, c = frame.shape
            index_x, index_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            # Desenha no canvas se o dedo está em movimento.
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = index_x, index_y
            else:
                cv2.line(canvas, (prev_x, prev_y), (index_x, index_y), (255, 0, 0), 5)
                prev_x, prev_y = index_x, index_y
    else:
        # Reseta a posição do dedo quando não for detectado.
        prev_x, prev_y = 0, 0

    # Combina a imagem original com o canvas.
    combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Mostra a imagem.
    cv2.imshow('Write with Index Finger', combined)

    if cv2.waitKey(1) & 0xFF == 27:  # Pressione 'Esc' para sair.
        break

# Libera os recursos.
cap.release()
cv2.destroyAllWindows()
