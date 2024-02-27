import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np

# Initialiser MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Charger le modèle TensorFlow pour la reconnaissance de la langue des signes
model = tf.keras.models.load_model('modele_langue_des_signes.h5')

# Dictionnaire pour mapper les indices de classe aux lettres correspondantes
class_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

# Capturer la vidéo en utilisant OpenCV
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détecter les mains dans l'image
    results = hands.process(gray)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Récupérer les coordonnées de points clés des mains
            hand_landmarks_list = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

            # Prétraitement des données
            hand_image = cv2.resize(gray, (224, 224))  # Adapter à la taille attendue par le modèle
            hand_image = hand_image.reshape(-1, 224, 224, 1)  # Adapter la forme pour le modèle

            # Faire une prédiction avec le modèle TensorFlow
            prediction = model.predict(hand_image)
            predicted_class = np.argmax(prediction)

            # Récupérer la lettre prédite
            predicted_letter = class_to_letter[predicted_class]

            # Afficher la lettre prédite
            cv2.putText(frame, predicted_letter, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Afficher la vidéo en temps réel
    cv2.imshow('Sign Language Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
