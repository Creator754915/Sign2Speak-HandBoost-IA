import cv2
import os
import mediapipe as mp

def track_hands(input_folder, output_folder):
    # Créer un objet de détection de mains de Mediapipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)

    # Configuration de l'affichage des mains détectées
    mp_drawing = mp.solutions.drawing_utils

    # Liste des noms de fichiers des images dans le dossier d'entrée
    input_files = os.listdir(input_folder)

    # Créer le dossier de sortie s'il n'existe pas
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Parcourir chaque image dans le dossier d'entrée
    for filename in input_files:
        # Construire le chemin d'accès complet de l'image d'entrée
        input_image_path = os.path.join(input_folder, filename)

        # Lire l'image à partir du chemin d'accès
        image = cv2.imread(input_image_path)

        # Convertir l'image en RGB car Mediapipe utilise des images RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Détecter les mains dans l'image
        results = hands.process(image_rgb)

        # Dessiner les mains détectées sur l'image
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Enregistrer l'image avec le suivi des mains dans le dossier de sortie
        output_image_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_image_path, image)

        print(f"Suivi des mains terminé pour {filename}")

    # Libérer les ressources utilisées par Mediapipe
    hands.close()

if __name__ == "__main__":
    input_folder = "./Hand"
    output_folder = "data/tracking_hands"
    track_hands(input_folder, output_folder)
