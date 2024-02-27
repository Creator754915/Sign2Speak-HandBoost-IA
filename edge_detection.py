import cv2
import os
import time

def capture_and_save_image():
    # Ouvrir la capture vidéo de la caméra
    cap = cv2.VideoCapture(0)
    
    # Vérifier si la caméra est ouverte
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la caméra.")
        return
    
    # Créer le dossier "Hand" s'il n'existe pas
    if not os.path.exists("Hand"):
        os.makedirs("Hand")
    
    # Initialiser le compteur d'images
    image_count = 1
    
    while True:
        # Lire une frame de la caméra
        ret, frame = cap.read()
        
        # Vérifier si la frame a été correctement lue
        if not ret:
            print("Erreur: Impossible de lire la frame.")
            break
        
        # Sauvegarder l'image dans le dossier "Hand" avec le nom formaté
        image_path = os.path.join("Hand", f"hand_{image_count}.jpg")
        cv2.imwrite(image_path, frame)
        print(f"Image sauvegardée: {image_path}")
        
        # Attendre que la prochaine image soit capturée
        time.sleep(5)
        
        # Capturer la prochaine frame
        ret, next_frame = cap.read()
        
        # Vérifier si la frame a été correctement lue
        if not ret:
            print("Erreur: Impossible de lire la frame.")
            break
        
        # Exécuter une détection de contours entre les deux images capturées
        edges = detect_edges(frame, next_frame)
        edges_path = os.path.join("Hand", f"hand_{image_count}_edges.jpg")
        cv2.imwrite(edges_path, edges)
        print(f"Contours détectés sauvegardés: {edges_path}")
        
        # Incrémenter le compteur d'images
        image_count += 1
        
        # Attendre 1 milliseconde par défaut pour la touche 'q' pour quitter
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Libérer la capture vidéo et fermer les fenêtres OpenCV
    cap.release()
    cv2.destroyAllWindows()

def detect_edges(prev_frame, curr_frame):
    # Convertir les images en niveaux de gris
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Appliquer un flou gaussien pour réduire le bruit
    prev_blurred = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    curr_blurred = cv2.GaussianBlur(curr_gray, (5, 5), 0)
    
    # Calculer la différence absolue entre les deux images
    diff = cv2.absdiff(prev_blurred, curr_blurred)
    
    # Appliquer un seuillage adaptatif pour obtenir les contours
    edges = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    
    return edges

if __name__ == "__main__":
    capture_and_save_image()
