import cv2
import os
import time

def capture_and_save_image():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la caméra.")
        return
    
    if not os.path.exists("Hand"):
        os.makedirs("Hand")
    
    image_count = 1
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Erreur: Impossible de lire la frame.")
            break
        
        image_path = os.path.join("Hand", f"hand_{image_count}.jpg")
        cv2.imwrite(image_path, frame)
        print(f"Image sauvegardée: {image_path}")
        
        time.sleep(5)
        
        image_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_save_image()
