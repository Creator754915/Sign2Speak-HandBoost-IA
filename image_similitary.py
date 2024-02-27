import cv2 as cv
import os

def compare_images(base_path, test_path):
    base = cv.imread(base_path)
    hsv_base = cv.cvtColor(base, cv.COLOR_BGR2HSV)
    hist_base = cv.calcHist([hsv_base], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv.normalize(hist_base, hist_base, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    
    test = cv.imread(test_path)
    hsv_test = cv.cvtColor(test, cv.COLOR_BGR2HSV)
    hist_test = cv.calcHist([hsv_test], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv.normalize(hist_test, hist_test, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    
    similarity = cv.compareHist(hist_base, hist_test, cv.HISTCMP_CORREL)
    
    return similarity

def main():
    base_folder = "./Hand"
    base_files = os.listdir(base_folder)
    training_folder = "data/training"
    training_subfolders = [os.path.join(training_folder, subfolder) for subfolder in ['A', 'B', 'C', 'D']]
    max_images = 1
    
    with open("word.txt", "w") as file:
        for subfolder in training_subfolders:
            print(f"\nComparaison pour le sous-dossier : {os.path.basename(subfolder)}")
            
            for i in range(1, max_images + 1):
                test_image_path = os.path.join(subfolder, f"{i}.jpg")
                
                if os.path.isfile(test_image_path):
                    for base_file in base_files:
                        base_image_path = os.path.join(base_folder, base_file)
                        similarity = compare_images(base_image_path, test_image_path)
                        print(f"Similarité entre {os.path.basename(base_image_path)} et {os.path.basename(test_image_path)} : {similarity}")
                        
                        if similarity > 0.95:
                            file.write(f"{os.path.basename(base_image_path)} : {os.path.basename(subfolder)}\n")
                    
                    print()
                else:
                    print(f"Fichier {test_image_path} non trouvé, saut de comparaison.")

if __name__ == "__main__":
    main()
