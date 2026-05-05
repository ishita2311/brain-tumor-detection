print("🔥 File is running")
import os
import cv2
import joblib
import numpy as np

class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

    PCA_MODEL_PATH = os.path.join(OUTPUT_DIR, 'pca_model.pkl')
    SVM_MODEL_PATH = os.path.join(OUTPUT_DIR, 'svm_model.pkl')

    IMG_SIZE = (128, 128)
    CLASSES = ['No Tumor', 'Tumor']

def preprocess_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError("Image not found")

    img = cv2.imread(path, 0)
    img = cv2.resize(img, Config.IMG_SIZE)
    img = img.flatten() / 255.0

    return np.array(img).reshape(1, -1)

def main():
    path = input("Enter image path: ").strip()

    if not os.path.isabs(path):
        path = os.path.join(Config.BASE_DIR, path)

    print("📦 Loading models...")
    pca = joblib.load(Config.PCA_MODEL_PATH)
    svm = joblib.load(Config.SVM_MODEL_PATH)

    img = preprocess_image(path)
    img_pca = pca.transform(img)

    pred = svm.predict(img_pca)[0]

    print("\n🎯 RESULT:", Config.CLASSES[pred])

if __name__ == "__main__":
    print("🚀 Prediction script started")
    main()