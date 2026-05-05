print("🚀 Script Started")
import os
import cv2
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'dataset')
    TRAIN_DIR = os.path.join(DATA_DIR, 'training')
    TEST_DIR = os.path.join(DATA_DIR, 'testing')

    OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
    PCA_MODEL_PATH = os.path.join(OUTPUT_DIR, 'pca_model.pkl')
    SVM_MODEL_PATH = os.path.join(OUTPUT_DIR, 'svm_model.pkl')

    IMG_SIZE = (128, 128)
    CLASSES = ['no_tumor', 'tumor']

def load_data(path):
    X, y = [], []

    for label, category in enumerate(Config.CLASSES):
        folder = os.path.join(path, category)

        if not os.path.exists(folder):
            print(f"❌ Folder not found: {folder}")
            continue

        for file in tqdm(os.listdir(folder), desc=f"Loading {category}"):
            img_path = os.path.join(folder, file)

            img = cv2.imread(img_path, 0)
            if img is None:
                continue

            img = cv2.resize(img, Config.IMG_SIZE)
            img = img.flatten() / 255.0

            X.append(img)
            y.append(label)

    return np.array(X), np.array(y)

def main():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    print("📂 Loading Training Data...")
    X_train, y_train = load_data(Config.TRAIN_DIR)

    print("📂 Loading Testing Data...")
    X_test, y_test = load_data(Config.TEST_DIR)

    print("🔄 Applying PCA...")
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    joblib.dump(pca, Config.PCA_MODEL_PATH)

    print("🤖 Training SVM...")
    svm = SVC(kernel='rbf')
    svm.fit(X_train_pca, y_train)

    joblib.dump(svm, Config.SVM_MODEL_PATH)

    print("📊 Evaluating Model...")
    y_pred = svm.predict(X_test_pca)

    print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    print("\n✅ Training Completed!")

if __name__ == "__main__":
    print("🔥 Main function is running")
    main()