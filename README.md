# 🧠 Brain Tumor Detection using Machine Learning

## 📌 Overview

This project focuses on detecting brain tumors from MRI images using machine learning techniques. The goal is to assist in early diagnosis by classifying images into tumor and non-tumor categories.

---

## 🚀 Features

* Image classification for brain tumor detection
* Data preprocessing and feature extraction
* Machine Learning models for prediction
* Simple and easy-to-run Python scripts

---

## 🛠️ Tech Stack

* Python
* NumPy, Pandas
* Scikit-learn
* OpenCV
* Matplotlib

---

## 📂 Project Structure

```
brain-tumor-detection/
│
├── train_model.py      # Script to train the model
├── predict.py          # Script to make predictions
├── dataset/            # Dataset (not included in repo)
├── output/             # Saved models (ignored in repo)
├── .gitignore
└── README.md
```

---

## 📊 Dataset

The dataset used in this project contains MRI images categorized into:

* Tumor
* No Tumor

⚠️ The dataset is not included in this repository due to size limitations.

👉 You can download it from:
[Add your dataset link here]

---

## ⚙️ How to Run

### 1️⃣ Clone the repository

```
git clone https://github.com/ishita2311/brain-tumor-detection.git
cd brain-tumor-detection
```

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Add dataset

* Download dataset from the link above
* Place it inside the `dataset/` folder

### 4️⃣ Train the model

```
python train_model.py
```

### 5️⃣ Run prediction

```
python predict.py
```

---

## 📈 Model Details

* Feature extraction using PCA
* Classification using Support Vector Machine (SVM)
* Trained on labeled MRI image dataset

---

## 🎯 Results

* Achieved good accuracy in classifying tumor vs non-tumor images
  90-95%

