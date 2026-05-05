# 🧠 Brain Tumor Detection using Machine Learning

## 📌 Overview

This project focuses on detecting brain tumors from MRI images using machine learning techniques. The system classifies brain MRI scans into tumor and non-tumor categories to assist in early diagnosis.

---

## 🚀 Features

* Brain tumor classification using MRI images
* Data preprocessing and feature extraction
* Dimensionality reduction using PCA
* Classification using Support Vector Machine (SVM)
* Easy-to-run Python scripts

---

## 🛠️ Tech Stack

* Python
* NumPy
* Pandas
* Scikit-learn
* OpenCV
* Matplotlib

---

## 📂 Project Structure

```
brain-tumor-detection/
│
├── train_model.py      # Train the ML model
├── predict.py          # Predict tumor from input image
├── dataset/            # Dataset (not included)
├── output/             # Saved models (ignored)
├── .gitignore
└── README.md
```

---

## 📊 Dataset

The dataset used in this project contains MRI images categorized into:

* Tumor
* No Tumor

⚠️ Due to large size, the dataset is not included in this repository.

👉 Download Dataset:
https://drive.google.com/drive/folders/1Rpsx1bfcbYjBDvkKibnKAJDaxhRMyGYv?usp=sharing

---

## 📁 Dataset Structure

After downloading and extracting, organize the dataset as follows:

```
dataset/
 ├── training/
 │    ├── tumor/
 │    └── no_tumor/
 ├── testing/
 │    ├── tumor/
 │    └── no_tumor/
```

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
* Extract it
* Place inside `dataset/` folder

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

* Feature extraction using PCA (Principal Component Analysis)
* Classification using Support Vector Machine (SVM)
* Trained on labeled MRI image dataset

---

## 🎯 Results

* Successfully classifies MRI images into tumor and non-tumor categories
97%
---

## 📌 Future Improvements

* Implement Deep Learning (CNN) for higher accuracy
* Build a web-based interface
* Use a larger and more diverse dataset



