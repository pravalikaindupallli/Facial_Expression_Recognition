# 😄 Facial Expression Recognition System

## 📌 Project Overview

This project is a real-time **Facial Expression Recognition System** built using Deep Learning techniques. It detects faces in live video and classifies facial expressions into categories such as **Happy**, **Sad**, **Angry**, **Surprised**, **Neutral**, and more.

The system is designed to assist in building emotionally intelligent applications by recognizing human emotions through facial expressions.

---

## 🎯 Objectives

- Detect human faces from real-time webcam input.
- Classify facial expressions accurately using a trained deep learning model.
- Display the recognized expression in real-time for user feedback.

---

## 🛠️ Technologies Used

- **Python**  
- **OpenCV** – for face detection and video processing  
- **TensorFlow / Keras** – for building and training CNN model  
- **NumPy, Matplotlib** – for data handling and visualization  

---

## 📂 Dataset

- **FER-2013** dataset (Facial Expression Recognition 2013)  : https://www.kaggle.com/datasets/msambare/fer2013
- Contains over 35,000 labeled grayscale images (48x48 pixels) across 7 emotion classes.

---

## 🧠 Model Architecture

- Convolutional Neural Network (CNN) with multiple Conv2D, MaxPooling, and Dense layers.
- Activation functions: ReLU and Softmax.
- Loss function: Categorical Cross-Entropy.
- Optimizer: Adam.

---

## 📊 Performance

- Achieved an accuracy of **~70%** on validation data.
- Tested on real-time input with consistent results across common facial expressions.

---
