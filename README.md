# 🔥 Fire Detection using Deep Learning

A real-time fire detection system using **Convolutional Neural Networks (CNNs)** with **TensorFlow/Keras** and **OpenCV**. The system detects fire from live webcam feed and displays the result with a confidence score.

---

## 🛠 Technologies Used
- **Python**  
- **TensorFlow / Keras**  
- **OpenCV**  
- **NumPy**  
- **Convolutional Neural Networks (CNNs)**  

---

## ⚡ Features
- Real-time fire detection from webcam feed  
- Displays class label (`Fire` / `No Fire`) with confidence percentage  
- Color-coded output:  
  - 🔴 Red for Fire  
  - 🟢 Green for No Fire  

---
## 🧠 How It Works

The webcam captures frames in real-time.

Each frame is resized to 224x224 (matching the model input).

Frame data is normalized and passed to the CNN model.

The model predicts whether the frame contains fire or not.

The result is displayed on the video feed with a confidence score and color-coded label.

## 🔗 References

TensorFlow Keras Documentation

OpenCV Documentation

CNN models for image classification

## ⚡ Future Improvements

Add fire location bounding boxes

Integrate alarm system for real fire alerts

Train on larger datasets for better accuracy

## 💻 How to Run

1. Clone this repository:
```bash
git clone https://github.com/Rou-maissa/Fire_Detection.git
