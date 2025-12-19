# 🌱 Plant Disease Detection and Advisory System

## 📌 Project Overview
The **Plant Disease Detection and Advisory System** is an AI-powered application designed to help farmers and agricultural professionals identify plant diseases at an early stage using leaf images. The system not only detects diseases using deep learning but also provides detailed treatment recommendations, preventive measures, and safety guidelines in a user-friendly interface.

This project aims to reduce crop loss, improve agricultural productivity, and support smart farming practices.

---

## 🎯 Objectives
- Detect plant diseases accurately using image classification
- Assist farmers with proper treatment and preventive guidance
- Support multiple crops and disease classes
- Provide an easy-to-use and farmer-friendly interface
- Promote early disease identification to minimize crop damage

---

## 🚀 Key Features
- 🌿 Multi-crop plant disease detection
- 🧠 Deep Learning-based image classification
- 📊 Prediction confidence score
<!-- - 🔍 Top-3 disease predictions -->


- 💊 Disease description and medicine recommendations

- ⚠️ Safety precautions and dosage instructions
- 🗣️ Bangla and English language support
- 💬 AI-powered advisory chatbot
- 📱 Mobile-friendly Streamlit web interface
- 🕒 Prediction history tracking
- 📈 Disease trend analysis 

---

## 🛠️ Technologies Used
- **Python**
- **TensorFlow / Keras**
- **MobileNetV2 (Transfer Learning)**
- **Streamlit**
- **NumPy & Pandas**
- **OpenCV**
- **Matplotlib**
- **Gemini AI (for advisory chatbot)**


---

## 🧪 Dataset
- Publicly available **Plant Disease Image Dataset**
- Contains healthy and diseased leaf images
- Images are preprocessed and augmented to improve model performance
- Supports multiple plant species and disease classes

---

## 🧠 Model Architecture
- Base Model: **MobileNetV2**
- Transfer learning with frozen base layers
- Custom classification layers added on top
- Fine-tuning applied to last layers for better accuracy
- Trained using categorical cross-entropy loss

---

## 📊 Model Performance
- Accuracy: 90%
- Precision, Recall, and F1-score evaluated
- Confusion matrix used for performance analysis
*(Update metrics after final training)*

---

## 💊 Advisory System
For each detected disease, the system provides:
- Disease name and description
- Symptoms and causes
- Recommended medicines (fungicides/pesticides)
- Organic treatment options
- Preventive measures
- Safety instructions and waiting period before harvest

---

## 🌐 Application Workflow
1. User uploads a plant leaf image
2. Image is preprocessed and passed to the model
3. Disease is predicted with confidence score
4. Heatmap highlights affected areas
5. Advisory system displays treatment and prevention details

---

## 🖥️ Installation & Setup

### 🔹 Prerequisites
- Python 3.8+
- pip

### 🔹 Installation Steps
```bash
git clone https://github.com/Shihab119/Plant-diseases-detection
cd plant-disease-detection
pip install -r requirements.txt
