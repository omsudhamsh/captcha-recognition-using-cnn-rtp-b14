# CAPTCHA Recognition using CNN

A deep learning project that uses a Convolutional Neural Network (CNN) to recognize and predict CAPTCHA text from distorted images. This project was developed as a learning initiative for the RTP-B14 team.

---

## 📌 Table of Contents
- [Project Description](#-project-description)
- [Dataset](#-dataset)
- [Features](#-features)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Contributors](#-contributors)
- [License](#-license)

---

## 📖 Project Description

CAPTCHAs are widely used to differentiate humans from bots by displaying distorted text that is difficult for automated systems to read.

The objective of this project is to build a CNN-based image recognition model capable of identifying and predicting characters present in CAPTCHA images. The model is trained on custom CAPTCHA datasets containing both letters and digits with varying levels of distortion and noise.

---

## 🗂 Dataset

This project uses a custom-generated CAPTCHA dataset.

### Dataset Characteristics
- CAPTCHA length: **5–6 characters**
- Includes:
  - Uppercase letters
  - Lowercase letters
  - Digits (0–9)
- Contains:
  - Noise
  - Distortion
  - Random fonts
  - Rotation effects

> Example CAPTCHA: `A7K9P`

---

## ✨ Features

- CNN-based CAPTCHA recognition
- Image preprocessing using OpenCV
- Character prediction using TensorFlow/Keras
- Supports alphanumeric CAPTCHA images
- Visualization of training performance
- Easy-to-understand project structure for beginners

---

## ⚙️ Requirements

Make sure you have the following installed:

- Python 3.6+
- TensorFlow 2.x
- Keras
- OpenCV
- NumPy
- Matplotlib

Install all dependencies using:

```bash
pip install tensorflow keras opencv-python numpy matplotlib
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/captcha-recognition-cnn.git
```

Navigate to the project folder:

```bash
cd captcha-recognition-cnn
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the training script:

```bash
python train.py
```

Run prediction on CAPTCHA images:

```bash
python predict.py
```

---

## 🧠 Model Architecture

The CNN model includes:

- Convolutional Layers
- MaxPooling Layers
- Dropout Layers
- Fully Connected Dense Layers
- Softmax Activation for Character Prediction

### Workflow
1. Load CAPTCHA images
2. Preprocess images
3. Train CNN model
4. Predict CAPTCHA characters
5. Evaluate model performance

---

## 📊 Results

The model achieves good accuracy on custom CAPTCHA datasets and demonstrates the effectiveness of CNNs for image-based text recognition tasks.

Training graphs and prediction samples can be visualized using Matplotlib.

---

## 📁 Project Structure

```bash
captcha-recognition-cnn/
│
├── dataset/
├── models/
├── train.py
├── predict.py
├── requirements.txt
├── README.md
└── outputs/
```

---

## 👥 Contributors

- RTP-B14 Team

---

## 📄 License

This project is intended for educational and learning purposes only.
```
