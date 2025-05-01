# CAPTCHA Recognition using CNN

This project implements a Convolutional Neural Network (CNN) to recognize CAPTCHA images. It's designed as a learning project for the RTP-B14 team.

## Table of Contents
- [Project Description](#project-description)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributors](#contributors)
- [License](#license)

## Project Description
The goal of this project is to build a CNN model that can recognize characters in CAPTCHA images. The model is trained to predict individual characters in the CAPTCHA, handling both letters and digits.

## Dataset
The project uses a custom dataset of CAPTCHA images. Each image contains 5-6 characters (both letters and digits) with some noise and distortion typical of CAPTCHAs.

## Requirements
- Python 3.6+
- TensorFlow 2.x
- Keras
- OpenCV
- NumPy
- Matplotlib

## Installation
1. Clone the repository:
```bash
git clone https://github.com/omsudhamsh/captcha-recognition-using-cnn-rtp-b14.git
cd captcha-recognition-using-cnn-rtp-b14
```