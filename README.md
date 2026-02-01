System_Control – Hand Gesture Controlled Interface

A modular hand gesture–controlled system that enables mouse control and keyboard input using real-time hand tracking and machine-learning-based gesture recognition.

The project combines:

Rule-based mouse gestures

ML-based gesture classification (ASL, ISL, CSL)

MediaPipe Hands for tracking

TensorFlow / Keras models trained using Teachable Machine

This repository is a refactored and modernised version of an earlier prototype, updated to work smoothly with Python 3.12.

FEATURES

  Mouse movement and clicks using hand gestures
  
  Sign-language gesture recognition:
  
  ASL (American Sign Language)
  
  ISL (Indian Sign Language)
  
  CSL (Custom Sign Language)
  
  Gesture-based switching between modes
  
  Keyboard actions mapped to gestures
  
  Clean and modular Python architecture

PROJECT STRUCTURE

  System_Control
  └── gesture_control
  ├── actions.py (Keyboard and mouse actions)
  ├── camera.py (OpenCV camera wrapper)
  ├── config.py (Global configuration)
  ├── main.py (Application entry point)
  ├── mediapipe_hand.py (MediaPipe Hands interface)
  ├── model_classifier.py (ML model loader and predictor)
  ├── modes.py (Mouse and ML control loops)
  ├── utils.py (Image preprocessing helpers)
  └── assets
  ├── keras_model.keras (CSL model)
  ├── keras_model1.keras (ASL model)
  ├── keras_model2.keras (ISL model)
  ├── labels.txt
  ├── labels1.txt
  └── labels2.txt

REQUIREMENTS

  Python version:
  Python 3.12 (tested)

Python packages (exact versions tested):

  tensorflow 2.17.0
  tf-keras 2.17.0
  mediapipe 0.10.18
  opencv-python 4.9.0.80
  numpy 1.26.4
  protobuf 4.25.8
  pyautogui 0.9.54

INSTALLATION

Step 1: Create a virtual environment (recommended)

  python3.12 -m venv .venv
  source .venv/bin/activate

Step 2: Install dependencies

  pip install -r requirements.txt
  
  RUNNING THE APPLICATION
  
  From the repository root directory:
  
  python gesture_control/main.py

Make sure:

  A webcam is connected
  
  The gesture_control/assets folder contains the .keras model files

NOTES

  TensorFlow may show warnings about CUDA or GPU libraries.
  This is normal when running on CPU.
  
  MediaPipe uses OpenGL internally if available.
  
  Moving the mouse to the top-left corner triggers PyAutoGUI’s safety stop.
  
  CONFIGURATION
  
  All configurable parameters such as camera index, delays, and confidence thresholds are located in:
  
  gesture_control/config.py
