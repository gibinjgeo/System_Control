Hand Gesture Recognition with Mediapipe and PyAutoGUI

This Python project allows users to control their computer using hand gestures. It leverages the Mediapipe library to detect hand gestures, and PyAutoGUI to map these gestures to various mouse and keyboard actions. Additionally, the project includes a sign language detection system using Keras-based pre-trained models to detect gestures for alphabets.
Project Overview

The project has four main features:

    Mouse Control: Controls the mouse pointer and buttons based on finger positions.
    Hand Gesture Control: Detects hand gestures and performs keyboard actions like switching tabs, pressing keys, and navigating.
    ASL (American Sign Language) Recognition: Recognizes sign language gestures from webcam input and maps them to keyboard inputs (e.g., typing letters or performing actions).
    ISL (Indian Sign Language) and CSL (Custom Sign Language) Recognition: Similar to ASL recognition but for ISL and CSL gestures.

Installation
Prerequisites

Make sure you have the following dependencies installed:

    Python 3.7+
    OpenCV
    Mediapipe
    PyAutoGUI
    Keras
    TensorFlow
    NumPy
    SciPy
    Matplotlib

You can install these packages using the following commands:

bash

pip install opencv-python-headless
pip install mediapipe
pip install pyautogui
pip install numpy
pip install tensorflow
pip install keras

Setting Up the Pre-Trained Models

    Place the pre-trained Keras models (keras_model.h5, keras_model1.h5, keras_model2.h5) in the same directory as the Python script.
    Place the associated label files (labels.txt, labels1.txt, labels2.txt) in the same directory.

Additional Files

Ensure the following files are in the same directory as the script:

    combined_text.txt: This file contains the input text for hand gesture detection.
    labels.txt, labels1.txt, labels2.txt: Label files for the models.
    keras_model.h5, keras_model1.h5, keras_model2.h5: Pre-trained models for hand gesture classification.

Usage

To run the project, simply execute the Python script:

bash

python your_script_name.py

The system will start capturing video input from the webcam and begin detecting gestures.
Mouse Control

    Use your hand gestures to move the mouse.
    Specific finger positions will trigger mouse clicks:
        Index Finger Up: Move the mouse pointer.
        Thumb + Index: Left-click.
        Thumb + Middle Finger: Right-click.

Hand Gesture Control

    Various hand gestures perform specific keyboard actions:
        Three fingers up: Presses the f11 key (fullscreen).
        Four fingers up: Presses win+tab (switching windows).
        Five fingers up: Presses alt+tab (application switcher).

ASL, ISL, CSL Recognition

    After switching to ASL, ISL, or CSL mode, the system will recognize hand gestures and map them to respective letters or commands.
    These gestures are processed using the Mediapipe framework and Keras models.

Exiting the Program

To exit the program, simply press the ESC key or perform the specific gesture to raise a SystemExit.
How It Works

    Mediapipe Hands: The Mediapipe library is used to detect hand landmarks. It tracks the position of each finger and hand landmark in real-time from webcam input.
    Gesture Detection: Based on the position of the fingers, a gesture is identified, which is then mapped to an action such as moving the mouse or pressing a key.
    Keras Model for Sign Language: The pre-trained models are used to classify hand gestures for sign language recognition (ASL, ISL, CSL). The recognized gestures are mapped to specific alphabets or commands.
    PyAutoGUI: Used to control the mouse and keyboard based on the recognized gestures.

Customization

You can modify the program to perform different actions for each gesture by adjusting the keybord and keybordl functions.

For example, you can change the action for gesture ctt == 1 from pressing ctrl+c to another combination by editing the code:

python

elif ctt == 1:
    pyautogui.hotkey('ctrl', 'c')  # You can change this to any other action
    print("index finger only-one")

Future Improvements

    Multi-hand gesture support.
    Integrating voice commands.
    Improving the accuracy of gesture recognition by training with a larger dataset.

Contributing

Contributions are welcome. Feel free to submit issues or pull requests.
License

This project is licensed under the MIT License.
Acknowledgments

    Mediapipe by Google for hand tracking.
    PyAutoGUI for easy automation of mouse and keyboard actions.
    Keras for pre-trained models used for hand gesture classification.