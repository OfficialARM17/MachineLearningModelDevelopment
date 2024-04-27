
Sign Language Translation System


Machine Learning Model Development


This repository contains datasets and code needed to develop two machine learning models for a sign language translation system.

------------------------------------------------------------------------------------------------

Code

------------------------------------------------------------------------------------------------

hand_detection.py
This script contains code to develop a model that recognizes a hand from a camera view.
The output is a TensorFlow Lite Model suitable for integration into Android IDEs.
sign_language_recognition.py
This script contains code to develop a model that recognizes different sign language symbols from a hand.
The output is a TensorFlow Lite Model for use in Android IDEs.

------------------------------------------------------------------------------------------------

Dependencies


Python 3.5
TensorFlow
OpenCV
NumPy
Matplotlib
Keras
Scikit-learn
OS
Pandas

------------------------------------------------------------------------------------------------


Usage

Clone this repository to your local machine.
Install the required dependencies listed in requirements.txt.
Prepare your data or use the provided datasets.
Run hand_detection.py to train the hand detection model.
Run sign_language_recognition.py to train the sign language recognition model.
Convert the trained models to TensorFlow Lite Models using appropriate tools.
Integrate the TensorFlow Lite Models into Android IDEs for mobile applications.

------------------------------------------------------------------------------------------------


Acknowledgments

Datasets:

Here are some repositories that I found really helpful during development:

https://github.com/Mquinn960/sign-language

https://github.com/syauqy/handsign-tensorflow

https://github.com/pramod722445/hand_detection

