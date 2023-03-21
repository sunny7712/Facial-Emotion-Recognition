# Facial-Emotion-Recognition

Facial Emotion Recognition is a project that uses machine learning algorithms to detect emotions from facial expressions. This project aims to develop an application that can detect emotions from a real-time video stream.

Dataset
The dataset used in this project is the FER-2013 dataset. The dataset contains 48x48-pixel grayscale images of faces, labeled with one of seven emotions: anger, disgust, fear, happiness, sadness, surprise, or neutral. The dataset has a total of 35,887 images.

Dependencies
Python 3.7 or higher
OpenCV
NumPy
TensorFlow
Keras
You can install the required dependencies using the following command:

Copy code
pip install -r requirements.txt
Usage
You can use the emotion_recognition.py file to detect emotions from facial images or real-time video streams. To detect emotions from a facial image, run the following command:

css
Copy code
python emotion_recognition.py --image <path-to-image>
To detect emotions from a real-time video stream, run the following command:

Copy code
python emotion_recognition.py
Results
The model achieves an accuracy of around 65% on the validation set. The accuracy can be improved by using more sophisticated machine learning algorithms or by increasing the size of the dataset.

Future Work
Improving the accuracy of the model
Developing a user-friendly application for real-time emotion detection
Integrating the model with a robot or virtual assistant to recognize emotions and respond accordingly
Credits
The FER-2013 dataset was created by Pierre-Luc Carrier and Aaron Courville.
The model architecture was inspired by the paper "Real-time Convolutional Neural Networks for Emotion and Gender Classification" by Levi and Hassner.
