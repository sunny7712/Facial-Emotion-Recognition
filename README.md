# Facial-Emotion-Recognition

Facial Emotion Recognition is a project that uses machine learning algorithms to detect emotions from facial expressions. This project aims to develop an application that can detect emotions from a real-time video stream. It uses Deep Learning Technique like the Covolutional Neural Networks to train the model.

# Dataset
The dataset used in this project is the Emotion Detection dataset. The dataset contain 35,685 examples of 48x48 pixel gray scale images of faces divided into train and test dataset. Images are categorized based on the emotion shown in the facial expressions (happiness, neutral, sadness, anger, surprise, disgust, fear).

# Dependencies
- Python 3.7 or higher
- OpenCV
- NumPy
- TensorFlow
- Keras

You can install the required dependencies using the following command:

`pip install -r requirements.txt`

# Usage
You can use the **main.py** file to detect emotions from real-time video streams. To detect emotions from a real-time video stream, run the following command:

`python main.py`

# Results
The model achieves an accuracy of around 65% on the validation set. For reference, Human performance on this dataset is estimated to be 65.5%.

The accuracy can be improved by using more sophisticated architecture or by fine tuning the hyperparameters or by increasing the size of the dataset. 

I have used the Haar Cascade model in OpenCV library for face detection.

# Future Work
- Improving the accuracy of the model

- Developing a user-friendly application for real-time emotion detection

- Integrating the model with a robot or virtual assistant to recognize emotions and respond accordingly

# Credits
The Emotion Recognition dataset was inspired from the FER-2013 dataset.

The model architecture was inspired from a Kaggle User AAYUSH MISHRA.
