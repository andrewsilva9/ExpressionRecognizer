# import os
import cv2
import numpy as np
from scipy import stats
from sklearn.externals import joblib
from sklearn.svm import SVC
# Define class that can be instantiated and have images passed to it
# images will be processed, and an emotion will be returned


class EmotionClassifier:
    # For the Fisher face model (okay performance):
    def __init__(self):
        # Emotion list
        # Giant Dataset Emotions:
        # self.emotions = ["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral"]  # Emotion list
        # Small Dataset Emotions:
        self.emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
        self.models = []
        for i in range(10):
            model_temp = cv2.face.createFisherFaceRecognizer()
            model_temp.load('fish_models/fish_model' + str(i) + '.xml')
            self.models.append(model_temp)

    def classify_emotion(self, input_image):
        emotion_guesses = np.zeros((len(self.models), 1))
        for index in range(len(self.models)):
            prediction, confidence = self.models[index].predict(input_image)
            emotion_guesses[index][0] = prediction
            # emotion_guesses[index][1] = confidence
        return int(stats.mode(emotion_guesses)[0][0])

    # For the SVM model (terrible performance):
    # def __init__(self):
    #     # Emotion list
    #     self.emotions = ["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral"]  # Emotion list
    #     self.model = joblib.load('face_svm.pkl')
    #
    #
    # def classify_emotion(self, input_image):
    #     input_image = np.reshape(input_image, (1, 2304))
    #     return self.model.predict(input_image)
