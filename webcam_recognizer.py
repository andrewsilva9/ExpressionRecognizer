import cv2
from classify_emotion import EmotionClassifier


class CameraRunner:
    def __init__(self):
        # Giant Dataset Emotions:
        # self.emotions = ["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        # Small Dataset Emotions:
        self.emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
        self.face_finder = cv2.CascadeClassifier("/Users/andrewsilva/opencv-3.1.0/data/haarcascades/haarcascade"
                                             "_frontalface_default.xml")
        self.face_finder2 = cv2.CascadeClassifier("/Users/andrewsilva/opencv-3.1.0/data/haarcascades/haarcascade"
                                              "_frontalface_alt2.xml")
        self.face_finder3 = cv2.CascadeClassifier("/Users/andrewsilva/opencv-3.1.0/data/haarcascades/haarcascade"
                                              "_frontalface_alt.xml")
        self.face_finder4 = cv2.CascadeClassifier("/Users/andrewsilva/opencv-3.1.0/data/haarcascades/haarcascade"
                                              "_frontalface_alt_tree.xml")
        self.classifier = EmotionClassifier()
        # width / height for fisher model: (350, 350). SVM: (48, 48)
        self.image_width = 350
        self.image_height = 350

    def run(self):
        capture = cv2.VideoCapture(0)
        while True:
            ret, frame = capture.read()
            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face1 = self.face_finder.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                                  flags=cv2.CASCADE_SCALE_IMAGE)
            face2 = self.face_finder2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5,5),
                                                   flags=cv2.CASCADE_SCALE_IMAGE)
            face3 = self.face_finder3.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                              flags=cv2.CASCADE_SCALE_IMAGE)
            face4 = self.face_finder4.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                              flags=cv2.CASCADE_SCALE_IMAGE)

            # Go over detected faces, stop at first detected face, return empty if no face.
            if len(face1) == 1:
                face = face1[0]
            elif len(face2) == 1:
                face = face2[0]
            elif len(face3) == 1:
                face = face3[0]
            elif len(face4) == 1:
                face = face4[0]
            else:
                continue
            # Cut and save face
            # scale image
            gray = gray[face[1]:face[1] + face[3],
                        face[0]:face[0] + face[2]]
            out = cv2.resize(gray, (self.image_height, self.image_width))  # Resize face so all images have same size
            print(self.emotions[self.classifier.classify_emotion(out)])
