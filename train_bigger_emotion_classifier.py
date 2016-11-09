# import cv2
import pandas
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import numpy as np
emotions = ["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral"]  # Emotion list

data = pandas.read_csv('fer2013.csv')
labels = np.array(data['emotion'])
pixels = np.array(data['pixels'])
image_data = []
for i in range(len(pixels)):
    img_data = pixels[i].split()
    img_data = np.array(img_data).astype(int)
    image_data.append(img_data)
image_data = np.array(image_data)
# image_data = image_data.reshape((len(image_data), 48, 48))


def get_files(emotion):  # Define function to get file list, randomly shuffle it and split 80/20
    training_full = image_data[labels == emotion]
    np.random.shuffle(training_full)
    training = training_full[:int(len(training_full) * 0.8)]  # get first 80% of file list
    prediction = training_full[-int(len(training_full) * 0.2):]  # get last 20% of file list
    return training, prediction


def make_sets():
    training_set_size = 0.8
    test_set_size = 1-training_set_size
    rng_state = np.random.get_state()
    np.random.shuffle(labels)
    np.random.set_state(rng_state)
    np.random.shuffle(image_data)
    training_data = image_data[:int(len(image_data)*training_set_size)]
    training_labels = labels[:int(len(labels)*training_set_size)]
    prediction_data = image_data[-int(len(image_data)*test_set_size):]
    prediction_labels = labels[-int(len(labels)*test_set_size):]
    return training_data, training_labels, prediction_data, prediction_labels


def run_recognizer():
    training_data, training_labels, prediction_data, prediction_labels = make_sets()

    print("training SVM classifier")
    print("size of training set is:", len(training_labels), "images")
    clf = LinearSVC()

    clf.fit(np.asarray(training_data), np.asarray(training_labels))
    joblib.dump(clf, 'linear_face_svm.pkl')
    print("predicting classification set")
    predictions = clf.predict(np.asarray(prediction_data))
    correct = predictions - np.asarray(prediction_labels)
    correct = correct[correct == 0]
    # print("Percent correct: %f" % (len(correct)/len(prediction_labels)))
    return len(correct)/len(prediction_labels)

#
percent = run_recognizer()
print("Percent correct: %f" % percent)
