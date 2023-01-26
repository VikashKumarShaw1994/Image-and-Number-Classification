import util
from math import sqrt
import numpy as np
import collections
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3)
Test=None


class KnnClassifier:
    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "knn"
        self.num_neighbors =1

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        # initialize the data
        P_Y_Count = collections.Counter(trainingLabels)
        global model
        global Test
        if len(P_Y_Count.keys()) == 2:
            Test = "Face"
            if len(trainingLabels) < 20:
                model = KNeighborsClassifier(n_neighbors=19)
            else:
                model = KNeighborsClassifier(n_neighbors=29)

        else:
            Test = "Digit"
            model = KNeighborsClassifier(n_neighbors=1)

        trainingLabels = trainingLabels + validationLabels
        trainingData = trainingData + validationData
        self.trainingLabels = trainingLabels
        self.trainingData = trainingData

        # make features of training data
        self.size = len(list(trainingData))
        features = [];
        for datum in trainingData:
            feature = list(datum.values())
            features.append(feature)

        # combine features and labels of training data as train_set
        train_set = [];
        for i in range(self.size):
            train_datum = list(np.append(features[i], self.trainingLabels[i]))
            train_set.append(train_datum)
        self.train_set = train_set




        # Train the model using the training sets
        model.fit(features, trainingLabels)
        # Predict Output


    def classify(self, testData):
        # make features of testing data
        self.size = len(list(testData))
        features = [];
        guesses=[]
        for datum in testData:
            feature = list(datum.values())
            features.append(feature)
            predicted = model.predict([feature])
            guesses.append(predicted)
        return guesses
'''
        # combine features and labels of testing data as test_set
        test_set = [];
        for i in range(self.size):
            train_datum = list(np.append(features[i], None))
            test_set.append(train_datum)
        self.test_set = test_set

        # predict the class of all testing data
        guesses = []
        for test_datum in test_set:
            train_set = self.train_set
            num_neighbors = self.num_neighbors
            # call predict_classification function to predict the class of one test data

            guess = predict_classification(train_set, test_datum, num_neighbors)
            # save the data in guesses
            guesses.append(guess)'''

