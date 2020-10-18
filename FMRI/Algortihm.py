import collections
import warnings

import numpy as np
import sklearn.exceptions
from sklearn import preprocessing
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


def Classification(infoX, dataY):
    # print(np.shape(dataY))

    l = int((len(infoX) * 5) / 6)
    X = dataY[:l]
    y = list(map(int, infoX[:l, 1]))

    models = [
        MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 3), random_state=1),
        svm.LinearSVC(random_state=0, tol=1e-5),
        KNeighborsClassifier(n_neighbors=3),
        GaussianNB(),
        svm.SVC(gamma='scale', decision_function_shape='ovo')
    ]

    # models = [svm.LinearSVC(random_state=1, tol=1e-5)]
    # model = svm.SVR()
    # model = svm.SVC(gamma='scale', decision_function_shape='ovo')
    # model = svm.SVC(kernel='rbf')
    # model = tree.DecisionTreeClassifier()
    # model = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5,2), random_state=1)
    # model = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
    # model = KNeighborsClassifier(n_neighbors=3)
    # model = RandomForestClassifier(n_estimators=10)
    # model = GaussianNB()

    result = []
    for model in models:
        model.fit(X, y)
        trueReal, p = Accuracy(infoX, dataY, model, l)
        result.append((model, (ModelValidation(trueReal, p))))
    return result


def Accuracy(infoX, dataY, model, l):
    correct = 0
    incorrect = 0

    yTrue = []
    yPredict = []

    for i in range(l, len(dataY)):

        prediction = model.predict([dataY[i]])[0]
        expected = int(infoX[i][1])

        yTrue.append(expected)
        yPredict.append(prediction)

        if prediction == expected:
            correct += 1
        else:
            incorrect += 1

    accuracy = ((correct * 100) / (correct + incorrect))

    return yTrue, yPredict


def Clustering(X):
    for i in range(2, 22):
        y_pred = KMeans(n_clusters=i, random_state=170).fit_predict(X)
        counter = collections.Counter(y_pred)
        print(counter)


def ModelValidation(y_true, y_pred):
    warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
    scores = []
    scores.append(("Accuracy Score ", accuracy_score(y_true, y_pred)))
    scores.append(("Balanced Accuracy Score ", balanced_accuracy_score(y_true, y_pred)))
    scores.append(("F1 Score ", f1_score(y_true, y_pred, average='weighted', labels=np.unique(y_pred))))
    scores.append(("Precision Score ", precision_score(y_true, y_pred, average='weighted')))
    scores.append(("Recall Score ", recall_score(y_true, y_pred, average='weighted')))
    return scores


def TrainingAndTesting(subjects):
    for subject in subjects:

        info = subject.info
        data = subject.dataset

        binarizer = preprocessing.Binarizer().fit(data)
        data = binarizer.transform(data)

        # inf = []
        # dat = []

        # for i in range(meta.ntrials):
        #     # if  info[i][0]=='animal' or info[i][0]=='manmade' or info[i][0]=='tool':
        #     if  info[i][0]=='animal' or info[i][0]=='vehicle':
        #     # if  info[i][2]=='cat' or info[i][2]=='bicycle':
        #     # if  int(info[i][1])<=7:
        #     # if True:
        #         inf.append(info[i])
        #         # dat.append(data[i,part])
        #         dat.append(data[i])
        # inf = np.array(inf)

        # Clustering(meta.colToCoord)

        # sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
        # data = sel.fit_transform(data)
        # dat = sel.fit_transform(dat)
        # print(len(inf),len(data[0]))
        # Classification(inf,dat)

        results = Classification(info, data)
        print("Person ", subjects.index(subject) + 1)
        print()
        for result in results:
            # print(result[0])
            print(type(result[0]).__name__)
            for score in result[1]:
                print(score[0], score[1])
            print()
        print()
        print()
        print()
