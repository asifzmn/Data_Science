import pandas as pd
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as prfs, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':
    dataset = pd.read_csv('/home/az/Desktop/archive/Iris.csv')
    X, y = dataset.drop(['Id', 'Species'], axis=1), dataset['Species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    classifier = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=3)).fit(X_train, y_train)
    y_pred, y_score = classifier.predict(X_test), classifier.predict_proba(X_test)

    accuracy,(precision, recall, fscore, support) = accuracy_score(y_test, y_pred),prfs(y_test, y_pred, labels=y.unique())

    skplt.metrics.plot_roc(y_test, y_score)
    plt.show()