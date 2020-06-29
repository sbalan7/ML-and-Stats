import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

def three_class():
    iris = load_iris()
    r = np.random.randint(0, 100)

    X_train, X_test, y_train, y_test =  train_test_split(iris.data, iris.target, random_state=r)

    svm = SVC(kernel='poly', C=0.01)
    svm.fit(X_train, y_train)

    plot_confusion_matrix(svm, X_test, y_test, display_labels=iris.target_names, cmap=plt.cm.Blues)
    plt.show()

def two_class():
    bc = load_breast_cancer()
    r = np.random.randint(0, 100)

    X_train, X_test, y_train, y_test =  train_test_split(bc.data, bc.target, random_state=r)

    svm = SVC(kernel='linear', C=0.1)
    svm.fit(X_train, y_train)

    plot_confusion_matrix(svm, X_test, y_test, display_labels=bc.target_names, cmap=plt.cm.Blues)
    plt.show()

#three_class()
two_class()
