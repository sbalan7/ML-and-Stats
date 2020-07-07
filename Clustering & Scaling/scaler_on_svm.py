import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y)

svm = SVC(C=100)
print('\n')
print(svm)
svm.fit(X_train, y_train)
print("Train set accuracy: {:.4f}".format(svm.score(X_train, y_train)))
print("Test set accuracy: {:.4f}".format(svm.score(X_test, y_test)))

scaler = MinMaxScaler()
X_train_ = scaler.fit_transform(X_train)
X_test_ = scaler.transform(X_test)
svm.fit(X_train_, y_train)
print("Scaled train set accuracy: {:.4f}".format(svm.score(X_train_, y_train)))
print("Scaled test set accuracy: {:.4f}".format(svm.score(X_test_, y_test)))
