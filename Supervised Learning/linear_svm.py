import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC, SVC


def DifferentKernels():
    data = load_iris()
    X, y = data.data, data.target
    X = X[:, :2]
    h = 0.02

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    fig = plt.figure(figsize=(10, 6))

    svm = SVC(kernel='linear').fit(X, y)
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.subplot(231)
    
    plt.title('Kernel = Linear')
    plt.contourf(xx, yy, Z, cmap=plt.cm.brg, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.brg, marker='.')
    
    plt.xticks([])
    plt.yticks([])

    svm = SVC(kernel='poly', degree=3).fit(X, y)
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.subplot(232)
    
    plt.title('Kernel = Poly, Degree = 3')
    plt.contourf(xx, yy, Z, cmap=plt.cm.brg, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.brg, marker='.')
    
    plt.xticks([])
    plt.yticks([])

    svm = SVC(kernel='poly', degree=7).fit(X, y)
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.subplot(233)
    
    plt.title('Kernel = Poly, Degree = 7')
    plt.contourf(xx, yy, Z, cmap=plt.cm.brg, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.brg, marker='.')
    
    plt.xticks([])
    plt.yticks([])

    svm = SVC(kernel='rbf', gamma=0.01).fit(X, y)
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.subplot(234)
    
    plt.title('Kernel = RBF, Gamma = 0.01')
    plt.contourf(xx, yy, Z, cmap=plt.cm.brg, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.brg, marker='.')
    
    plt.xticks([])
    plt.yticks([])

    svm = SVC(kernel='rbf', gamma=1).fit(X, y)
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.subplot(235)
    
    plt.title('Kernel = RBF, Gamma = 1')
    plt.contourf(xx, yy, Z, cmap=plt.cm.brg, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.brg, marker='.')
    
    plt.xticks([])
    plt.yticks([])

    svm = SVC(kernel='rbf', gamma=100).fit(X, y)
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.subplot(236)
    
    plt.title('Kernel = RBF, Gamma = 100')
    plt.contourf(xx, yy, Z, cmap=plt.cm.brg, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.brg, marker='.')
    
    plt.xticks([])
    plt.yticks([])

    plt.show()

def DifferentC():
    data = load_iris()
    X, y = data.data, data.target
    X = X[:, :2]
    h = 0.02

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    fig, ax = plt.subplots(1, 3, figsize=(10, 3))

    for c, axis in zip([0.001, 1, 1000], ax):
        svm = SVC(C=c).fit(X, y)
        axis.set_title('C = {}'.format(c))
        axis.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.brg, marker='.')
        
        Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        axis.contourf(xx, yy, Z, cmap=plt.cm.brg, alpha=0.4)
        axis.set_xticks([])
        axis.set_yticks([])

    plt.show()




DifferentC()
DifferentKernels()