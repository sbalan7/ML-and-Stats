import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

n_neighbors = [1, 3, 7, 13, 21, 51]

iris = load_iris()

X = iris.data[:, :2]
y = iris.target
h = .02

cmap_light = ListedColormap(['#FF8888', '#88FF88','#8888FF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00','#0000FF'])

Fig = plt.subplots(figsize=(12, 7))
for i in range(len(n_neighbors)):
    knn = KNeighborsClassifier(n_neighbors[i], weights='distance')
    knn.fit(X, y)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    fig = plt.subplot(2, 3, i+1)
    fig.pcolormesh(xx, yy, Z, cmap=cmap_light)

    fig.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, marker='.')
    fig.set_xlim(xx.min(), xx.max())
    fig.set_ylim(yy.min(), yy.max())
    fig.set_title("k = {}".format(n_neighbors[i]))
    fig.set_xticks([])
    fig.set_yticks([])

plt.show()
