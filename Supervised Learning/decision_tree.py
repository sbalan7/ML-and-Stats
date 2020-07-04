import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

data = load_iris()
X = data.data
y = data.target
X = X[:, :2]
h = 0.02

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

fig1 = plt.figure(figsize=(10, 3))

dt1 = DecisionTreeClassifier().fit(X, y)
dt2 = DecisionTreeClassifier(max_depth=3).fit(X, y)
dt3 = DecisionTreeClassifier(criterion='entropy').fit(X, y)

plt.subplot(131)

Z = dt1.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.title('Defaults')
plt.contourf(xx, yy, Z, cmap=plt.cm.brg, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.brg, marker='.')
plt.xticks([])
plt.yticks([])

plt.subplot(132)

Z = dt2.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.title('Max Depth = 3')
plt.contourf(xx, yy, Z, cmap=plt.cm.brg, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.brg, marker='.')
plt.xticks([])
plt.yticks([])

plt.subplot(133)

Z = dt3.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.title('Entropy Split')
plt.contourf(xx, yy, Z, cmap=plt.cm.brg, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.brg, marker='.')
plt.xticks([])
plt.yticks([])

plt.show()

fig2 = plt.figure(figsize=(50, 24))
plot_tree(dt1, filled=True, fontsize=6)
plt.savefig('tree_hi_res.png', dpi=150)

fig3 = plt.figure(figsize=(14, 10))
plot_tree(dt2, filled=True, fontsize=10)
plt.savefig('tree_small.png', dpi=100)