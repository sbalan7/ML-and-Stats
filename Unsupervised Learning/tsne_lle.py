import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_s_curve
from sklearn.manifold import LocallyLinearEmbedding, TSNE

n_points = 1000
X, color = make_s_curve(n_points, random_state=0)
n_neighbors = 10
n_components = 2

# Create figure
fig = plt.figure(figsize=(12, 4))

ax = fig.add_subplot(131, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.viridis)
ax.view_init(4, -72)
ax.set_title('Original Data')

lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components, method='standard')
Y1 = lle.fit_transform(X)
ax = fig.add_subplot(132)
ax.scatter(Y1[:, 0], Y1[:, 1], c=color, cmap=plt.cm.viridis, alpha=0.8)
ax.set_title('LLE')

tsne = TSNE(n_components=n_components, init='pca')
Y2 = tsne.fit_transform(X)
ax = fig.add_subplot(133)
ax.scatter(Y2[:, 0], Y2[:, 1], c=color, cmap=plt.cm.viridis, alpha=0.8)
ax.set_title('t-SNE')

plt.show()