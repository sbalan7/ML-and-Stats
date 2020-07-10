from sklearn.datasets import make_circles, make_blobs, make_moons
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import numpy as np
import matplotlib.pyplot as plt

n_samples = 1000
rings = make_circles(n_samples=n_samples, factor=.5, noise=.05)
moons = make_moons(n_samples=n_samples, noise=.05)
blobs = make_blobs(n_samples=n_samples, centers=2)
blubs = make_blobs(n_samples=n_samples, centers=3)
datasets = [rings, moons, blobs, blubs]

km = KMeans(n_clusters=2)
db = DBSCAN(eps=0.18)
sc = SpectralClustering(n_clusters=2, affinity="nearest_neighbors")
gm = GaussianMixture(n_components=2, covariance_type='full')
models = [('K means', km), ('Spectral Clustering', sc), ('DBSCAN', db), ('Gaussian Mixture', gm)]



fig = plt.figure(figsize=(6, 6))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.1, hspace=.1)
i = 1
for dataset in datasets:
    X, _ = dataset
    X = StandardScaler().fit_transform(X)
    for name, model in models:

        plt.subplot(len(datasets)+1, len(models), i)
        model.fit(X)
        if hasattr(model, 'labels_'):
            pred = model.labels_.astype(np.int)
        else:
            pred = model.predict(X)

        colors = np.array(list(islice(cycle(['#ee5555', '#5555ee', '#55ee55']), int(max(pred) + 1))))
        colors = np.append(colors, ["#000000"])
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[pred])

        plt.xticks(())
        plt.yticks(())
        if i // (len(models)+1) == 0:
            plt.title(name)
        i += 1


km = KMeans(n_clusters=3)
db = DBSCAN(eps=0.2)
sc = SpectralClustering(n_clusters=3, affinity="nearest_neighbors")
gm = GaussianMixture(n_components=3, covariance_type='full')
models = [('K means', km), ('DBSCAN', db), ('Spectral Clustering', sc), ('Gaussian Mixture', gm)]
X, _ = blubs
X = StandardScaler().fit_transform(X)
for name, model in models:

    plt.subplot(len(datasets)+1, len(models), i)
    model.fit(X)
    if hasattr(model, 'labels_'):
        pred = model.labels_.astype(np.int)
    else:
        pred = model.predict(X)

    colors = np.array(list(islice(cycle(['#ee5555', '#5555ee', '#55ee55']), int(max(pred) + 1))))
    colors = np.append(colors, ["#000000"])
    plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[pred])

    plt.xticks(())
    plt.yticks(())
    if i // (len(models)+1) == 0:
        plt.title(name)
    i += 1
plt.show()


