from sklearn.decomposition import NMF, PCA
import matplotlib.pyplot as plt
import numpy as np

X = np.linspace(0, 30, 1000)
y1 = np.sin(X) + (np.random.randn(1000, )/3) + 6
y2 = [1 for x in range(100)]
y2 = np.concatenate((np.negative(y2), y2), axis=None)
y2 = np.tile(y2, 5) + (np.random.randn(1000, )/7) + 6

fig = plt.figure(figsize=(10, 8))

plt.subplot(411)
plt.plot(X, y1)
plt.plot(X, y2)

A = np.random.RandomState(0).uniform(size=(100, 3))
X = np.dot(np.array([X, y1, y2]).T, A.T)

nmf = NMF(n_components=2)
P = nmf.fit_transform(X)
plt.subplot(412)
plt.plot(P)


nmf = NMF(n_components=3)
P = nmf.fit_transform(X)
plt.subplot(413)
plt.plot(P)

nmf = NMF(n_components=4)
P = nmf.fit_transform(X)
plt.subplot(414)
plt.plot(P)

plt.tight_layout()
plt.show()