import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp
from sklearn.datasets import make_blobs

def centeralise_axes(ax):
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

def plot_random_clusters(points=30):

    X1, _ = make_blobs(n_samples=points, centers=3, cluster_std=2)
    X2, _ = make_blobs(n_samples=points, centers=3, cluster_std=2)

    fig, ax = plt.subplots()
    ax.scatter(X1[:, 0], X1[:, 1], c='red', marker='.')
    ax.scatter(X2[:, 0], X2[:, 1], c='blue', marker='.')
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.grid(True)
    centeralise_axes(ax)
    plt.show()
    return X1, X2

def plot_scalers(X1, X2):

    scalers = [pp.StandardScaler(), pp.MinMaxScaler(), pp.Normalizer(), pp.RobustScaler()]
    scaler_names = ['Standard', 'MinMax', 'Normalizer', 'Robust']
    no_of_scalers = len(scalers)

    i = 1
    fig = plt.figure(figsize=(6, 7))

    for scaler, scaler_name in zip(scalers, scaler_names):
        
        X1 = scaler.fit_transform(X1)
        X2 = scaler.fit_transform(X2)
        
        ax = plt.subplot(2, 2, i)
        ax.scatter(X1[:, 0], X1[:, 1], c='red', marker='.')
        ax.scatter(X2[:, 0], X2[:, 1], c='blue', marker='.')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.grid(True)
        ax.set_title(scaler_name)
        centeralise_axes(ax)
        i += 1

    plt.show()


X1, X2 = plot_random_clusters(50)
plot_scalers(X1, X2)

