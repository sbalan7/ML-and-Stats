from sklearn.datasets import load_iris, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(palette='Set1')

def  normalPCA():
    data = load_iris()
    X = data.data
    y = data.target
    iris = pd.DataFrame(X)
    iris['target'] = y
    iris.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'target']


    pca = PCA(n_components=2).fit(X)
    X_principal = pca.transform(X)


    fig1 = plt.figure()
    plt.scatter(X_principal[0:50, 0], X_principal[0:50, 1], c=[plt.cm.Set1(0)], marker='.')
    plt.scatter(X_principal[50:100, 0], X_principal[50:100, 1], c=[plt.cm.Set1(1)], marker='.')
    plt.scatter(X_principal[100:150, 0], X_principal[100:150, 1], c=[plt.cm.Set1(2)], marker='.')
    plt.xlabel('Principal Axis 1')
    plt.ylabel('Principal Axis 2')
    plt.show()

    pp = sns.pairplot(iris, hue='target', height=1.5)
    plt.show()

    print('PCA components : {}'.format(pca.components_))
    plt.matshow(pca.components_, cmap='viridis')
    plt.yticks([0, 1], ['Axis 1', 'Axis 2'])
    plt.colorbar()
    plt.xticks(range(len(data.feature_names)), data.feature_names, rotation=60, ha='left')
    plt.xlabel("Feature")
    plt.ylabel("Principal components")
    plt.show()

def kernelPCA():
    X, y = make_circles(n_samples=500, factor=0.3, noise=0.1)
    r = y == 0
    b = y == 1
    kp = KernelPCA(kernel='rbf', gamma=10)
    pc = PCA(n_components=2)
    X_k = kp.fit_transform(X)
    X_p = pc.fit_transform(X)

    fig = plt.figure(figsize=(9, 3))
    plt.subplot(131)
    plt.scatter(X[r, 0], X[r, 1], c="red", alpha=0.8)
    plt.scatter(X[b, 0], X[b, 1], c="blue", alpha=0.8)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Original Space')
    
    plt.subplot(132)
    plt.scatter(X_p[r, 0], X_p[r, 1], c="red", alpha=0.8)
    plt.scatter(X_p[b, 0], X_p[b, 1], c="blue", alpha=0.8)
    plt.title('Normal PCA')
    plt.xlabel('1st component')
    plt.ylabel('2nd component')

    plt.subplot(1, 3, 3)
    plt.scatter(X_k[r, 0], X_k[r, 1], c="red", alpha=0.8)
    plt.scatter(X_k[b, 0], X_k[b, 1], c="blue", alpha=0.8)
    plt.title('RBF PCA')
    plt.xlabel('1st component')
    plt.ylabel('2nd component')
    
    plt.show()

kernelPCA()
    