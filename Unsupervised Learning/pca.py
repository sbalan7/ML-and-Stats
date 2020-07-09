from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(palette='Set1')

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
