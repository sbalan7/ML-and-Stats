import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB


model_names = ['MLP', 'kNN (k=5)', 'SVM RBF', 'SVM P d=5', 'SVM P d=11', 'Dec Tree', 'Rand Forest', 'AdaBoost', 'GradBoost', 'Gaus NB']
models = [
    MLPClassifier(alpha=1, max_iter=1000),
    KNeighborsClassifier(n_neighbors=5),
    SVC(kernel='rbf', gamma=1),
    SVC(kernel='poly', degree=5),
    SVC(kernel='poly', degree=11),
    DecisionTreeClassifier(max_depth=7),
    RandomForestClassifier(max_depth=3, n_estimators=10),
    AdaBoostClassifier(learning_rate=0.2),
    GradientBoostingClassifier(learning_rate=0.3),
    GaussianNB()
]
no_of_models = len(models)

X1, y1 = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1)
X1 = StandardScaler().fit_transform(X1)
X2, y2 = make_circles(n_samples=200, noise=0.2, factor=0.3)
X2 = StandardScaler().fit_transform(X2)
X3, y3 = make_moons(n_samples=200, noise=0.4)
X3 = StandardScaler().fit_transform(X3)

datasets = [(X1, y1), (X2, y2), (X3, y3)]
no_of_datasets = len(datasets)

h = 0.02
fig = plt.figure(figsize=(no_of_models*2, (no_of_datasets-1)*2))
i = 1

for c, dataset in enumerate(datasets):

    X, y = dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    for model_name, model in zip(model_names, models):

        model = model.fit(X_train, y_train)
        score = model.score(X_test, y_test)

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax = plt.subplot(no_of_datasets, no_of_models, i)
        ax.contourf(xx, yy, Z, cmap=plt.cm.winter, alpha=.4)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.winter, marker='.')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score), size=5, horizontalalignment='right')
        if c == 0:
            ax.set_title(model_name)
        i += 1

fig.savefig('decisions.png', dpi=150)
plt.show()