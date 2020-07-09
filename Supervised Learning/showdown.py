import numpy as np
import matplotlib.pyplot as plt
import time as t
from sklearn.datasets import load_breast_cancer, load_wine, load_iris
from sklearn.datasets import load_boston, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor


def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.3f}'.format(height), xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

def classification_data(X_train, X_test, y_train, y_test, dataset):
    algos = ['KNN*', 'Log Reg*', 'Gauss NB', 'RBF SVM', 'Poly5 SVM', 'Dec Tree*', 'Rand For*', 'AdaBoost*', 'GradBoost*']
    models = [KNeighborsClassifier(), LogisticRegression(), GaussianNB(), SVC(kernel='rbf'), SVC(kernel='poly', degree=5),
              DecisionTreeClassifier(), RandomForestClassifier(), AdaBoostClassifier(), GradientBoostingClassifier()]
    scores = []
    times = []
    ttimes = []

    for algo, model in zip(algos, models):
        score = 0
        time = 0
        ttime = 0

        for i in range(10):
            tic = t.time()
            model = model.fit(X_train, y_train)
            toc = t.time()
            time += (toc-tic)
            tic = t.time()
            score += accuracy_score(y_test, model.predict(X_test))
            toc = t.time()
            ttime += (toc-tic)
        
        scores.append(score/10)
        times.append(time/10)
        ttimes.append(ttime/10)

    plotting(algos, times, ttimes, scores, dataset)

def regression_data(X_train, X_test, y_train, y_test, dataset):
    algos = ['Lin Reg*', 'Ridge*', 'Lasso*', 'RBF SVR', 'Poly5 SVR', 'Dcc Tree*', 'Rand For*', 'AdaBoost*', 'GradBoost*']
    models = [LinearRegression(), Ridge(), Lasso(), SVR(kernel='rbf'), SVR(kernel='poly', degree=5),
              DecisionTreeRegressor(), RandomForestRegressor(), AdaBoostRegressor(), GradientBoostingRegressor()]
    scores = []
    times = []
    ttimes = []

    for algo, model in zip(algos, models):
        score = 0
        time = 0
        ttime = 0

        for i in range(10):
            tic = t.time()
            model = model.fit(X_train, y_train)
            toc = t.time()
            time += (toc-tic)
            tic = t.time()
            score += r2_score(y_test, model.predict(X_test))
            toc = t.time()
            ttime += (toc-tic)
        
        scores.append(score/10)
        times.append(time/10)
        ttimes.append(ttime/10)

    plotting(algos, times, ttimes, scores, dataset)

def plotting(algos, ttimes, times, scores, dataset):
    x = np.arange(len(algos)) * 2
    x_ = x + 1
    width = 1.0

    fig1, ax = plt.subplots()
    tr = ax.bar(x, ttimes, width=width, label='Training Times')
    pr = ax.bar(x_, times, width=width, label='Prediction Times')
    autolabel(tr, ax)
    autolabel(pr, ax)
    ax.set_ylabel('Time taken')
    ax.set_title('Time Comparison of Algorithms on the {} dataset'.format(dataset))
    ax.set_xticks((x+x_)/2)
    ax.set_xticklabels(algos)
    ax.legend()
    ax.grid(True)
    plt.show()

    fig2, ax = plt.subplots()
    tr = ax.bar(x, ttimes, width=width, label='Training Times')
    pr = ax.bar(x_, times, width=width, label='Prediction Times')
    autolabel(tr, ax)
    autolabel(pr, ax)
    ax.set_ylabel('Time taken')
    ax.set_title('Time Comparison of Algorithms on the {} dataset'.format(dataset))
    ax.set_xticks((x+x_)/2)
    ax.set_xticklabels(algos)
    ax.legend()
    ax.grid(True)
    ax.set_yscale('log')
    plt.show()

    fig3, ax = plt.subplots()
    z = ax.bar(algos, scores)
    autolabel(z, ax)
    ax.set_ylim(min(scores)-0.15, 1)
    plt.xticks(rotation=30)
    plt.ylabel('Score on Test Set')
    plt.title('Accuracy Comparison on {} Dataset'.format(dataset))
    plt.show()

def classifcation_datasets():
    names = ['Breast Cancer', 'Iris', 'Wine']
    datasets = [load_breast_cancer(), load_iris(), load_wine()]

    for dataset, name in zip(datasets, names):
        data = dataset
        X = data.data
        y = data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        classification_data(X_train, X_test, y_train, y_test, name)

classification_datasets()

data = load_boston()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=47)
regression_data(X_train, X_test, y_train, y_test, 'Boston Housing Prices')
