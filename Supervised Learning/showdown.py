import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import load_breast_cancer, load_wine, load_iris
from sklearn.datasets import load_boston, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor


def evaluate_c(clf):
    tic = time.time()
    s = metrics.accuracy_score(y_test, clf.predict(X_test))
    toc = time.time()
    t = toc - tic
    return s, t

def evaluate_r(clf):
    tic = time.time()
    s = metrics.r2_score(y_test, clf.predict(X_test))
    toc = time.time()
    t = toc - tic
    return s, t

def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.3f}'.format(height), xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

def classification_data():
    algos = []
    ttimes = []
    scores = []
    times = []

    tic = time.time()
    knn = KNeighborsClassifier().fit(X_train, y_train)
    toc = time.time()
    knn_t = toc-tic
    algos.append('KNN*')
    ttimes.append(knn_t)
    s, t = evaluate_c(knn)
    scores.append(s)
    times.append(t)

    tic = time.time()
    log = LogisticRegression().fit(X_train, y_train)
    toc = time.time()
    log_t = toc-tic
    algos.append('Log Reg*')
    ttimes.append(log_t)
    s, t = evaluate_c(log)
    scores.append(s)
    times.append(t)

    tic = time.time()
    gnb = GaussianNB().fit(X_train, y_train)
    toc = time.time()
    gnb_t = toc-tic
    algos.append('Gauss NB')
    ttimes.append(gnb_t)
    s, t = evaluate_c(gnb)
    scores.append(s)
    times.append(t)

    tic = time.time()
    srb = SVC(kernel='rbf').fit(X_train, y_train)
    toc = time.time()
    srb_t = toc-tic
    algos.append('RBF SVM')
    ttimes.append(srb_t)
    s, t = evaluate_c(srb)
    scores.append(s)
    times.append(t)

    tic = time.time()
    sp5 = SVC(kernel='poly', degree=5).fit(X_train, y_train)
    toc = time.time()
    sp5_t = toc-tic
    algos.append('Deg 5 SVM')
    ttimes.append(sp5_t)
    s, t = evaluate_c(sp5)
    scores.append(s)
    times.append(t)

    tic = time.time()
    dtr = DecisionTreeClassifier().fit(X_train, y_train)
    toc = time.time()
    dtr_t = toc-tic
    algos.append('Dec Tree*')
    ttimes.append(dtr_t)
    s, t = evaluate_c(dtr)
    scores.append(s)
    times.append(t)

    tic = time.time()
    rfc = RandomForestClassifier().fit(X_train, y_train)
    toc = time.time()
    rfc_t = toc-tic
    algos.append('Rand For*')
    ttimes.append(rfc_t)
    s, t = evaluate_c(rfc)
    scores.append(s)
    times.append(t)

    tic = time.time()
    abc = AdaBoostClassifier().fit(X_train, y_train)
    toc = time.time()
    abc_t = toc-tic
    algos.append('AdaBoost*')
    ttimes.append(abc_t)
    s, t = evaluate_c(abc)
    scores.append(s)
    times.append(t)

    tic = time.time()
    gbc = GradientBoostingClassifier().fit(X_train, y_train)
    toc = time.time()
    gbc_t = toc-tic
    algos.append('GradBoost*')
    ttimes.append(gbc_t)
    s, t = evaluate_c(gbc)
    scores.append(s)
    times.append(t)

    return algos, np.array(ttimes), np.array(times), np.array(scores)

def regression_data():
    algos = []
    ttimes = []
    scores = []
    times = []

    tic = time.time()
    lin = LinearRegression().fit(X_train, y_train)
    toc = time.time()
    lin_t = toc-tic
    algos.append('Lin Reg*')
    ttimes.append(lin_t)
    s, t = evaluate_r(lin)
    scores.append(s)
    times.append(t)

    tic = time.time()
    rid = Ridge().fit(X_train, y_train)
    toc = time.time()
    rid_t = toc-tic
    algos.append('Ridge*')
    ttimes.append(rid_t)
    s, t = evaluate_r(rid)
    scores.append(s)
    times.append(t)

    tic = time.time()
    las = Lasso().fit(X_train, y_train)
    toc = time.time()
    las_t = toc-tic
    algos.append('Lasso*')
    ttimes.append(las_t)
    s, t = evaluate_r(las)
    scores.append(s)
    times.append(t)

    tic = time.time()
    srb = SVR(kernel='rbf').fit(X_train, y_train)
    toc = time.time()
    srb_t = toc-tic
    algos.append('RBF SVR')
    ttimes.append(srb_t)
    s, t = evaluate_r(srb)
    scores.append(s)
    times.append(t)

    tic = time.time()
    sp5 = SVR(kernel='poly', degree=5).fit(X_train, y_train)
    toc = time.time()
    sp5_t = toc-tic
    algos.append('Deg 5 SVR')
    ttimes.append(sp5_t)
    s, t = evaluate_r(sp5)
    scores.append(s)
    times.append(t)

    tic = time.time()
    dtr = DecisionTreeRegressor().fit(X_train, y_train)
    toc = time.time()
    dtr_t = toc-tic
    algos.append('Dec Tree*')
    ttimes.append(dtr_t)
    s, t = evaluate_r(dtr)
    scores.append(s)
    times.append(t)

    tic = time.time()
    rfc = RandomForestRegressor().fit(X_train, y_train)
    toc = time.time()
    rfc_t = toc-tic
    algos.append('Rand For*')
    ttimes.append(rfc_t)
    s, t = evaluate_r(rfc)
    scores.append(s)
    times.append(t)

    tic = time.time()
    abc = AdaBoostRegressor().fit(X_train, y_train)
    toc = time.time()
    abc_t = toc-tic
    algos.append('AdaBoost*')
    ttimes.append(abc_t)
    s, t = evaluate_r(abc)
    scores.append(s)
    times.append(t)

    tic = time.time()
    gbc = GradientBoostingRegressor().fit(X_train, y_train)
    toc = time.time()
    gbc_t = toc-tic
    algos.append('GradBoost*')
    ttimes.append(gbc_t)
    s, t = evaluate_r(gbc)
    scores.append(s)
    times.append(t)

    return algos, np.array(ttimes), np.array(times), np.array(scores)

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
 
def plot_data(dataset, k):
    algos = []
    ottimes = np.zeros((9, ))
    otimes = np.zeros((9, ))
    oscores = np.zeros((9, ))

    if k=='c':
        for i in range(10):
            algos, ttimes, times, scores = classification_data()
            ottimes = ottimes + ttimes
            otimes = otimes + times
            oscores = oscores + scores
    if k=='r':
        for i in range(10):
            algos, ttimes, times, scores = regression_data()
            ottimes = ottimes + ttimes
            otimes = otimes + times
            oscores = oscores + scores

    ottimes /= 10
    otimes /= 10
    oscores /= 10

    plotting(algos, ottimes, otimes, oscores, dataset)


data = load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=27)
plot_data('Breast Cancer', 'c')

data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=82)
plot_data('Iris', 'c')

data = load_wine()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)
plot_data('Wine', 'c')

data = load_boston()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=47)
plot_data('Boston Housing Prices', 'r')
