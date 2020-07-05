import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import * 
from sklearn.datasets import load_breast_cancer, load_wine, load_iris
from sklearn.model_selection import train_test_split


def evaluate(clf):
    tic = time.time()
    s = metrics.accuracy_score(y_test, clf.predict(X_test))
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
    knn = neighbors.KNeighborsClassifier().fit(X_train, y_train)
    toc = time.time()
    knn_t = toc-tic
    algos.append('KNN*')
    ttimes.append(knn_t)
    s, t = evaluate(knn)
    scores.append(s)
    times.append(t)

    tic = time.time()
    log = linear_model.LogisticRegression().fit(X_train, y_train)
    toc = time.time()
    log_t = toc-tic
    algos.append('Log Reg*')
    ttimes.append(log_t)
    s, t = evaluate(log)
    scores.append(s)
    times.append(t)

    tic = time.time()
    gnb = naive_bayes.GaussianNB().fit(X_train, y_train)
    toc = time.time()
    gnb_t = toc-tic
    algos.append('Gauss NB')
    ttimes.append(gnb_t)
    s, t = evaluate(gnb)
    scores.append(s)
    times.append(t)

    tic = time.time()
    srb = svm.SVC(kernel='rbf').fit(X_train, y_train)
    toc = time.time()
    srb_t = toc-tic
    algos.append('RBF SVM')
    ttimes.append(srb_t)
    s, t = evaluate(srb)
    scores.append(s)
    times.append(t)

    tic = time.time()
    sp5 = svm.SVC(kernel='poly', degree=5).fit(X_train, y_train)
    toc = time.time()
    sp5_t = toc-tic
    algos.append('Deg 5 SVM')
    ttimes.append(sp5_t)
    s, t = evaluate(sp5)
    scores.append(s)
    times.append(t)

    tic = time.time()
    dtr = tree.DecisionTreeClassifier().fit(X_train, y_train)
    toc = time.time()
    dtr_t = toc-tic
    algos.append('Dec Tree*')
    ttimes.append(dtr_t)
    s, t = evaluate(dtr)
    scores.append(s)
    times.append(t)

    tic = time.time()
    rfc = ensemble.RandomForestClassifier().fit(X_train, y_train)
    toc = time.time()
    rfc_t = toc-tic
    algos.append('Rand For*')
    ttimes.append(rfc_t)
    s, t = evaluate(rfc)
    scores.append(s)
    times.append(t)

    tic = time.time()
    abc = ensemble.AdaBoostClassifier().fit(X_train, y_train)
    toc = time.time()
    abc_t = toc-tic
    algos.append('AdaBoost*')
    ttimes.append(abc_t)
    s, t = evaluate(abc)
    scores.append(s)
    times.append(t)

    tic = time.time()
    gbc = ensemble.GradientBoostingClassifier().fit(X_train, y_train)
    toc = time.time()
    gbc_t = toc-tic
    algos.append('GradBoost*')
    ttimes.append(gbc_t)
    s, t = evaluate(gbc)
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
 

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=27)

algos = []
ottimes = np.zeros((9, ))
otimes = np.zeros((9, ))
oscores = np.zeros((9, ))

for i in range(10):
    algos, ttimes, times, scores = classification_data()
    ottimes = ottimes + ttimes
    otimes = otimes + times
    oscores = oscores + scores

ottimes /= 10
otimes /= 10
oscores /= 10

plotting(algos, ottimes, otimes, oscores, 'Breast Cancer')

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=82)

algos = []
ottimes = np.zeros((9, ))
otimes = np.zeros((9, ))
oscores = np.zeros((9, ))

for i in range(10):
    algos, ttimes, times, scores = classification_data()
    ottimes = ottimes + ttimes
    otimes = otimes + times
    oscores = oscores + scores

ottimes /= 10
otimes /= 10
oscores /= 10

plotting(algos, ottimes, otimes, oscores, 'Iris')

data = load_wine()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)

algos = []
ottimes = np.zeros((9, ))
otimes = np.zeros((9, ))
oscores = np.zeros((9, ))

for i in range(10):
    algos, ttimes, times, scores = classification_data()
    ottimes = ottimes + ttimes
    otimes = otimes + times
    oscores = oscores + scores

ottimes /= 10
otimes /= 10
oscores /= 10

plotting(algos, ottimes, otimes, oscores, 'Wine')