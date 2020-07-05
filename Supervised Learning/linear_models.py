import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso

def ridge_lasso():
    boston = load_boston()
    X = boston.data
    y = boston.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    lir = LinearRegression().fit(X_train, y_train)

    rid1 = Ridge(alpha=0.01).fit(X_train, y_train)
    rid2 = Ridge(alpha=1).fit(X_train, y_train)
    rid3 = Ridge(alpha=100).fit(X_train, y_train)

    las1 = Lasso(alpha=0.01).fit(X_train, y_train)
    las2 = Lasso(alpha=1).fit(X_train, y_train)
    las3 = Lasso(alpha=100).fit(X_train, y_train)

    fig1 = plt.figure(figsize=(10, 6))
    plt.plot(lir.coef_, '^', label='Alpha=0')
    plt.plot(rid1.coef_, 'v', label='Alpha=0.01')
    plt.plot(rid2.coef_, '>', label='Alpha=1')
    plt.plot(rid3.coef_, '<', label='Alpha=100')
    plt.title('Ridge Regression')
    plt.legend(loc=4)
    plt.show()

    fig2 = plt.figure(figsize=(10, 6))
    plt.plot(lir.coef_, '^', label='Alpha=0')
    plt.plot(las1.coef_, 'v', label='Alpha=0.01')
    plt.plot(las2.coef_, '>', label='Alpha=1')
    plt.plot(las3.coef_, '<', label='Alpha=100')
    plt.title('Lasso Regression')
    plt.legend(loc=4)
    plt.show()

def polynomial():
    x = np.array([i*np.pi/180 for i in range(60,300,3)])
    y = np.sin(x) + np.random.normal(0,0.15,len(x))

    x = x[:, np.newaxis]
    y = y[:, np.newaxis]

    fig = plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'k.')

    linreg = LinearRegression().fit(x, y)
    y_preds = linreg.predict(x)
    plt.plot(x, y_preds, 'b-', label='Degree = 1')

    polreg3 = PolynomialFeatures(degree=3)
    X = polreg3.fit_transform(x)
    linreg3 = LinearRegression().fit(X, y)
    y_preds = linreg3.predict(X)
    plt.plot(x, y_preds, 'g-', label='Degree = 3')

    polreg7 = PolynomialFeatures(degree=7)
    X = polreg7.fit_transform(x)
    linreg7 = LinearRegression().fit(X, y)
    y_preds = linreg7.predict(X)
    plt.plot(x, y_preds, 'r-', label='Degree = 7')

    polreg15 = PolynomialFeatures(degree=15)
    X = polreg15.fit_transform(x)
    linreg15 = LinearRegression().fit(X, y)
    y_preds = linreg15.predict(X)
    plt.plot(x, y_preds, 'y-', label='Degree = 15')

    plt.title('Polynomial Features')
    plt.legend()
    plt.show()

def logistic_reg():

    iris = load_iris()
    X = iris.data[:, :2]
    Y = iris.target

    logreg = LogisticRegression(C=1e5)
    logreg.fit(X, Y)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.02

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.brg, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.brg, marker='.')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title('Logistic Regression with 3 classes')
    plt.show()

logistic_reg()




