import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso

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


