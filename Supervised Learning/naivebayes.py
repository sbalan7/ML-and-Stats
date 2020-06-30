from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, random_state=0)

gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train)
print('Gaussian Naive Bayes accuracy: {}'.format(gnb.score(X_test, y_test)))

mnb = MultinomialNB()
y_pred = mnb.fit(X_train, y_train)
print('Multinomial Naive Bayes accuracy: {}'.format(mnb.score(X_test, y_test)))

bnb = BernoulliNB()
y_pred = bnb.fit(X_train, y_train)
print('Bernoulli Naive Bayes accuracy: {}'.format(bnb.score(X_test, y_test)))


