import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_curve, roc_auc_score


r = np.random.randint(0, 100)
X, y = make_classification(n_samples=1000, n_classes=2, random_state=r)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=r+1)

lir = LinearRegression()
lir.fit(X_train, y_train)
lir_preds = lir.predict(X_test) 

lor = LogisticRegression(C=1e-4)
lor.fit(X_train, y_train)
lor_preds = lor.predict_proba(X_test)
lor_preds = lor_preds[:, 1]

zero = np.zeros(len(y_test))

lir_fpr, lir_tpr, _ = roc_curve(y_test, lir_preds)
lir_auc = roc_auc_score(y_test, lir_preds)

lor_fpr, lor_tpr, _ = roc_curve(y_test, lor_preds)
lor_auc = roc_auc_score(y_test, lor_preds)

zero_fpr, zero_tpr, _ = roc_curve(y_test, zero)
zero_auc = roc_auc_score(y_test, zero)

fig = plt.figure(figsize=(6, 6))
plt.plot(zero_fpr, zero_tpr, 'g--', label='Random Classifier, AUC = {:.4f}'.format(zero_auc))
plt.plot(lir_fpr, lir_tpr, 'r-', label='Linear Regression, AUC = {:.4f}'.format(lir_auc))
plt.plot(lor_fpr, lor_tpr, 'b-', label='Logistic Regression, AUC = {:.4f}'.format(lor_auc))
plt.legend(frameon=False)
plt.show()

