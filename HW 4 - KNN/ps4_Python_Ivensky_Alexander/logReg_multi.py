import numpy as np
from sklearn.linear_model import LogisticRegression


# X_train: m x (n + 1)
# y_train: m x 1
# X_test: d x (n + 1)
# y_predict: d x 1
def logReg_multi(X_train, y_train, X_test):
    labels = np.unique(y_train)
    m = X_train.shape[0]
    n = X_train.shape[1] - 1
    y_predict = np.zeros((X_test.shape[0], 1))
    for i in range(labels.shape[0]): # for each class
        y_c = np.array([1 if label == labels[i] else 0 for label in y_train])
        mdl = LogisticRegression(random_state=0).fit(X_train, y_c)
        proba_c = mdl.predict_proba(X_test)[:, 1]
        for j in range(proba_c.shape[0]):
            if proba_c[j] > 0.5:
                y_predict[j] = labels[i]
    return y_predict
        
