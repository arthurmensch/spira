# Author: Mathieu Blondel
# License: BSD

from joblib import load, delayed, Parallel
import time
import sys

import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from spira.datasets import load_movielens
from spira.cross_validation import train_test_split
from spira.completion import ExplicitMF, DictMF
from spira.preprocessing import StandardScaler


def sqnorm(M):
    m = M.ravel()
    return np.dot(m, m)


class Callback(object):
    def __init__(self, X_tr, X_te):
        self.X_tr = X_tr
        self.X_te = X_te
        self.obj = []
        self.rmse = []
        self.times = []
        # self.q_values = []
        self.start_time = time.clock()
        self.test_time = 0

    def __call__(self, mf):
        test_time = time.clock()

        X_pred = mf.predict(self.X_tr)
        loss = sqnorm(X_pred.data - self.X_tr.data) / 2
        regul = mf.alpha * (sqnorm(mf.P_))  # + sqnorm(mf.Q_))
        self.obj.append(loss + regul)

        X_pred = mf.predict(self.X_te)
        rmse = np.sqrt(np.mean((X_pred.data - self.X_te.data) ** 2))
        print(rmse)
        self.rmse.append(rmse)

        self.test_time += time.clock() - test_time
        self.times.append(time.clock() - self.start_time - self.test_time)


try:
    version = sys.argv[1]
except:
    version = "100k"

X = load_movielens(version)
print(X.shape)

X_tr, X_te = train_test_split(X, train_size=0.75, random_state=0)
X_tr = X_tr.tocsr()
X_te = X_te.tocsr()
#
# X_tr = load('/volatile/arthur/spira_data/nf_prize/X_tr.pkl')
# X_te = load('/volatile/arthur/spira_data/nf_prize/X_te.pkl')


# _, X_te = train_test_split(X_te, train_size=0.9, random_state=0)
# X_te = X_te.tocsr()
#
# print(X_tr.shape)


def call(alpha, learning_rate):
    cb = Callback(X_tr, X_te)
    mf = DictMF(n_components=30, n_epochs=4, alpha=alpha, verbose=1,
                normalize=True,
                fit_intercept=True,
                random_state=0,
                learning_rate=learning_rate)

    mf.fit(X_tr)
    X_pred = mf.predict(X_te)
    rmse = np.sqrt(np.mean((X_pred.data - X_te.data) ** 2))
    print('rmse %.2f: %.4f' % (.5, rmse))
    return np.array([alpha, learning_rate, rmse])


res = Parallel(n_jobs=40)(delayed(call)(alpha,
                                        0.5) for alpha in
                          np.logspace(-1, 3, 10)
                          for learning_rate in np.linspace(.5, 1, 4))

res = np.concatenate(res)
np.save(res, 'res')
