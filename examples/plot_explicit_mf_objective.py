# Author: Mathieu Blondel
# License: BSD

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
        regul = mf.alpha * (sqnorm(mf.P_))  #  + sqnorm(mf.Q_))
        self.obj.append(loss + regul)

        X_pred = mf.predict(self.X_te)
        rmse = np.sqrt(np.mean((X_pred.data - self.X_te.data) ** 2))
        print(rmse)
        self.rmse.append(rmse)
        # self.q_values.append(mf.Q_[3, 0:100].copy())

        self.test_time += time.clock() - test_time
        self.times.append(time.clock() - self.start_time - self.test_time)


try:
    version = sys.argv[1]
except:
    version = "10m"

X = load_movielens(version)
# X = X.T
print(X.shape)

X_tr, X_te = train_test_split(X, train_size=0.75, random_state=0)
X_tr = X_tr.tocsr()
X_te = X_te.tocsr()

# mf = ExplicitMF(n_components=30, max_iter=50, alpha=0.1, verbose=1,
#                 callback=cb)
cb = Callback(X_tr, X_te)
mf = DictMF(n_components=30, n_epochs=2, alpha=1, verbose=1,
            batch_size=2,
            callback=cb, normalize=True,
            fit_intercept=True,
            learning_rate=1)

mf.fit(X_tr)

plt.figure()
plt.plot(cb.times, cb.obj)
plt.xlabel("CPU time")
plt.xscale("log")
plt.ylabel("Objective value")

plt.savefig('objective.pdf')

plt.figure()
plt.plot(cb.times, cb.rmse)
plt.xlabel("CPU time")
plt.xscale("log")
plt.ylabel("RMSE")

plt.savefig('RMSE.pdf')

# plt.figure()
# plt.plot(cb.times, np.row_stack(cb.q_values), marker='o')
# plt.xlabel("CPU time")
# plt.xscale("log")
# plt.ylabel("Q values")

plt.savefig('Q_values.pdf')
