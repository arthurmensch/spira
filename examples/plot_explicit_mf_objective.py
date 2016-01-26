# Author: Mathieu Blondel
# License: BSD

import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from spira.completion import ExplicitMF, DictMF
from spira.cross_validation import train_test_split
from spira.datasets import load_movielens
from spira.impl.dict_fact import csr_center_data


def sqnorm(M):
    m = M.ravel()
    return np.dot(m, m)


class Callback(object):
    def __init__(self, X_tr, X_te, refit=False):
        self.X_tr = X_tr
        self.X_te = X_te
        self.obj = []
        self.rmse = []
        self.times = []
        self.start_time = time.clock()
        self.test_time = 0
        self.refit = refit

    def __call__(self, mf):
        test_time = time.clock()
        if self.refit:
            if mf.normalize:
                if not hasattr(self, 'X_tr_c_'):
                    self.X_tr_c_, _, _ = csr_center_data(X_tr)
                mf._refit_code(self.X_tr_c_)
            else:
                mf._refit_code(self.X_tr)
        X_pred = mf.predict(self.X_tr)
        loss = sqnorm(X_pred.data - self.X_tr.data) / 2
        regul = mf.alpha * (sqnorm(mf.P_))  # + sqnorm(mf.Q_))
        self.obj.append(loss + regul)

        X_pred = mf.predict(self.X_te)
        rmse = np.sqrt(np.mean((X_pred.data - self.X_te.data) ** 2))
        print(rmse)
        self.rmse.append(rmse)

        # print(mf.B_[0])
        # print(mf.counter_)
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

cb = {}
cd_mf = ExplicitMF(n_components=30, max_iter=50, alpha=.05, normalize=True,
                   verbose=1, )
dl_mf_noimpute = DictMF(n_components=30, n_epochs=20, alpha=1, verbose=5,
               batch_size=10, normalize=True,
               fit_intercept=True,
               random_state=0,
               learning_rate=.5,
               impute=False,
               backend='c')
dl_mf_impute = DictMF(n_components=30, n_epochs=20, alpha=.01, verbose=5,
               batch_size=10, normalize=True,
               fit_intercept=True,
               random_state=0,
               learning_rate=1,
               impute=True,
               backend='c')

mf_list = [dl_mf_impute]
for mf in mf_list:
    cb[mf] = Callback(X_tr, X_te, refit=True)
    mf.set_params(callback=cb[mf])
    mf.fit(X_tr)
    print('Done')

# plt.figure()
# plt.plot(cb.times, cb.obj)
# plt.xlabel("CPU time")
# plt.xscale("log")
# plt.ylabel("Objective value")
#
# plt.savefig('objective.pdf')

plt.figure()
label = ['No imputation', 'imputation']
for i, mf in enumerate(mf_list):
    plt.plot(cb[mf].times, cb[mf].rmse, label=label[i])
    np.save(label[i] + 'rmse', cb[mf].rmse)
    np.save(label[i] + 'time', cb[mf].rmse)
plt.legend()
plt.xlabel("CPU time")
plt.xscale("log")
plt.ylabel("RMSE")

plt.savefig('RMSE_%s.pdf' % version)

# plt.figure()
# plt.plot(cb.times, np.row_stack(cb.q_values), marker='o')
# plt.xlabel("CPU time")
# plt.xscale("log")
# plt.ylabel("Q values")

# plt.savefig('Q_values.pdf')
