# Author: Mathieu Blondel
# License: BSD
import datetime
import os
import time
from os.path import expanduser, join

import numpy as np
from joblib import load, delayed, Parallel

from spira.completion import DictMF


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


# try:
#     version = sys.argv[1]
# except:
#     version = "10m"
#
# X = load_movielens(version)
# print(X.shape)
#
# X_tr, X_te = train_test_split(X, train_size=0.75, random_state=0)
# X_tr = X_tr.tocsr()
# X_te = X_te.tocsr()
#
X_tr = load(expanduser('~/spira_data/nf_prize/X_tr.pkl'))
X_te = load(expanduser('~/spira_data/nf_prize/X_te.pkl'))


# X_tr = X_tr.T.tocsr()
# X_te = X_te.T.tocsr()

# _, X_te = train_test_split(X_te, train_size=0.9, random_state=0)
# X_te = X_te.tocsr()
#
# print(X_tr.shape)


def call(n_components, alpha, learning_rate, batch_size):
    cb = Callback(X_tr, X_te)
    mf = DictMF(n_components=n_components, n_epochs=8, alpha=alpha, verbose=10,
                batch_size=batch_size,
                normalize=True,
                fit_intercept=True,
                random_state=0,
                learning_rate=learning_rate)

    start_time = time.time()
    mf.fit(X_tr)
    runtime = time.time() - start_time
    X_pred = mf.predict(X_te)
    rmse = np.sqrt(np.mean((X_pred.data - X_te.data) ** 2))
    print('rmse %i %.2f %.2f %.2f: %.4f' % (n_components,
                                            alpha, learning_rate,
                                            batch_size, rmse))
    return [n_components, alpha, learning_rate, batch_size, runtime, rmse]


timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
output_dir = join(expanduser('~/output/dl_fast'), timestamp)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

res = Parallel(n_jobs=20,
               verbose=10)(delayed(call)(n_components,
                                         alpha,
                                         learning_rate,
                                         batch_size)
                           for n_components in [60]
                           for alpha in np.linspace(0.7, 1.7, 20)
                           for batch_size in np.logspace(4, 4, 1).astype('int')
                           for learning_rate in np.linspace(.75, .75, 1))

res = np.array(res)

np.save(join(output_dir, 'res_nf'), res)
