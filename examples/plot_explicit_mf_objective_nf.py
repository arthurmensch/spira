# Author: Mathieu Blondel
# License: BSD
import datetime
import os
import time
from os.path import expanduser, join

import matplotlib.pyplot as plt
import numpy as np
from joblib import load

from spira.completion import ExplicitMF, DictMF
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

        self.test_time += time.clock() - test_time
        self.times.append(time.clock() - self.start_time - self.test_time)

X_tr = load(expanduser('~/spira_data/nf_prize/X_tr.pkl'))
# X_tr = X_tr.T.tocsr()
X_te = load(expanduser('~/spira_data/nf_prize/X_te.pkl'))
# X_te = X_te.T.tocsr()

cb = {}
cd_mf = ExplicitMF(n_components=30, max_iter=50, alpha=0.1, verbose=1, )
dl_mf = DictMF(n_components=30, n_epochs=5, alpha=.3, verbose=10,
               batch_size=400, normalize=True,
               fit_intercept=True,
               random_state=0,
               learning_rate=1,
               backend='c')

for mf in [dl_mf]:
    cb[mf] = Callback(X_tr, X_te, refit=(mf is dl_mf))
    mf.set_params(callback=cb[mf])
    mf.fit(X_tr)

timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
output_dir = join(expanduser('~/output/dl_fast'), timestamp)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

plt.figure()
label = {cd_mf: 'CD',
         dl_mf: 'DL'}
for mf in [dl_mf]:
    plt.plot(cb[mf].times, cb[mf].rmse, label=label[mf])
    np.save(join(output_dir, 'rmse' + label[mf]), np.r_[cb[mf].times, cb[mf].rmse])
plt.legend()
plt.xlabel("CPU time")
plt.xscale("log")
plt.ylabel("RMSE")

plt.savefig(join(output_dir, 'RMSE.pdf'))
