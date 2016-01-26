# Author: Mathieu Blondel
# License: BSD

import sys

import matplotlib.pyplot as plt
import numpy as np

from spira.completion import Dummy
from spira.cross_validation import ShuffleSplit
from spira.cross_validation import cross_val_score
from spira.datasets import load_movielens
from spira.impl.dict_fact import DictMF

try:
    version = sys.argv[1]
except:
    version = "10m"

X = load_movielens(version)
print(X.shape)

alphas = np.logspace(-2, 2, 15)
mf_scores = []

cv = ShuffleSplit(n_iter=3, train_size=0.75, random_state=0)

for alpha in alphas:
    # mf = ExplicitMF(n_components=30, max_iter=10, alpha=alpha)
    mf = DictMF(n_components=30, n_epochs=3, alpha=alpha, verbose=1,
                batch_size=100, normalize=True,
                fit_intercept=True,
                random_state=0,
                learning_rate=1)
    mf_scores.append(cross_val_score(mf, X, cv))

# Array of size n_alphas x n_folds.
mf_scores = np.array(mf_scores)

dummy = Dummy()
dummy_scores = cross_val_score(dummy, X, cv)

dummy = Dummy(axis=0)
dummy_scores2 = cross_val_score(dummy, X, cv)

plt.figure()
plt.plot(alphas, mf_scores.mean(axis=1), label="Matrix Factorization")
plt.plot(alphas, [dummy_scores.mean()] * len(alphas), label="User mean")
plt.plot(alphas, [dummy_scores2.mean()] * len(alphas), label="Movie mean")
plt.xlabel("alpha")
plt.xscale("log")
plt.ylabel("RMSE")
plt.legend()
plt.savefig('cross_val.pdf')
