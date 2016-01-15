from math import pow

import numpy as np
import scipy.sparse as sp
from numpy.linalg import LinAlgError
from numpy.testing import assert_array_almost_equal
from scipy import linalg

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state, gen_batches
from spira.impl.dict_fact_fast import _update_dict_fast, _update_code_fast
from spira.impl.matrix_fact_fast import _predict
from spira.metrics import rmse


class DictMF(BaseEstimator):
    def __init__(self, alpha=1.0, learning_rate=1.,
                 n_components=30, n_epochs=2,
                 normalize=False,
                 fit_intercept=False,
                 callback=None, random_state=None, verbose=0,
                 batch_size=1):
        self.batch_size = batch_size
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.callback = callback
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.n_components = n_components
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.verbose = verbose

    def _init(self, X, random_state):
        n_rows, n_cols = X.shape

        # P code
        P = np.zeros((self.n_components, n_rows), order='F')

        # Q dictionary
        Q = np.empty((self.n_components, n_cols), order='F')
        if self.fit_intercept:
            # Intercept on first line
            Q[0] = 1
            Q[1:] = random_state.randn(self.n_components - 1,
                                       n_cols)  # + mean(X, 0)[np.newaxis, :]
        else:
            Q[:] = random_state.randn(self.n_components,
                                      n_cols)  # + mean(X, 0)[np.newaxis, :]

        S = np.sqrt(np.sum(Q ** 2, axis=1))
        Q /= S[:, np.newaxis]

        return P, Q

    def _refit_code(self, X):
        X = sp.csr_matrix(X, dtype=np.float64)
        _, n_cols = X.shape
        _online_refit(X.data, X.indices, X.indptr,
                      n_cols,
                      self.alpha, self.P_, self.Q_)

    def fit(self, X, y=None):
        X = sp.csr_matrix(X, dtype=np.float64)
        n_rows, n_cols = X.shape

        random_state = check_random_state(self.random_state)

        self.P_, self.Q_ = self._init(X, random_state)

        self.A_ = np.zeros((self.n_components, self.n_components),
                           order='F')
        self.B_ = np.zeros((self.n_components, n_cols), order="F")

        self.counter_ = np.zeros(n_cols + 1, dtype='int')

        if self.normalize:
            (X, self.row_mean_, self.col_mean_) = csr_center_data(X)

        _online_dl(X,
                   float(self.alpha), float(self.learning_rate),
                   self.A_, self.B_,
                   self.counter_,
                   self.P_, self.Q_,
                   self.fit_intercept,
                   self.n_epochs,
                   self.batch_size,
                   random_state,
                   self.verbose,
                   self._callback)
        self._callback()
        _online_refit(X, self.alpha, self.P_, self.Q_, self.verbose,
                      self._callback)

    def _callback(self):
        if self.callback is not None:
            self.callback(self)

    def predict(self, X):
        X = sp.csr_matrix(X)
        out = np.zeros_like(X.data)
        _predict(out, X.indices, X.indptr, self.P_.T,
                 self.Q_)

        if self.normalize:
            for i in range(X.shape[0]):
                out[X.indptr[i]:X.indptr[i + 1]] += self.row_mean_[i]
            out += self.col_mean_.take(X.indices, mode='clip')

        return sp.csr_matrix((out, X.indices, X.indptr), shape=X.shape)

    def score(self, X):
        X = sp.csr_matrix(X)
        X_pred = self.predict(X)
        return rmse(X, X_pred)


def _online_refit(X, alpha, P, Q, verbose, callback):
    row_range = X.getnnz(axis=1).nonzero()[0]
    n_cols = X.shape[1]
    n_components = P.shape[0]

    for ii, j in enumerate(row_range):
        nnz = X.indptr[j + 1] - X.indptr[j]
        idx = X.indices[X.indptr[j]:X.indptr[j + 1]]
        x = X.data[X.indptr[j]:X.indptr[j + 1]]

        Q_idx = Q[:, idx]
        C = Q_idx.dot(Q_idx.T)
        Qx = Q_idx.dot(x)
        C.flat[::n_components + 1] += 2 * alpha * nnz / n_cols
        try:
            P[:, j] = linalg.solve(C, Qx, sym_pos=True,
                                   overwrite_a=True, check_finite=False)
        except LinAlgError:
            print('Linalg error')
            P[:, j] = 0
        if verbose:
            if ii % 5000 == 0:
                print("Refit iteration %i" % ii)
                callback()


def _update_code_slow(X, alpha, learning_rate,
                      A, B, counter,
                      P, Q, row_batch):
    len_batch = len(row_batch)
    n_cols = X.shape[1]
    n_components = P.shape[0]
    for j in row_batch:
        nnz = X.indptr[j + 1] - X.indptr[j]
        idx = X.indices[X.indptr[j]:X.indptr[j + 1]]
        x = X.data[X.indptr[j]:X.indptr[j + 1]]

        Q_idx = Q[:, idx]
        C = Q_idx.dot(Q_idx.T)
        Qx = Q_idx.dot(x)
        C.flat[::n_components + 1] += 2 * alpha * nnz / n_cols
        P[:, j] = linalg.solve(C, Qx, sym_pos=True,
                               overwrite_a=True, check_finite=False)

        counter[1][idx] += 1
        w_B = np.power(counter[1][idx], - learning_rate)[np.newaxis, :]
        B[:, idx] *= 1 - w_B
        B[:, idx] += np.outer(P[:, j], x) * w_B

    X_indices = np.concatenate([X.indices[X.indptr[j]:X.indptr[j + 1]]
                                for j in row_batch])

    if len_batch > 1:
        idx = np.unique(X_indices)
    else:
        idx = X_indices

    counter[0] += len_batch
    w_A = pow(counter[0], -learning_rate)
    A *= 1 - w_A * len_batch
    A += P[:, row_batch].T.dot(P[:, row_batch]) * w_A

    return idx


def _update_dict(X, fit_intercept,
                 A, B, P, Q_idx, idx, components_range, norm):
    R = B[:, idx] - np.dot(Q_idx.T, A).T

    _update_dict_fast(
            Q_idx,
            A,
            R,
            fit_intercept,
            components_range,
            norm)

    Q[:, idx] = Q_idx


def _online_dl(X,
               alpha, learning_rate,
               A, B, counter,
               P, Q,
               fit_intercept, n_epochs, batch_size, random_state, verbose,
               callback):
    row_nnz = X.getnnz(axis=1)
    max_idx_size = row_nnz.max() * batch_size
    row_range = row_nnz.nonzero()[0]

    n_rows, n_cols = X.shape
    n_components = P.shape[0]
    max_idx = n_cols
    Q_idx = np.zeros((n_components, max_idx_size), order='F')
    P_batch = np.zeros((n_components, batch_size), order='F')
    C = np.zeros((n_components, n_components), order='F')
    idx_mask = np.zeros(n_cols, dtype='i1')
    idx_concat = np.zeros(max_idx, dtype='int')

    norm = np.zeros(n_components)

    if not fit_intercept:
        components_range = np.arange(n_components)
    else:
        components_range = np.arange(1, n_components)

    for _ in range(n_epochs):
        random_state.shuffle(row_range)
        batches = gen_batches(len(row_range), batch_size)
        for batch in batches:
            row_batch = row_range[batch]
            last = _update_code_fast(X.data, X.indices,
                                     X.indptr, n_rows, n_cols,
                                     alpha, learning_rate,
                                     A, B,
                                     counter,
                                     P, Q,
                                     row_batch,
                                     Q_idx,
                                     P_batch,
                                     C,
                                     idx_mask,
                                     idx_concat)
            random_state.shuffle(components_range)
            _update_dict(X, fit_intercept,
                         A, B, P, Q_idx, idx_concat[:last], components_range,
                         norm)

            if verbose:
                if counter[0] % 10000 == 0:
                    print("Iteration %i" % (counter[0]))
                    callback()


def csr_center_data(X, inplace=False):
    if not inplace:
        X = X.copy()

    acc_u = np.zeros(X.shape[0])
    acc_m = np.zeros(X.shape[1])

    n_u = X.getnnz(axis=1)
    n_m = X.getnnz(axis=0)
    n_u[n_u == 0] = 1
    n_m[n_m == 0] = 1
    for i in range(2):
        w_u = X.sum(axis=1).A[:, 0] / n_u
        for i, (left, right) in enumerate(zip(X.indptr[:-1], X.indptr[1:])):
            X.data[left:right] -= w_u[i]
        w_m = X.sum(axis=0).A[0] / n_m
        X.data -= w_m.take(X.indices, mode='clip')
        acc_u += w_u
        acc_m += w_m

    return X, acc_u, acc_m
