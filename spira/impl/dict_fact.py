import itertools
import numpy as np
import scipy
import scipy.sparse as sp
from numba import jit
from scipy import linalg

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state, gen_batches
from spira.impl.dataset import mean
from spira.impl.dict_fact_fast import _update_dict_fast, _online_dl_fast
from spira.impl.matrix_fact_fast import _predict
from spira.metrics import rmse

import scipy.linalg as sp_linalg


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
        P = np.zeros((n_rows, self.n_components), order='C')

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
                           order='C')
        self.B_ = np.zeros((self.n_components, n_cols), order="F")

        self.seen_rows_ = 0
        self.seen_cols_ = np.zeros(n_cols, dtype=np.int32)
        self.n_iter_ = 0

        if self.normalize:
            (X, self.row_mean_, self.col_mean_) = csr_center_data(X)

        _online_dl(X.data, X.indices, X.indptr, n_rows, n_cols,
                   float(self.alpha), float(self.learning_rate),
                   self.A_, self.B_,
                   self.seen_rows_, self.seen_cols_, self.n_iter_,
                   self.P_, self.Q_,
                   self.fit_intercept,
                   self.n_epochs,
                   self.batch_size,
                   random_state,
                   self.verbose,
                   self._callback)
        _online_refit(X.data, X.indices, X.indptr,
                      n_cols,
                      self.alpha, self.P_, self.Q_)
        self._callback()

    def _callback(self):
        if self.callback is not None:
            self.callback(self)

    def predict(self, X):
        X = sp.csr_matrix(X)
        out = np.zeros_like(X.data)
        _predict(out, X.indices, X.indptr, self.P_, np.asfortranarray(self.Q_))

        if self.normalize:
            for i in range(X.shape[0]):
                out[X.indptr[i]:X.indptr[i + 1]] += self.row_mean_[i]
            out += self.col_mean_.take(X.indices, mode='clip')

        return sp.csr_matrix((out, X.indices, X.indptr), shape=X.shape)

    def score(self, X):
        X = sp.csr_matrix(X)
        X_pred = self.predict(X)
        return rmse(X, X_pred)


# @jit("void(f8[:], u8[:], u8[:], u8,"
#      "f8, f8[:, ::1], f8[::1, :])")
def _online_refit(X_data, X_indices, X_indptr, n_cols,
                  alpha, P, Q):
    n_rows = len(X_indptr) - 1
    for j in range(n_rows):
        idx_range = slice(X_indptr[j], X_indptr[j + 1])
        idx = X_indices[idx_range]
        y = X_data[idx_range]
        col_nnz = len(idx)
        if col_nnz != 0:
            P[j] = _solve_cholesky(Q[:, idx].T,
                                   y,
                                   2 * alpha * col_nnz / n_cols)


def _update_code(X_data, X_indices, X_indptr, n_cols, n_rows,
                 alpha, learning_rate,
                 A, B, seen_rows, seen_cols,
                 P, Q, row_batch):
    idx_list = []
    row_nnz_list = []
    for j in row_batch:
        idx = X_indices[X_indptr[j]:X_indptr[j + 1]]
        col_nnz = len(idx)
        if col_nnz != 0:
            idx_list.append(idx)
            row_nnz_list.append(j)
            x = X_data[X_indptr[j]:X_indptr[j + 1]]
            P[j] = _solve_cholesky(Q[:, idx].T, x,
                                   2 * alpha * col_nnz / n_cols)
            seen_rows += 1
            seen_cols[idx] += 1
            w_B = np.power(seen_cols[idx], - learning_rate)[
                  np.newaxis, :]
            B[:, idx] *= 1 - w_B
            B[:, idx] += np.outer(P[j], x) * w_B

    len_row_nnz = len(row_nnz_list)

    if len_row_nnz >= 1:
        w_A = pow(seen_rows, -learning_rate)
        A *= 1 - w_A * len_row_nnz
        A += P[row_nnz_list].T.dot(P[row_nnz_list]) * w_A
        if len_row_nnz > 1:
            idx = np.unique(np.concatenate(idx_list))
        elif len_row_nnz == 1:
            idx = idx_list[0]
    return idx, seen_rows


# @jit("void(f8[:], u8[:], u8[:], u8,"
#      "f8, f8,"
#      "f8[:, :], f8[:, :], u8, u8[:], u8,"
#      "f8[:, ::1], f8[::1, :],"
#      "i1, u8, u8, u8, i1,"
#      "pyobject)")
def _online_dl(X_data, X_indices, X_indptr, n_cols, n_rows,
               alpha, learning_rate,
               A, B, seen_rows, seen_cols, n_iter,
               P, Q,
               fit_intercept, n_epochs, batch_size, random_state, verbose,
               callback):
    n_rows = len(X_indptr) - 1
    n_components = Q.shape[0]

    row_range = np.arange(n_rows)

    norm = np.zeros(n_components)
    if not fit_intercept:
        components_range = np.arange(n_components)
    else:
        components_range = np.arange(1, n_components)

    for _ in range(n_epochs):
        batches = gen_batches(n_rows, batch_size)
        np.random.shuffle(row_range)
        for batch in batches:
            row_batch = row_range[batch]

            idx, seen_rows = _update_code(X_data, X_indices, X_indptr, n_cols,
                                          n_rows,
                                          alpha, learning_rate,
                                          A, B, seen_rows, seen_cols,
                                          P, Q, row_batch)

            if len(idx) > 0:
                Q_idx = Q[:, idx]
                R = B[:, idx] - np.dot(A, Q_idx)
                Q_idx = np.asfortranarray(Q_idx)

                random_state.shuffle(components_range)
                _update_dict_fast(
                        Q_idx,
                        A,
                        R,
                        fit_intercept,
                        components_range,
                        norm)
                Q[:, idx] = Q_idx

            if verbose:
                if n_iter % (1000 / batch_size) == 0:
                    print("Iteration %i" % (n_iter * batch_size))
                    # callback()
            n_iter += 1


def _sample(X_data, X_indices, X_indptr, n_cols, row_batch):
    len_batch = len(row_batch)
    if len_batch == 1:
        j = row_batch[0]
        y = X_data[X_indptr[j]:X_indptr[j + 1]][np.newaxis, :]
        idx = X_indices[X_indptr[j]:X_indptr[j + 1]]
        counts = np.ones(len(idx), dtype='i4')
    else:
        counts = np.zeros(n_cols, dtype='i4')
        position_array = np.zeros(n_cols, dtype='i4')

        idx = np.array([], dtype='i4')
        for j in row_batch:
            counts[X_indices[X_indptr[j]:X_indptr[j + 1]]] += 1
            idx = np.union1d(idx, X_indices[X_indptr[j]:X_indptr[j + 1]])
        idx = np.sort(idx)
        counts = counts[idx]
        position_array[idx] = np.arange(len(idx))

        y = np.zeros((len_batch, len(idx)))
        for j_idx, j in enumerate(row_batch):
            position = position_array[X_indices[X_indptr[j]:
            X_indptr[j + 1]]]
            y[j_idx, position] = X_data[X_indptr[j]:X_indptr[j + 1]]
    return y, idx, counts


def _solve_cholesky(X, y, alpha):
    _, n_features = X.shape
    A = X.T.dot(X)
    Xy = X.T.dot(y)
    A.flat[::n_features + 1] += alpha
    return linalg.solve(A, Xy, sym_pos=True,
                        overwrite_a=True).T


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


# TODO: move to test to compare with _update_dict_fast
def _update_dict(Q, A, R,
                 fit_intercept, random_seed):
    n_components = Q.shape[0]
    norm = np.sqrt(np.sum(Q ** 2, axis=1))

    ger, = linalg.get_blas_funcs(('ger',), (A, Q))
    # Intercept on first column
    if fit_intercept:
        components_range = np.arange(1, n_components)
    else:
        components_range = np.arange(n_components)
    np.random.shuffle(components_range)
    for j in components_range:
        ger(1.0, A[j], Q[j], a=R, overwrite_a=True)
        Q[j] = R[j] / A[j, j]
        new_norm = np.sqrt(np.sum(Q[j] ** 2))
        if new_norm > norm[j]:
            Q[j] /= new_norm / norm[j]
        ger(-1.0, A[j], Q[j], a=R, overwrite_a=True)

    return Q
